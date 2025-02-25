# Ultralytics YOLOv3 🚀, AGPL-3.0 license
"""
Train a NeuralMeta Network for YOLOv3 model on a custom dataset. Models and datasets download automatically from the latest YOLOv3 release.

"""

import argparse
import math
import os
import random
import subprocess
import sys
import time
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import torch_pruning as tp
from models.common import Bottleneck3

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.optim import lr_scheduler
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv3 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

import val as validate  # for end-of-epoch mAP
from models.experimental import attempt_load
from models.yolo import Model
from utils.autoanchor import check_anchors
from utils.autobatch import check_train_batch_size
from utils.callbacks import Callbacks
from utils.dataloaders import create_dataloader
from utils.downloads import attempt_download, is_url
from utils.general import (
    LOGGER,
    TQDM_BAR_FORMAT,
    check_amp,
    check_dataset,
    check_file,
    check_git_info,
    check_git_status,
    check_img_size,
    check_requirements,
    check_suffix,
    check_yaml,
    colorstr,
    get_latest_run,
    increment_path,
    init_seeds,
    intersect_dicts,
    labels_to_class_weights,
    labels_to_image_weights,
    methods,
    one_cycle,
    print_args,
    print_mutation,
    strip_optimizer,
    yaml_save,
)
from utils.loggers import Loggers
from utils.loggers.comet.comet_utils import check_comet_resume
from utils.loss import ComputeLoss
from utils.metrics import fitness
from utils.plots import plot_evolve
from utils.torch_utils import (
    EarlyStopping,
    ModelEMA,
    de_parallel,
    select_device,
    smart_DDP,
    smart_optimizer,
    smart_resume,
    torch_distributed_zero_first,
)
from utils.neumeta import (create_model_yolov3, initialize_wandb, 
                           init_model_dict, print_namespace, train_one_epoch_yolov3,
                           validate_single_yolov3_single_cls)

# neumeta
import wandb
from neumeta.utils import (AverageMeter, EMA, load_checkpoint, print_omegaconf, 
                       sample_coordinates, sample_merge_model, 
                       sample_subset, sample_weights, save_checkpoint, 
                       set_seed, shuffle_coordiates_all, sample_single_model,
                       get_cifar10, 
                       get_hypernet, get_optimizer, 
                       parse_args, 
                       weighted_regression_loss)
from omegaconf import OmegaConf

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))
GIT_INFO = check_git_info()

def train_neumeta(hyp, opt, device, callbacks): # hyp is path/to/hyp.yaml or hyp dictionary
    save_dir, epochs, batch_size, weights, single_cls, data, cfg, noval, nosave, workers, freeze = (
        Path(opt.save_dir),
        opt.epochs,
        opt.batch_size,
        opt.weights,
        opt.single_cls,
        opt.data,
        opt.cfg,
        opt.noval,
        opt.nosave,
        opt.workers,
        opt.freeze,
    )
    callbacks.run("on_pretrain_routine_start")
    
    # Directories
    w = save_dir / "weights"  # weights dir
    # (w.parent if evolve else w).mkdir(parents=True, exist_ok=True)  # make dir
    (w).mkdir(parents=True, exist_ok=True)  # make dir
    last, best = w / "last.pt", w / "best.pt"

    # Hyperparameters
    if isinstance(hyp, str):
        with open(hyp, errors="ignore") as f:
            hyp = yaml.safe_load(f)  # load hyps dict
    LOGGER.info(colorstr("hyperparameters: ") + ", ".join(f"{k}={v}" for k, v in hyp.items()))
    opt.hyp = hyp.copy()  # for saving hyps to checkpoints

    # Loggers
    data_dict = None
    if RANK in {-1, 0}:
        loggers = Loggers(save_dir, weights, opt, hyp, LOGGER)  # loggers instance

        # Register actions
        for k in methods(loggers):
            callbacks.register_action(k, callback=getattr(loggers, k))

        # Process custom dataset artifact link
        data_dict = loggers.remote_dataset

    # get model_cls
    # hidden_dim = dimension start
    model_cls = create_model_yolov3(LOCAL_RANK, device, opt, hyp, path=opt.model.pretrained_path, smooth=opt.model.smooth, hidden_dim=opt.dimensions.start, change_layers=(opt.dimensions.start != 240))
    amp = check_amp(model_cls)  # check AMP
    # Image size
    gs = max(int(model_cls.stride.max()), 32)  # grid size (max stride)
    imgsz = check_img_size(opt.imgsz, gs, floor=gs * 2)  # verify imgsz is gs-multiple
    
    # Batch size
    if RANK == -1 and batch_size == -1:  # single-GPU only, estimate best batch size
        batch_size = check_train_batch_size(model, imgsz, amp)
        loggers.on_params_update({"batch_size": batch_size})
    
    cuda = device.type != "cpu"  
    # get dataloader
    with torch_distributed_zero_first(LOCAL_RANK):
        data_dict = data_dict or check_dataset(data)
    train_path, val_path = data_dict["train"], data_dict["val"]
    nc = 1 if single_cls else int(data_dict["nc"])  # number of classes
    names = {0: "item"} if single_cls and len(data_dict["names"]) != 1 else data_dict["names"]  # class names
    train_loader, dataset = create_dataloader(
        train_path,
        imgsz,
        batch_size // WORLD_SIZE,
        gs,
        single_cls,
        hyp=hyp,
        augment=True,
        cache=None if opt.cache == "val" else opt.cache,
        rect=opt.rect,
        rank=LOCAL_RANK,
        workers=workers,
        image_weights=opt.image_weights,
        quad=opt.quad,
        prefix=colorstr("train: "),
        shuffle=True,
        seed=opt.seed,
    )
    labels = np.concatenate(dataset.labels, 0)
    mlc = int(labels[:, 0].max())  # max label class
    assert mlc < nc, f"Label class {mlc} exceeds nc={nc} in {data}. Possible class labels are 0-{nc - 1}"

    # Process 0
    if RANK in {-1, 0}:
        val_loader = create_dataloader(
            val_path,
            imgsz,
            batch_size // WORLD_SIZE * 2,
            gs,
            single_cls,
            hyp=hyp,
            cache=None if noval else opt.cache,
            rect=True,
            rank=-1,
            workers=workers * 2,
            pad=0.5,
            prefix=colorstr("val: "),
        )[0]

        if not opt.resume:
            if not opt.noautoanchor:
                check_anchors(dataset, model=model_cls, thr=hyp["anchor_t"], imgsz=imgsz)  # run AutoAnchor
            model_cls.half().float()  # pre-reduce anchor precision

        callbacks.run("on_pretrain_routine_end", labels, names)
    
    # DDP mode
    # if cuda and RANK != -1:
    #     model = smart_DDP(model)
        
    # get parameters
    checkpoints = model_cls.learnable_parameter
    number_param = len(checkpoints)
    print("Number of keys", number_param)
    print(f"Number of learnable parameters: {number_param}")

    # hypernet
    hyper_model = get_hypernet(opt, number_param, device)
    # initialize EMA
    ema = EMA(hyper_model, decay=opt.hyper_model.ema_decay)
    # Get the criterion, validation criterion, optimizer, and scheduler
    _criterion, val_criterion, optimizer, scheduler = get_optimizer(opt, hyper_model)
    # criterion is task-specific
    criterion = weighted_regression_loss
    # Initialize the starting epoch and best accuracy
    start_epoch = 0
    best_acc = 0.0
    
    # Create the directory to save the model
    os.makedirs(opt.training.save_model_path, exist_ok=True)
     # If specified, load the checkpoint
    if opt.resume_from:
        print(f"Resuming from checkpoint: {opt.resume_from}")
        checkpoint_info = load_checkpoint(opt.resume_from, hyper_model, optimizer, ema)
        start_epoch = checkpoint_info['epoch']
        best_acc = checkpoint_info['best_acc']
        print(f"Resuming from epoch: {start_epoch}, best accuracy: {best_acc*100:.2f}%")
        # Note: If there are more elements to retrieve, do so here.
    
    # If not testing, initialize wandb, the model dictionary, and the ground truth model dictionary
    if opt.test == False:
        initialize_wandb(opt)
        dim_dict, gt_model_dict = init_model_dict(opt, LOCAL_RANK, device)
        dim_dict = shuffle_coordiates_all(dim_dict)     # 这里会create mask
        selected_dim_sum = 0
        # Iterate over the epochs
        for epoch in range(start_epoch, opt.experiment.num_epochs):
            
            # Train the model for one epoch
            train_loss, dim_dict, gt_model_dict, selected_dim_sum = train_one_epoch_yolov3(hyper_model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx=epoch, device=device, ema=ema, args=opt, half=False, selected_dim_sum=selected_dim_sum)
            # Step the scheduler
            scheduler.step()

            # Print the training loss and learning rate
            print(f"Epoch [{epoch+1}/{opt.experiment.num_epochs}], Training Loss: {train_loss:.4f}, Learning Rate: {scheduler.get_last_lr()[0]:.6f}")

            # If it's time to evaluate the model
            if (epoch + 1) % opt.experiment.eval_interval == 0:
                # If EMA is specified, apply it
                if ema:
                    ema.apply()
                    
                # Sample the merged model
                model_cls = sample_merge_model(hyper_model, model_cls, opt)
                model_cls.names = names
                # Validate the merged model
                val_loss, val_mp, val_mr, val_map50, val_map = validate_single_yolov3_single_cls(model_cls, val_loader, val_criterion, args=opt, device=device, plots=True, save_dir=save_dir)
                
                # If EMA is specified, restore the original weights
                if ema:
                    ema.restore()  # Restore the original weights

                log_val_loss = sum(val_loss) / len(val_loss)
                # Log the validation loss and accuracy to wandb
                wandb.log({
                    "Validation Loss": log_val_loss,
                    "Validation mean precision": val_mp,
                    "Validation mean recall": val_mr,
                    "Validation mAP@50": val_map50,
                    "Validation mAP": val_map,
                    "Averate dim": selected_dim_sum / opt.experiment.eval_interval,
                })
                
                selected_dim_sum = 0
                
                # Print the validation loss and accuracy
                # print(f"Epoch [{epoch+1}/{opt.experiment.num_epochs}], Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
                print(f"Epoch [{epoch+1}/{opt.experiment.num_epochs}], Validation Loss: {log_val_loss:.4f}, Validation mean precision: {val_mp:.4f}, Validation mean recall: {val_mr:.4f}, Validation mAP@50: {val_map50:.4f}, Validation mAP: {val_map:.4f}")
                
                # Save the checkpoint if the accuracy is better than the previous best
                if val_map50 > best_acc:
                    best_acc = val_map50
                    save_checkpoint(f"{opt.training.save_model_path}/{opt.experiment.name}_best_{opt.ratio}.pth",hyper_model,optimizer,ema,epoch,best_acc)
                    print(f"Checkpoint saved at epoch {epoch} with accuracy: {best_acc*100:.2f}%")
        wandb.finish()
    # If testing, iterate over the hidden dimensions and test the model
    else:
        initialize_wandb(opt)
        for hidden_dim in range(1, 65):
            # Create a model for the given hidden dimension
            model = create_model(args.model.type, 
                                 hidden_dim=hidden_dim, 
                                 path=args.model.pretrained_path, 
                                 smooth=args.model.smooth).to(device)

            # If EMA is specified, apply it
            if ema:
                print("Applying EMA")
                ema.apply()
                
            # Sample the merged model
            accumulated_model = sample_merge_model(hyper_model, model, args, K=100)

            # Validate the merged model
            val_loss, acc = validate_single(accumulated_model, val_loader, val_criterion, args=args)
            
            # If EMA is specified, restore the original weights after applying EMA
            if ema:
                ema.restore()  # Restore the original weights after applying EMA
            
            # Save the model
            save_name = os.path.join(args.training.save_model_path, f"cifar10_{accumulated_model.__class__.__name__}_dim{hidden_dim}_single.pth")
            torch.save(accumulated_model.state_dict(),save_name)
            wandb.log({
                    "Validation Loss": val_loss,
                    "Validation Accuracy": acc
                }, step= int(hidden_dim / args.dimensions.start * 100))
            # Print the results
            print(f"Test using model {args.model}: hidden_dim {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%")
            
            # Define the directory and filename structure
            filename = f"cifar10_results_{args.experiment.name}.txt"
            filepath = os.path.join(args.training.save_model_path, filename)

            # Write the results. 'a' is used to append the results; a new file will be created if it doesn't exist.
            with open(filepath, "a") as file:
                file.write(f"Hidden_dim: {hidden_dim}, Validation Loss: {val_loss:.4f}, Validation Accuracy: {acc*100:.2f}%\n")
        wandb.finish()
    # Print message
    print("Training finished.")
    pass

def parse_opt(known=False):
    """
    Parse command line arguments for configuring the training of a YOLO model.

    Args:
        known (bool): Flag to parse known arguments only, defaults to False.

    Returns:
        (argparse.Namespace): Parsed command line arguments.

    Examples:
        ```python
        options = parse_opt()
        print(options.weights)
        ```

    Notes:
        * The default weights path is 'yolov3-tiny.pt'.
        * Set `known` to True for parsing only the known arguments, useful for partial arguments.

    References:
        * Models: https://github.com/ultralytics/yolov5/tree/master/models
        * Datasets: https://github.com/ultralytics/yolov5/tree/master/data
        * Training Tutorial: https://docs.ultralytics.com/yolov5/tutorials/train_custom_data
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "yolov3-tiny.pt", help="initial weights path")
    parser.add_argument("--cfg", type=str, default="", help="model.yaml path")
    parser.add_argument("--data", type=str, default=ROOT / "data/coco128.yaml", help="dataset.yaml path")
    parser.add_argument("--hyp", type=str, default=ROOT / "data/hyps/hyp.scratch-low.yaml", help="hyperparameters path")
    parser.add_argument("--epochs", type=int, default=100, help="total training epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="total batch size for all GPUs, -1 for autobatch")
    parser.add_argument("--imgsz", "--img", "--img-size", type=int, default=640, help="train, val image size (pixels)")
    parser.add_argument("--rect", action="store_true", help="rectangular training")
    parser.add_argument("--resume", nargs="?", const=True, default=False, help="resume most recent training")
    parser.add_argument("--nosave", action="store_true", help="only save final checkpoint")
    parser.add_argument("--noval", action="store_true", help="only validate final epoch")
    parser.add_argument("--noautoanchor", action="store_true", help="disable AutoAnchor")
    parser.add_argument("--noplots", action="store_true", help="save no plot files")
    parser.add_argument("--evolve", type=int, nargs="?", const=300, help="evolve hyperparameters for x generations")
    parser.add_argument("--bucket", type=str, default="", help="gsutil bucket")
    parser.add_argument("--cache", type=str, nargs="?", const="ram", help="image --cache ram/disk")
    parser.add_argument("--image-weights", action="store_true", help="use weighted image selection for training")
    parser.add_argument("--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu")
    parser.add_argument("--multi-scale", action="store_true", help="vary img-size +/- 50%%")
    parser.add_argument("--single-cls", action="store_true", help="train multi-class data as single-class")
    parser.add_argument("--optimizer", type=str, choices=["SGD", "Adam", "AdamW"], default="SGD", help="optimizer")
    parser.add_argument("--sync-bn", action="store_true", help="use SyncBatchNorm, only available in DDP mode")
    parser.add_argument("--workers", type=int, default=8, help="max dataloader workers (per RANK in DDP mode)")
    parser.add_argument("--project", default=ROOT / "runs/train", help="save to project/name")
    parser.add_argument("--name", default="exp", help="save to project/name")
    parser.add_argument("--exist-ok", action="store_true", help="existing project/name ok, do not increment")
    parser.add_argument("--quad", action="store_true", help="quad dataloader")
    parser.add_argument("--cos-lr", action="store_true", help="cosine LR scheduler")
    parser.add_argument("--label-smoothing", type=float, default=0.0, help="Label smoothing epsilon")
    parser.add_argument("--patience", type=int, default=100, help="EarlyStopping patience (epochs without improvement)")
    parser.add_argument("--freeze", nargs="+", type=int, default=[0], help="Freeze layers: backbone=10, first3=0 1 2")
    parser.add_argument("--save-period", type=int, default=-1, help="Save checkpoint every x epochs (disabled if < 1)")
    parser.add_argument("--seed", type=int, default=0, help="Global training seed")
    parser.add_argument("--local_rank", type=int, default=-1, help="Automatic DDP Multi-GPU argument, do not modify")
    parser.add_argument("--dynamic-weight", action="store_true", help="Dynamic weight in computing loss")

    # Logger arguments
    parser.add_argument("--entity", default=None, help="Entity")
    parser.add_argument("--upload_dataset", nargs="?", const=True, default=False, help='Upload data, "val" option')
    parser.add_argument("--bbox_interval", type=int, default=-1, help="Set bounding-box image logging interval")
    parser.add_argument("--artifact_alias", type=str, default="latest", help="Version of dataset artifact to use")

    # neumeta arguments
    parser.add_argument("--neumeta-cfg", type=str, required=True, 
                        help="neumeta configuration yaml")
    parser.add_argument('--ratio', type=float, default=1.0,
                        help='Ratio used for training purposes')
    parser.add_argument('--resume_from', type=str,
                        help='Checkpoint file path to resume training from')
    parser.add_argument('--load_from', type=str,
                        help='Checkpoint file path to load')
    parser.add_argument('--test_result_path', type=str,
                        help='Path to save the test result')
    parser.add_argument('--test', action='store_true',
                        default=False, help='Test the model')

    parse_result = parser.parse_known_args()[0] if known else parser.parse_args()
    neumeta_config = OmegaConf.load(parse_result.neumeta_cfg)

    if neumeta_config.get("base_config", None):
        print("Loading base config from " + neumeta_config.base_config)
        base_config = OmegaConf.load(neumeta_config.base_config)
        neumeta_config = OmegaConf.merge(base_config, neumeta_config)
    
    cli_args = {k: v for k, v in vars(parse_result).items()}
    config = OmegaConf.merge(neumeta_config, cli_args)
    
    if len(config.dimensions.range) == 2:
        interval = config.dimensions.get('interval', 1)
        config.dimensions.range = list(
            range(config.dimensions.range[0], config.dimensions.range[1] + 1, interval))
    # return parser.parse_known_args()[0] if known else parser.parse_args()
    # return parse_result
    args = argparse.Namespace(**config)
    return args


def main(opt, callbacks=Callbacks()):
    
    if RANK in {-1, 0}:
        print_args(vars(opt))
        check_git_status()
        check_requirements(ROOT / "requirements.txt")

    # do check
    if opt.resume and not check_comet_resume(opt) and not opt.evolve:
        last = Path(check_file(opt.resume) if isinstance(opt.resume, str) else get_latest_run())
        opt_yaml = last.parent.parent / "opt.yaml"  # train options yaml
        opt_data = opt.data  # original dataset
        if opt_yaml.is_file():
            with open(opt_yaml, errors="ignore") as f:
                d = yaml.safe_load(f)
        else:
            d = torch.load(last, map_location="cpu")["opt"]
        opt = argparse.Namespace(**d)  # replace
        opt.cfg, opt.weights, opt.resume = "", str(last), True  # reinstate
        if is_url(opt_data):
            opt.data = check_file(opt_data)  # avoid HUB resume auth timeout
    else:
        opt.data, opt.cfg, opt.hyp, opt.weights, opt.project = (
            check_file(opt.data),
            check_yaml(opt.cfg),
            check_yaml(opt.hyp),
            str(opt.weights),
            str(opt.project),
        )  # checks
        assert len(opt.cfg) or len(opt.weights), "either --cfg or --weights must be specified"
        if opt.name == "cfg":
            opt.name = Path(opt.cfg).stem  # use model.yaml as name
        opt.save_dir = str(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))

    # DDP mode
    device = select_device(opt.device, batch_size=opt.batch_size)
    if LOCAL_RANK != -1:
        msg = "is not compatible with YOLOv3 Multi-GPU DDP training"
        assert not opt.image_weights, f"--image-weights {msg}"
        assert not opt.evolve, f"--evolve {msg}"
        assert opt.batch_size != -1, f"AutoBatch with --batch-size -1 {msg}, please pass a valid --batch-size"
        assert opt.batch_size % WORLD_SIZE == 0, f"--batch-size {opt.batch_size} must be multiple of WORLD_SIZE"
        assert torch.cuda.device_count() > LOCAL_RANK, "insufficient CUDA devices for DDP command"
        torch.cuda.set_device(LOCAL_RANK)
        device = torch.device("cuda", LOCAL_RANK)
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo")

    # set random seed
    set_seed(opt.experiment.seed)
    
    # Train

    if not opt.evolve:
        # train(opt.hyp, opt, device, callbacks)
        train_neumeta(opt.hyp, opt, device, callbacks)

def run(**kwargs):
    """
    Run the training process for a YOLOv3 model with the specified configurations.

    Args:
        data (str): Path to the dataset YAML file.
        weights (str): Path to the pre-trained weights file or '' to train from scratch.
        cfg (str): Path to the model configuration file.
        hyp (str): Path to the hyperparameters YAML file.
        epochs (int): Total number of training epochs.
        batch_size (int): Total batch size across all GPUs.
        imgsz (int): Image size for training and validation (in pixels).
        rect (bool): Use rectangular training for better aspect ratio preservation.
        resume (bool | str): Resume most recent training if True, or resume training from a specific checkpoint if a string.
        nosave (bool): Only save the final checkpoint and not the intermediate ones.
        noval (bool): Only validate model performance in the final epoch.
        noautoanchor (bool): Disable automatic anchor generation.
        noplots (bool): Do not save any plots.
        evolve (int): Number of generations for hyperparameters evolution.
        bucket (str): Google Cloud Storage bucket name for saving run artifacts.
        cache (str | None): Cache images for faster training ('ram' or 'disk').
        image_weights (bool): Use weighted image selection for training.
        device (str): Device to use for training, e.g., '0' for first GPU or 'cpu' for CPU.
        multi_scale (bool): Use multi-scale training.
        single_cls (bool): Train a multi-class dataset as a single-class.
        optimizer (str): Optimizer to use ('SGD', 'Adam', or 'AdamW').
        sync_bn (bool): Use synchronized batch normalization (only in DDP mode).
        workers (int): Maximum number of dataloader workers (per rank in DDP mode).
        project (str): Location of the output directory.
        name (str): Unique name for the run.
        exist_ok (bool): Allow existing output directory.
        quad (bool): Use quad dataloader.
        cos_lr (bool): Use cosine learning rate scheduler.
        label_smoothing (float): Label smoothing epsilon.
        patience (int): EarlyStopping patience (epochs without improvement).
        freeze (list[int]): List of layers to freeze, e.g., [0] to freeze only the first layer.
        save_period (int): Save checkpoint every 'save_period' epochs (disabled if less than 1).
        seed (int): Global training seed for reproducibility.
        local_rank (int): For automatic DDP Multi-GPU argument parsing, do not modify.

    Returns:
        None

    Example:
        ```python
        from ultralytics import run
        run(data='coco128.yaml', weights='yolov5m.pt', imgsz=320, epochs=100, batch_size=16)
        ```

    Notes:
        - Ensure the dataset YAML file and initial weights are accessible.
        - Refer to the [Ultralytics YOLOv5 repository](https://github.com/ultralytics/yolov5) for model and data configurations.
        - Use the [Training Tutorial](https://docs.ultralytics.com/yolov5/tutorials/train_custom_data) for custom dataset training.
    """
    opt = parse_opt(True)
    for k, v in kwargs.items():
        setattr(opt, k, v)
    main(opt)
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    # print_omegaconf(opt)
    print_namespace(opt)
    main(opt)
