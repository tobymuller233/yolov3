import torch
import torch.nn as nn
import os
import random

from models.yolo import Model
from models.common import Conv, DWConv
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

from utils.general import (
    LOGGER, 
    intersect_dicts,
    Profile,
)

from utils.loss import ComputeLoss

import wandb
from prettytable import PrettyTable
import omegaconf

from utils.downloads import attempt_download, is_url

from smooth.permute import PermutationManager, compute_tv_loss_for_network
from neumeta.utils import (AverageMeter, EMA, load_checkpoint, print_omegaconf, 
                       sample_coordinates, sample_merge_model, 
                       sample_subset, sample_weights, save_checkpoint, 
                       set_seed, shuffle_coordiates_all, sample_single_model,
                       validate_single, get_cifar10, 
                       get_hypernet, get_optimizer, 
                       parse_args, 
                       weighted_regression_loss)

def print_namespace(opt):
    """
    Print the namespace of the given options.

    :param opt: The options to print.
    """
    # Create a table with PrettyTable
    table = PrettyTable()

    # Define the column names
    table.field_names = ["Key", "Value"]

    # Recursively go through the items and add rows
    def add_items(items, parent_key=""):
        for k, v in items.items():
            current_key = f"{parent_key}.{k}" if parent_key else k
            if isinstance(v, omegaconf.dictconfig.DictConfig):
                # If the value is another dict, recursively add its items
                add_items(dict(v), parent_key=current_key)
            else:
                # If it's a leaf node, add it to the table
                table.add_row([current_key, v])

    # Start adding items from the top-level configuration
    add_items(vars(opt))

    # Print the table
    print(table)

def initialize_wandb(opt):
    """
    Initialize the Weights & Biases tool.

    :param opt: The options used for the training.
    """
    import time
    run_name = f"{time.strftime('%Y%m%d%H%M%S')}-{opt.experiment.name}"
    
    wandb.init(project="ninr_yolov3", name=run_name, config=vars(opt), group='yoloface500kp')

def init_model_dict(opt, LOCAL_RANK, device="cpu"):
    """
    Initializes a dictionary of models for each dimension in the given range, along with ground truth models for the starting dimension.

    Args:
        args: An object containing the arguments for initializing the models.

    Returns:
        dim_dict: A dictionary containing the models for each dimension, along with their corresponding coordinates, keys, indices, size, and ground truth models.
        gt_model_dict: A dictionary containing the ground truth models for the starting dimension.
    """
    dim_dict = {}
    gt_model_dict = {}
    change_layers = {""}
    for dim in opt.dimensions.range:
        # Create a model for the given dimension
        model_cls = create_model(opt.model.type, LOCAL_RANK, opt, device, hidden_dim=dim, path=opt.model.pretrained_path, change_layers=not None).to(device)

        # Sample the coordinates, keys, indices, and size for the model
        ignored_keys = None
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls, ignored_keys=ignored_keys)
        # Add the model, coordinates, keys, indices, size, and key mask to the dictionary
        dim_dict[f"{dim}"] = (model_cls, coords_tensor, keys_list, indices_list, size_list, None)
        # If the dimension is the starting dimension, add the ground truth model to the dictionary

        if dim == opt.dimensions.start:
            print(f"Loading model for dim {dim}")
            model_trained = create_model(opt.model.type, 
                                         LOCAL_RANK,
                                         opt,
                                         device,
                                         hidden_dim=dim, 
                                         path=opt.model.pretrained_path, 
                                         smooth=opt.model.smooth).to(device)
            model_trained.eval()
            
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

def create_model(model_name, LOCAL_RANK, opt, device, hidden_dim=240, path=None, smooth=None, change_layers=None):
    hyp = opt.hyp
    if model_name == "yoloface500kp":
        model = create_model_yolov3(LOCAL_RANK, device, opt, hyp, hidden_dim=hidden_dim, path=path, smooth=smooth, change_layers=change_layers)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

# the modified model of yoloface500kp.yaml
def create_model_yolov3(LOCAL_RANK, device, opt, hyp, hidden_dim=240, path=None, smooth=None, change_layers=None):
    """
    Create a model based on the specified name.

    :param path: Optional path for the model's weights.
    :param opt: The options used for the training.
    :param hyp: The hyperparameters used for the training.
    :param device: The device to use for the model.
    :param LOCAL_RANK: The local rank of the process.
    :param smooth: The smoothing factor to use for the model.
    :param change_layers: The layers to change in the model.(currently it's a str)
    :return: The initialized model.
    """
    if path:
        weights = path
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(path)
        ckpt = torch.load(weights, map_location="cpu")  # load
        model = Model(opt.cfg or ckpt["model"].yaml, ch=3, nc=1, anchors=hyp.get("anchors"), change_layers=change_layers).to(device) # yoloface500kp.yaml
        exclude = ["anchor"] if (opt.cfg or hyp.get("anchors")) else []  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # model.load_state_dict(csd, strict=False)  # load
        if change_layers:
            model.model[21][1].cv1 = Conv(model.model[21][1].cv1.conv.in_channels, hidden_dim, 1, 1)
            model.model[21][1].cv2 = DWConv(hidden_dim, hidden_dim, 3, 1)
            model.model[21][1].cv3 = Conv(hidden_dim, model.model[21][1].cv3.conv.out_channels, 1, 1)
        model.model = load_checkpoint(model.model, csd, prefix="model.")  # load
        LOGGER.info(f"Transferred {len(csd)}/{len(ckpt['model'].float().state_dict())} items from {weights}")
    else:
        model = Model(opt.cfg, ch=3, nc=1, anchors=hyp.get("anchors")).to(device)  # create
    
    if smooth:
        print("Smooth the parameters of the model")
        print("TV original model: ", compute_tv_loss_for_network(model.model, lambda_tv=1.0).item())
        input_tensor = torch.randn(1, 3, 640, 640)
        permute_func = PermutationManager(model.model, input_tensor)
        permute_dict = permute_func.compute_permute_dict()
        model = permute_func.apply_permutations(permute_dict, ignored_keys=[('conv1.weight', 'in_channels'), ('fc.weight', 'out_channels'), ('fc.bias', 'out_channels')])
        print("TV original model: ", compute_tv_loss_for_network(model.model, lambda_tv=1.0).item())

    return model
    pass

def load_checkpoint(model, checkpoint, prefix='module.'):
    """
    Load model weights from a checkpoint file. This function handles checkpoints that may contain keys
    that are either redundant, prefixed, or absent in the model's state_dict.

    :param model: Model instance for which the weights will be loaded.
    :param checkpoint: checkpoint.
    :param prefix: Optional string to handle prefixed state_dict, common in models trained using DataParallel.
    :return: Model with state dict loaded from the checkpoint.
    """
    # If the state_dict is wrapped inside a dictionary under the 'state_dict' key, unpack it.
    state_dict = checkpoint.get('state_dict', checkpoint)

    # If the state_dict keys contain a prefix, remove it.
    if list(state_dict.keys())[0].startswith(prefix):
        state_dict = {key[len(prefix):]: value for key, value in state_dict.items()}

    # Retrieve the state_dict of the model.
    model_state_dict = model.state_dict()

    # Prepare a new state_dict to load into the model, ensuring that only keys that are present in the model
    # and have the same shape are included.
    updated_state_dict = {
        key: value for key, value in state_dict.items()
        if key in model_state_dict and value.shape == model_state_dict[key].shape
    }
    
    unupdated = {
        key: value for key, value in state_dict.items()
        if key not in updated_state_dict
    }
    # Update the original model state_dict.
    model_state_dict.update(updated_state_dict)
    # for key, value in model_state_dict.items():
    #     print(f"Updated {key}: {value.shape}")

    # Load the updated state_dict into the model.
    model.load_state_dict(model_state_dict)

    return model

def train_one_epoch_yolov3(model, train_loader, optimizer, criterion, dim_dict, gt_model_dict, epoch_idx, device="cpu", ema=None, args=None, half=True):
    # Set the model to training mode
    model.train()
    total_loss = 0.0

    # Initialize AverageMeter objects to track the losses
    losses = AverageMeter()
    cls_losses = AverageMeter()
    reg_losses = AverageMeter()
    reconstruct_losses = AverageMeter()

    dt = Profile(), Profile(), Profile()
    loss = torch.zeros(3, device=device)
    # Iterate over the training data
    for batch_idx, (x, target, paths, shapes) in enumerate(train_loader):
        # Zero the gradients
        optimizer.zero_grad()
        # Move the data to the device
        x, target = x.to(device), target.to(device)
        # Choose a random hidden dimension
        hidden_dim = random.choice(args.dimensions.range)
        # Get the model class, coordinates, keys, indices, size, and key mask for the chosen dimension
        model_cls, coords_tensor, keys_list, indices_list, size_list, key_mask = dim_dict[f"{hidden_dim}"]
        # Sample a subset of the coordinates, keys, indices, size, and selected keys
        # ratio 意义何在？
        coords_tensor, keys_list, indices_list, size_list, selected_keys = sample_subset(coords_tensor,
                                                                                         keys_list,
                                                                                         indices_list,
                                                                                         size_list,
                                                                                         key_mask,
                                                                                         ratio=args.ratio)
        # Add noise to the coordinates if specified
        if hasattr(args.training, 'coordinate_noise') and args.training.coordinate_noise > 0.0:
            coords_tensor = coords_tensor + (torch.rand_like(coords_tensor) - 0.5) * args.training.coordinate_noise
        # Sample the weights for the model
        model_cls, reconstructed_weights = sample_weights(model, model_cls,
                                                          coords_tensor, keys_list, indices_list, size_list, key_mask, selected_keys,
                                                          device=device, NORM=args.dimensions.norm)
        model_cls.half() if half else model_cls.float()
        hyp = args.hyp
        nl = model_cls.model[-1].nl  # number of detection layers (used for scaling hyp['obj'])
        nc = 1  # number of classes
        imgsz = args.imgsz  # image size
        hyp["box"] *= 3 / nl  # scale to layers
        hyp["cls"] *= nc / 80 * 3 / nl
        hyp["obj"] *= (imgsz / 640) ** 2 * 3 / nl
        hyp["label_smoothing"] = args.label_smoothing
        model_cls.hyp = hyp
        
        # Compute classification loss
        # cls_loss = criterion(predict, target) 
        with dt[0]:
            x = x.half() if half else x.float()
            x /= 255
            nb, _, height, width = x.shape  # batch size, channels, height, width
        
        # Forward Pass
        with dt[1]:
            pred = model_cls(x)
        
        dynamic_weight = vars(args).get('dynamic_weight', False)
        compute_loss = ComputeLoss(model_cls)
        cls_loss = compute_loss(pred, target, dynamic_weight=dynamic_weight)[0]
                
        # Compute regularization loss
        reg_loss = sum([torch.norm(w, p=2) for w in reconstructed_weights])

        # Compute reconstruction loss if ground truth model is available
        if f"{hidden_dim}" in gt_model_dict:
            gt_model = gt_model_dict[f"{hidden_dim}"]
            gt_selected_weights = [
                w for k, w in gt_model.learnable_parameter.items() if k in selected_keys]

            reconstruct_loss = weighted_regression_loss(
                reconstructed_weights, gt_selected_weights)
        else:
            reconstruct_loss = torch.tensor(0.0)

        # Compute the total loss
        loss = args.hyper_model.loss_weight.ce_weight * cls_loss + args.hyper_model.loss_weight.reg_weight * \
            reg_loss + args.hyper_model.loss_weight.recon_weight * reconstruct_loss

        # Zero the gradients of the updated weights
        for updated_weight in model_cls.parameters():
            updated_weight.grad = None

        # Compute the gradients of the reconstructed weights
        loss.backward(retain_graph=True)
        grad_list = [w.grad for k, w in model_cls.named_parameters() if k[6:] in selected_keys]
        torch.autograd.backward(reconstructed_weights, grad_list)

        # Clip the gradients if specified
        if args.training.get('clip_grad', 0.0) > 0:
            torch.nn.utils.clip_grad_value_(
                model.parameters(), args.training.clip_grad)

        # Update the weights
        optimizer.step()
        # Update the EMA if specified
        if ema:
            ema.update()  # Update the EMA after each training step
        total_loss += loss.item()

        # Update the AverageMeter objects
        losses.update(loss.item())
        cls_losses.update(cls_loss.item())
        reg_losses.update(reg_loss.item())
        reconstruct_losses.update(reconstruct_loss.item())

        # Log the losses and learning rate to wandb
        # if batch_idx % args.experiment.log_interval == 0:
        #     wandb.log({
        #         "Loss": losses.avg,
        #         "Cls Loss": cls_losses.avg,
        #         "Reg Loss": reg_losses.avg,
        #         "Reconstruct Loss": reconstruct_losses.avg,
        #         "Learning rate": optimizer.param_groups[0]['lr']
        #     }, step=batch_idx + epoch_idx * len(train_loader))
        #     # Print the losses and learning rate
        #     print(
        #         f"Iteration {batch_idx}: Loss = {losses.avg:.4f}, Reg Loss = {reg_losses.avg:.4f}, Reconstruct Loss = {reconstruct_losses.avg:.4f}, Cls Loss = {cls_losses.avg:.4f}, Learning rate = {optimizer.param_groups[0]['lr']:.4e}")
    return losses.avg, dim_dict, gt_model_dict
    pass