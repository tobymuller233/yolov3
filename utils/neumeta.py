import torch
import torch.nn as nn
import os

from models.yolo import Model
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
)

import wandb
from prettytable import PrettyTable
import omegaconf

from utils.downloads import attempt_download, is_url

from smooth.permute import PermutationManager, compute_tv_loss_for_network

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
    
    wandb.init(project="ninr_yolov3", name=run_name, config=dict(opt), group='yoloface500k')

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
    for dim in opt.dimensions.range:
        # Create a model for the given dimension
        model_cls = create_model("yoloface500k", LOCAL_RANK, opt, device, hidden_dim=dim, path=opt.model.pretrained_path).to(opt.device)

        # Sample the coordinates, keys, indices, and size for the model
        coords_tensor, keys_list, indices_list, size_list = sample_coordinates(model_cls)
        # Add the model, coordinates, keys, indices, size, and key mask to the dictionary
        dim_dict[f"{dim}"] = (model_cls, coords_tensor, keys_list, indices_list, size_list, None)
        # If the dimension is the starting dimension, add the ground truth model to the dictionary
        if dim == args.dimensions.start:
            print(f"Loading model for dim {dim}")
            model_trained = create_model(args.model.type, 
                                         hidden_dim=dim, 
                                         path=args.model.pretrained_path, 
                                         smooth=args.model.smooth).to(device)
            model_trained.eval()
            
            gt_model_dict[f"{dim}"] = model_trained
    return dim_dict, gt_model_dict

def create_model(model_name, LOCAL_RANK, opt, device, hidden_dim=240, path=None, smooth=None):
    hyp = opt.hyp
    if model_name == "yoloface500k":
        model = create_model_yolov3(LOCAL_RANK, device, opt, hyp, hidden_dim=hidden_dim, path=path, smooth=smooth)
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return model

# the modified model of yoloface500kp.yaml
def create_model_yolov3(LOCAL_RANK, device, opt, hyp, hidden_dim=240, path=None, smooth=None):
    """
    Create a model based on the specified name.

    :param path: Optional path for the model's weights.
    :param opt: The options used for the training.
    :param hyp: The hyperparameters used for the training.
    :param device: The device to use for the model.
    :param LOCAL_RANK: The local rank of the process.
    :param smooth: The smoothing factor to use for the model.
    :return: The initialized model.
    """
    if path:
        weights = path
        with torch_distributed_zero_first(LOCAL_RANK):
            weights = attempt_download(path)
        ckpt = torch.load(weights, map_location="cpu")  # load
        model = Model(opt.cfg or ckpt["model"].yaml, ch=3, nc=1, anchors=hyp.get("anchors")).to(device) # yoloface500k.yaml
        exclude = ["anchor"] if (opt.cfg or hyp.get("anchors")) else []  # exclude keys
        csd = ckpt["model"].float().state_dict()  # checkpoint state_dict as FP32
        # csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
        # model.load_state_dict(csd, strict=False)  # load
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