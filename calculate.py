
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

# m = torch.load("weights/model_plus_final.pt")
# model = m['model']

model = torch.load("toy/neumeta_test/gen_ninr_yoloface500kp-500e-coordnoise-largers-resmlp-dim150-240_0.005.pth")
model_org = torch.load("toy/neumeta_test/gen_ninr_yoloface500kp-500e-coordnoise-largers-resmlp-dim150-240_1.0.pth")


total_params = sum(p.numel() for p in model.values())
print("Total parameters:" + str(total_params))
filter_params = sum(p.numel() for k, p in model.items() if k.startswith("model.21.1."))
print("Filter parameters:" + str(filter_params))
total_params_org = sum(p.numel() for p in model_org.values())
print("Total parameters org:" + str(total_params_org))
filter_params_org = sum(p.numel() for k, p in model_org.items() if k.startswith("model.21.1."))
print("Filter parameters org:" + str(filter_params_org))

# layer_params = sum(p.numel() for p in model.model[21][1].parameters())
# print("Layer parameters:" + str(layer_params))



