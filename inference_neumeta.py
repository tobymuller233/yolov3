import argparse
import os
import platform
import sys
from pathlib import Path

import torch

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from ultralytics.utils.plotting import Annotator, colors, save_one_box

from models.common import DetectMultiBackend, Conv, DWConv
from models.yolo import Model
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (
    LOGGER,
    Profile,
    check_file,
    check_img_size,
    check_imshow,
    check_requirements,
    colorstr,
    cv2,
    increment_path,
    non_max_suppression,
    print_args,
    scale_boxes,
    strip_optimizer,
    xyxy2xywh,
)
from utils.torch_utils import select_device, smart_inference_mode
from utils.neumeta import load_checkpoint_mine
from neumeta.utils import get_hypernet, load_checkpoint, sample_merge_model

from omegaconf import OmegaConf

LOCAL_RANK = int(os.getenv("LOCAL_RANK", -1))  # https://pytorch.org/docs/stable/elastic/run.html
RANK = int(os.getenv("RANK", -1))
WORLD_SIZE = int(os.getenv("WORLD_SIZE", 1))

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default=ROOT / "weights/model_plus_final.pt", help="model path or triton URL")
    parser.add_argument("--cfg", type=str, default=ROOT / "models/yoloface-500kp.yaml", help="model yaml path")
    parser.add_argument("--hypernet", type=str, default=ROOT / "weights/hypernet_final.pt", help="hypernet path or triton URL")
    parser.add_argument("--ratio", type=float, default=1.0, help="model ratio")
    parser.add_argument("--save", action="store_true", default=True, help="save results")
    parser.add_argument("--save-path", type=str, default=ROOT / "inference", help="save path")
    parser.add_argument("--yaml", type=str, default=ROOT / "models/yolov3.yaml", help="model.yaml path")
    parser.add_argument("--test-model", action="store_true", default=False, help="test model")

    parser_result = parser.parse_args()

    yaml_args = OmegaConf.load(parser_result.yaml)
    if yaml_args.get("base_config", None):
        print("Loading base config from " + yaml_args.base_config)
        base_config = OmegaConf.load(yaml_args.base_config)
        yaml_args = OmegaConf.merge(base_config, yaml_args)
    
    cli_args = {k: v for k, v in vars(parser_result).items()}
    config = OmegaConf.merge(yaml_args, cli_args)
    
    args = argparse.Namespace(**config)
    return args
    pass

def create_model_yolov3(path, device, args, hidden_dim=240, change_layers=False):
    weights = path
    ckpt = torch.load(weights, map_location="cpu")
    model = Model(args.cfg, ch=3, nc=1, anchors=3, change_layers=change_layers).to(device)
    csd = ckpt["model"].float().state_dict()
    if change_layers:
        model.model[21][1].cv1 = Conv(model.model[21][1].cv1.conv.in_channels, hidden_dim, 1, 1)
        model.model[21][1].cv2 = DWConv(hidden_dim, hidden_dim, 3, 1)
        model.model[21][1].cv3 = Conv(hidden_dim, model.model[21][1].cv3.conv.out_channels, 1, 1)
    model.model = load_checkpoint_mine(model.model, csd, prefix="model.")    # load
    return model

def main(args):
    device = select_device("cuda:0", 1)
    model = create_model_yolov3(args.weights, device, args)
    hypernet = get_hypernet(args, len(model.learnable_parameter), device)

    # load checkpoint for hypernet
    print(f"Loading hypernet from {args.hypernet}")
    hyper_checkpoint = load_checkpoint(args.hypernet, hypernet, None, None, device)
    print(f"Hypernet best acc: {hyper_checkpoint['best_acc']}")
    
    # create model for given dimension
    model_cls = create_model_yolov3(args.weights, device, args, hidden_dim=int(240 * args.ratio), change_layers=True)
    accumulated_model = sample_merge_model(hypernet, model_cls, args)
    if args.save:
        # mkdir
        os.makedirs(args.save_path, exist_ok=True)
        
        save_path = os.path.join(args.save_path, f"{args.experiment.name}_{args.ratio}.pth")
        torch.save(accumulated_model.state_dict(), save_path)
    pass
if __name__ == "__main__":
    args = parse_opt()
    main(args)   