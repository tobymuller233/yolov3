from models.yolo import Model, Detect

import yaml
import torch

hyp = yaml.safe_load("data/hyps/hyp.scratch-high.yaml")
model = Model("models/yoloface-500kp-layer21-dim120-3class.yaml", ch=3, nc=3, anchors=3)
weights = "weights/neumeta_threeclass_alldata_e500_bg.pt"
ckpt = torch.load(weights, map_location="cpu")
model.load_state_dict(ckpt["model"].float().state_dict())

for x in model.model:
    if isinstance(x, Detect):
        num_param = sum([p.numel() for p in x.parameters()])
        print(f"Detect: {num_param} parameters")

total_param = sum([p.numel() for p in model.parameters()])
print(f"Other part: {total_param - num_param} parameters")
print(f"Total: {total_param} parameters")