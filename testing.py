import torch
from models.experimental import attempt_load
from utils.torch_utils import prune

weights = "weights/pruned_model_plus_final.pt"
# model = attempt_load(weights)
model = torch.load(weights)["model"].float()
prune(model, 0.3)
print(model)
result = {}
result["model"] = model
torch.save(result, "weights/pruned_model_plus_final2.pt")