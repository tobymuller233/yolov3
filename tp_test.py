import torch
import torch_pruning as tp
from models.experimental import attempt_load
from models.yolo import Detect
from models.yolo import Bottleneck3
from torchvision.models import resnet18
from datetime import datetime

weights = "weights/model_plus_final.pt"
model1 = torch.load(weights)
model = attempt_load(weights)
model.requires_grad_(True)
# weights = "weights/resnet.pt"
# model = torch.load(weights)
# f = open("model.txt", "w")
# f.write(str(model))
#print(model)
# for m in model.modules():
# 	if isinstance(m, Detect):
# 		print(m)

# 1. importance measure
imp = tp.importance.MagnitudeImportance(p=2)
# imp = tp.importance.GroupNormImportance()
example_inputs = torch.randn(1, 3, 640, 640) # dummy input
# 2. layers to be ignored
ignored_layers = [model.model[27], model.model[33], model.model[39], model.model[40]]
# 3. init pruner
iterative_steps = 10
pruner = tp.pruner.MetaPruner(
    model,
    example_inputs, 
    importance=imp, 
    iterative_steps=iterative_steps, 
    ch_sparsity=0.2, 
    ignored_layers=ignored_layers, 
    isomorphic=True,
    global_pruning=True
)

# 4. Pruning-Finetuning
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print("  Iter %d/%d, Params: %.2f M => %.2f M" % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6))
    print("  Iter %d/%d, MACs: %.2f G => %.2f G"% (i+1, iterative_steps, base_macs / 1e9, macs / 1e9))         	

result = {}
for item, value in model1.items():
    if item == "model":
        result[item] = model
    elif item == "date":
        result[item] = datetime.now().isoformat()
    else:
        result[item] = value

print(model)
torch.save(result, "weights/pruned_model_plus_20.pt")
# fp = open("pruned_model3.txt", "w")
# fp.write(str(model))
# for i in range(iterative_steps):
#     for group in pruner.step(interactive=True): # Warning: groups must be handled sequentially. Do not keep them as a list.
#         print(group) 
#         # do whatever you like with the group 
#         dep, idxs = group[0] # get the idxs
#         target_module = dep.target.module # get the root module
#         pruning_fn = dep.handler # get the pruning function
#         group.prune()
#         # group.prune(idxs=[0, 2, 6]) # It is even possible to change the pruning behaviour with the idxs parameter
#     macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)

# # 1. Build dependency graph for a resnet18. This requires a dummy input for forwarding
# base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
# DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,640, 640))

# # 2. Get the group for pruning model.conv1 with the specified channel idxs
# group = DG.get_pruning_group( model.model[0].conv , tp.prune_conv_out_channels, idxs=[2, 6, 7] )

# # 3. Do the pruning
# if DG.check_pruning_group(group): # avoid over-pruning, i.e., channels=0.
#     group.prune()

# print(group)
# macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
# # print("Params: %.2f M => %.2f M" % (base_nparams / 1e6, nparams / 1e6))
# # print("MACs: %.2f G => %.2f G"% (base_macs / 1e9, macs / 1e9))         	
# print(base_macs, base_nparams)
# print(macs, nparams)