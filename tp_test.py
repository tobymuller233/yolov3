import torch
import torch_pruning as tp
from models.experimental import attempt_load
from models.yolo import Detect
from torchvision.models import resnet18

weights = "weights/model_plus_final.pt"
model = attempt_load(weights)
model.requires_grad_(True)
# weights = "weights/resnet.pt"
# model = torch.load(weights)
print(model)
#print(model)
# for m in model.modules():
# 	if isinstance(m, Detect):
# 		print(m)

# 1. 选择合适的重要性评估指标，这里使用权值大小
imp = tp.importance.MagnitudeImportance(p=2)
# imp = tp.importance.GroupNormImportance()
example_inputs = torch.randn(1, 3, 640, 640) # dummy input
# 2. 忽略无需剪枝的层，例如最后的分类层（总不能剪完类别都变少了叭？）
ignored_layers = []
flag = False
for m in model.modules():
    if isinstance(m, Detect):
        flag = True
    if flag:
        ignored_layers.append(m)

# 3. 初始化剪枝器
iterative_steps = 5# 迭代式剪枝，重复5次Pruning-Finetuning的循环完成剪枝。
pruner = tp.pruner.MetaPruner(
    model,
    example_inputs, # 用于分析依赖的伪输入
    importance=imp, # 重要性评估指标
    iterative_steps=iterative_steps, # 迭代剪枝，设为1则一次性完成剪枝
    ch_sparsity=0.3, # 目标稀疏性，这里我们移除50%的通道 ResNet18 = {64, 128, 256, 512} => ResNet18_Half = {32, 64, 128, 256}
    ignored_layers=ignored_layers, # 忽略掉最后的分类层
)

# 4. Pruning-Finetuning的循环
base_macs, base_nparams = tp.utils.count_ops_and_params(model, example_inputs)
for i in range(iterative_steps):
    pruner.step()
    macs, nparams = tp.utils.count_ops_and_params(model, example_inputs)
    print("  Iter %d/%d, Params: %.2f M => %.2f M" % (i+1, iterative_steps, base_nparams / 1e6, nparams / 1e6))
    print("  Iter %d/%d, MACs: %.2f G => %.2f G"% (i+1, iterative_steps, base_macs / 1e9, macs / 1e9))         	

result = {}
result["model"] = model
torch.save(result, "weights/pruned_model_plus_final.pt")
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