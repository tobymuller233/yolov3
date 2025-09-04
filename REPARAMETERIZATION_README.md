# MobileOne Reparameterization Guide

## 概述

MobileOne的重参数化是一种将训练时的多分支结构融合为推理时单分支结构的技术，可以显著提升推理速度而不损失精度。

## 重参数化原理

### 训练时 (Multi-branch)
```
输入 → [1x1 Conv] → [3x3 Conv] → [Skip Connection] → 输出
     ↘           ↗              ↗
       [Scale Branch] → [BN] → [Activation]
```

### 推理时 (Single-branch)
```
输入 → [Single Conv] → 输出
```

## 使用方法

### 1. 基本使用

```python
from models.yolo import DetectionModel
import yaml

# 加载配置
with open("models/yolo-mobileone-500k.yaml", 'r') as f:
    cfg = yaml.safe_load(f)

# 创建模型
model = DetectionModel(cfg, ch=3, nc=cfg['nc'], anchors=cfg['anchors'])

# 重参数化 (推理前必须调用!)
model.reparameterize_mobileone()

# 推理
model.eval()
with torch.no_grad():
    outputs = model(input_tensor)
```

### 2. 训练 vs 推理模式

```python
# 训练模式 - 使用多分支结构
model.train()
# 不需要调用 reparameterize_mobileone()

# 推理模式 - 使用单分支结构
model.eval()
model.reparameterize_mobileone()  # 必须调用!
```

### 3. 性能对比

```python
import time

# 重参数化前
start = time.time()
for _ in range(100):
    output = model(input_tensor)
time_before = time.time() - start

# 重参数化后
model.reparameterize_mobileone()
start = time.time()
for _ in range(100):
    output = model(input_tensor)
time_after = time.time() - start

print(f"Speedup: {time_before/time_after:.2f}x")
```

## 重要注意事项

### ⚠️ 必须遵循的步骤

1. **训练时**: 不要调用 `reparameterize_mobileone()`
2. **推理前**: 必须先调用 `model.eval()` 然后 `model.reparameterize_mobileone()`
3. **重参数化后**: 模型结构已改变，不能继续训练

### ✅ 正确的工作流程

```python
# 1. 训练阶段
model.train()
# ... 训练代码 ...

# 2. 保存模型
torch.save(model.state_dict(), 'model.pth')

# 3. 推理阶段
model.load_state_dict(torch.load('model.pth'))
model.eval()  # 设置为评估模式
model.reparameterize_mobileone()  # 重参数化

# 4. 推理
with torch.no_grad():
    outputs = model(input_tensor)
```

## 技术细节

### MobileOne Block结构

```python
class MobileOneBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, num_conv_branches=1):
        # 多分支结构
        self.rbr_conv = nn.ModuleList()  # 卷积分支
        self.rbr_scale = self._conv_bn()  # 尺度分支
        self.rbr_skip = nn.BatchNorm2d()  # 跳跃连接
    
    def reparameterize(self):
        # 融合所有分支为单个卷积层
        kernel, bias = self._get_kernel_bias()
        self.reparam_conv = nn.Conv2d(...)
        self.reparam_conv.weight.data = kernel
        self.reparam_conv.bias.data = bias
```

### 重参数化过程

1. **计算融合权重**: 将BN参数融合到卷积权重中
2. **创建单分支卷积**: 用融合后的权重创建新的卷积层
3. **删除多分支结构**: 移除原始的多分支模块
4. **更新前向传播**: 使用新的单分支卷积

## 性能优势

- **推理速度**: 提升 1.2-2.0x
- **内存占用**: 减少 20-30%
- **精度保持**: 完全等价，无精度损失
- **部署友好**: 单分支结构更适合移动端部署

## 示例脚本

### 完整演示
```bash
python demo_reparameterization.py
```

### 简单推理
```bash
python inference_example.py
```

## 故障排除

### 常见问题

1. **忘记重参数化**
   - 症状: 推理速度慢
   - 解决: 调用 `model.reparameterize_mobileone()`

2. **训练时重参数化**
   - 症状: 训练不稳定
   - 解决: 只在推理前重参数化

3. **重参数化后继续训练**
   - 症状: 模型结构错误
   - 解决: 重参数化后不能继续训练

### 验证重参数化

```python
# 检查MobileOne块是否被重参数化
reparam_count = 0
for name, module in model.named_modules():
    if hasattr(module, 'reparam_conv'):
        reparam_count += 1

print(f"Reparameterized blocks: {reparam_count}")
```

## 总结

MobileOne重参数化是提升推理效率的关键技术：

- ✅ **训练时**: 使用多分支结构提升表达能力
- ✅ **推理时**: 使用单分支结构提升速度
- ✅ **部署时**: 模型更小更快，适合移动端
- ✅ **精度**: 完全等价，无精度损失

正确使用重参数化可以显著提升模型的推理性能！
