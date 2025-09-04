# YOLO-MobileOne 500K Model Configuration

## 概述

这是一个基于MobileOne架构的YOLO模型配置，目标参数量约为500K，与yoloface-500k相近。该配置结合了MobileOne的重参数化技术和YOLO的检测头，旨在提供更低的推理延迟。

## 主要特性

### 1. MobileOne架构特点
- **重参数化技术**: 训练时使用多分支结构，推理时合并为单一分支
- **多分支设计**: 包含1x1卷积、3x3卷积、BN等分支
- **高效推理**: 通过重参数化减少推理时的计算量

### 2. 模型结构
- **Backbone**: 使用MobileOneBlock构建的特征提取网络
- **Head**: 保持YOLO的检测头结构，支持多尺度检测
- **参数量**: 约500K参数，与yoloface-500k相近

## 文件结构

```
models/
├── yolo-mobileone-500k.yaml    # 主配置文件
├── common.py                   # 包含MobileOneBlock和MobileOneStage实现
└── yolo.py                     # 已更新支持MobileOne模块
```

## 新增模块

### MobileOneBlock
```python
class MobileOneBlock(nn.Module):
    def __init__(self, c1, c2, k=3, s=1, num_conv_branches=1):
        # c1: 输入通道数
        # c2: 输出通道数
        # k: 卷积核大小
        # s: 步长
        # num_conv_branches: 重参数化分支数量
```

### MobileOneStage
```python
class MobileOneStage(nn.Module):
    def __init__(self, c1, c2, num_blocks, k=3, s=1, num_conv_branches=1):
        # 包含多个MobileOneBlock的阶段
```

## 使用方法

### 1. 训练
```bash
python train.py --cfg models/yolo-mobileone-500k.yaml --data your_dataset.yaml
```

### 2. 推理
```bash
python detect.py --weights path/to/weights.pt --source path/to/images
```

### 3. 验证
```bash
python val.py --weights path/to/weights.pt --data your_dataset.yaml
```

## 配置说明

### 主要参数
- `nc: 1`: 类别数量（人脸检测）
- `depth_multiple: 1.0`: 深度倍数
- `width_multiple: 1.0`: 宽度倍数
- `anchors`: 锚点配置，与yoloface-500k相同

### Backbone结构
```
输入 -> Conv(8) -> MobileOneBlock(8) 
     -> Conv(16) -> MobileOneBlock(16) -> MobileOneBlock(16)
     -> Conv(24) -> MobileOneBlock(24) -> MobileOneBlock(24) -> MobileOneBlock(24)
     -> Conv(48) -> MobileOneBlock(48) -> MobileOneBlock(48) -> MobileOneBlock(48)
     -> Conv(96) -> MobileOneBlock(96) -> MobileOneBlock(96)
```

### Head结构
保持YOLO的多尺度检测头，支持P3、P4、P5三个检测层。

## 重参数化

MobileOne的核心优势在于重参数化技术：

1. **训练阶段**: 使用多分支结构（1x1卷积、3x3卷积、BN分支）
2. **推理阶段**: 将所有分支合并为单一3x3卷积，减少计算量

## 性能预期

- **参数量**: ~500K（与yoloface-500k相近）
- **推理速度**: 相比传统YOLO模型有显著提升
- **精度**: 在保持检测精度的同时降低延迟

## 注意事项

1. 确保PyTorch版本兼容
2. 训练时使用标准YOLO训练流程
3. 推理前可选择性进行重参数化优化
4. 建议在目标硬件上进行性能测试

## 扩展性

该配置可以进一步优化：
- 调整通道数配置
- 修改重参数化分支数量
- 添加更多MobileOneStage
- 结合其他轻量化技术

## 测试

运行测试脚本验证配置：
```bash
python simple_test.py
```

这将验证：
- 模型创建是否成功
- 参数量是否符合预期
- 前向传播是否正常
