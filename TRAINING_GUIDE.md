# 建筑物识别 DeepLabV3+ 训练指南

## 概述

本指南专门针对建筑物识别任务，使用 RTX 3050 4GB 显卡进行训练。

## 硬件配置

- GPU: RTX 3050 (4GB 显存)
- 功耗: 95W
- 推荐 PyTorch 版本: 1.7.1 或更高

## 步骤 1: 准备数据集

### 1.1 数据集结构

你的数据集应该有以下结构：
```
你的数据集路径/
├── image/          # 包含所有训练图像 (.tif, .jpg, .png)
└── label/          # 包含对应标签图像 (0/255 二值图)
```

### 1.2 转换为 VOC 格式

编辑 `convert_labels.py` 文件，修改以下路径：

```python
input_dataset_dir = r"E:\数据集\3. The cropped image tiles and raster labels\test"  # 修改为你的数据集路径
output_voc_dir = r"e:\project\deeplabv3-plus-pytorch\VOCdevkit\VOC2007"
```

然后运行：
```bash
python convert_labels.py
```

这将：
- 将图像转换为 JPG 格式
- 将标签从 0/255 转换为 0/1 格式
- 按 8:2 比例划分训练集和验证集
- 生成 VOC 格式的目录结构

## 步骤 2: 下载预训练权重

根据 README.md 中的说明，从百度网盘下载预训练权重：
- 链接: https://pan.baidu.com/s/1IQ3XYW-yRWQAy7jxCUHq8Q 
- 提取码: qqq4

将 `deeplab_mobilenetv2.pth` 放置在 `model_data/` 目录下。

## 步骤 3: 开始训练

使用针对 RTX 3050 4GB 优化的配置文件：

```bash
python train_building.py
```

### 训练参数说明（已针对 4GB 显存优化）

| 参数 | 值 | 说明 |
|------|-----|------|
| input_shape | [320, 320] | 输入图像尺寸 |
| downsample_factor | 16 | 下采样倍数（显存小用16） |
| Freeze_batch_size | 4 | 冻结阶段 batch size |
| Unfreeze_batch_size | 2 | 解冻阶段 batch size |
| Freeze_Epoch | 30 | 冻结训练轮数 |
| UnFreeze_Epoch | 60 | 总训练轮数 |
| fp16 | True | 混合精度训练（节省显存） |
| optimizer_type | adam | 优化器 |
| Init_lr | 5e-4 | 初始学习率 |
| num_classes | 2 | 类别数（背景+建筑物） |

### 如果显存不足（OOM）

如果遇到显存不足错误，可以尝试：
1. 将 `input_shape` 改为 `[256, 256]`
2. 将 `Freeze_batch_size` 改为 2
3. 将 `Unfreeze_batch_size` 改为 1

## 步骤 4: 训练过程监控

- 训练日志保存在 `logs/loss_YYYY_MM_DD_HH_MM_SS/` 目录
- 每 5 个 epoch 保存一次权重
- 最佳权重保存在 `logs/best_epoch_weights.pth`

## 步骤 5: 预测/推理

训练完成后，使用 `predict_building.py` 进行预测：

```bash
python predict_building.py
```

在代码中修改 `mode` 参数：
- `'predict'`: 单张图片预测
- `'dir_predict'`: 批量预测文件夹中的图片
- `'video'`: 视频检测

## 关键注意事项

### 1. 标签格式非常重要

- **错误格式**: 背景=0, 建筑物=255 ❌
- **正确格式**: 背景=0, 建筑物=1 ✅

`convert_labels.py` 会自动处理这个转换。

### 2. Dice Loss

对于二分类任务（建筑物识别），已启用 `dice_loss=True`，这有助于处理类别不平衡问题。

### 3. 冻结训练

先冻结主干网络训练 30 个 epoch，然后解冻全部网络再训练 30 个 epoch。这样可以：
- 节省显存
- 加快初期收敛
- 防止预训练权重被破坏

## 目录结构（训练后）

```
deeplabv3-plus-pytorch/
├── VOCdevkit/
│   └── VOC2007/
│       ├── JPEGImages/          # 转换后的图像
│       ├── SegmentationClass/   # 转换后的标签
│       └── ImageSets/
│           └── Segmentation/
│               ├── train.txt
│               └── val.txt
├── model_data/
│   └── deeplab_mobilenetv2.pth # 预训练权重
├── logs/
│   ├── loss_YYYY_MM_DD_HH_MM_SS/
│   └── best_epoch_weights.pth   # 最佳模型权重
├── convert_labels.py             # 数据集转换脚本
├── train_building.py             # 训练脚本
└── predict_building.py           # 预测脚本
```

## 常见问题

**Q: 训练时提示 "CUDA out of memory" 怎么办？**
A: 减小 batch size 和 input_shape，参考上面的"如果显存不足"部分。

**Q: 预测结果全黑或全红怎么办？**
A: 检查标签格式是否正确（0和1，不是0和255）。

**Q: 如何恢复中断的训练？**
A: 修改 `train_building.py` 中的 `model_path` 指向已保存的权重，调整 `Init_Epoch` 为已训练的轮数。
