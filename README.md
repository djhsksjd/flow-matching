# Flow Matching for ImageNet Image Generation

使用 Flow Matching 算法生成图像的完整实现。支持 ImageNet、CelebA 数据集或自定义图像文件夹。

## 项目结构

```
flowmatching/          # 主包目录
├── __init__.py       # 包初始化
├── models.py         # 模型架构 (UNet, ResidualBlock, TimeEmbedding)
├── flow_matching.py  # Flow Matching 算法实现
├── data.py           # 数据加载工具（支持ImageNet、CelebA和自定义文件夹）
├── train.py          # 训练函数
├── utils.py          # 工具函数（仅网格可视化）
└── config.py         # 配置参数

train_main.py         # 主训练脚本
generate_samples.py   # 生成样本脚本
requirements.txt      # 依赖包
README.md            # 本文件
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 使用方法

### 1. 准备 ImageNet 数据集

ImageNet 数据集需要特定的目录结构：

```
data/
└── imagenet/
    ├── train/
    │   ├── n01440764/        # 类别1
    │   │   ├── n01440764_18.JPEG
    │   │   ├── n01440764_36.JPEG
    │   │   └── ...
    │   ├── n01443537/        # 类别2
    │   │   └── ...
    │   └── ...
    └── val/                  # 可选：验证集
        ├── n01440764/
        └── ...
```

**目录要求：**
- `data/imagenet/train/` 目录包含所有训练图像
- 每个类别一个子文件夹，文件夹名即为类别名
- 图像格式支持：`.JPEG`, `.jpg`, `.png` 等

### 2. 配置数据集类型

在 `flowmatching/config.py` 中设置：

```python
DATA_CONFIG = {
    'data_dir': './data/imagenet',
    'batch_size': 16,
    'image_size': 128,  # 图像尺寸 (64, 128, 224等)
    'dataset_type': 'imagenet',  # 'imagenet', 'celeba', 或 'face_folder'
}
```

### 3. 开始训练

```bash
python train_main.py
```

训练过程会：
- 加载 ImageNet 数据集
- 训练 Flow Matching 模型
- 每 5 个 epoch 生成并保存样本网格图像
- 每 10 个 epoch 保存模型检查点
- 训练结束后生成最终的高质量样本

**训练结果保存在 `results/` 目录：**
- `checkpoint_epoch_N.pt`: 训练检查点
- `checkpoint_final.pt`: 最终模型
- `samples_epoch_N.png`: 训练过程中的样本网格
- `final_samples_grid.png`: 最终样本网格图

## 数据集选项

### ImageNet（推荐）

- **优点**: 数据集大，多样性高，图像质量好
- **缺点**: 数据集很大（~150GB），需要手动下载
- **配置**: `dataset_type: 'imagenet'`
- **目录结构**: `data/imagenet/train/类别名/图像文件`

### CelebA

- **优点**: 自动下载，较小（~1.3GB）
- **缺点**: 仅限人脸图像
- **配置**: `dataset_type: 'celeba'`
- **自动下载**: 首次运行自动下载

### 自定义图像文件夹

- **优点**: 可以使用任何图像数据
- **配置**: `dataset_type: 'face_folder'`
- **目录结构**: `data/faces/` 或按类别组织的文件夹

## 配置参数

在 `flowmatching/config.py` 中修改配置：

```python
# 模型配置（RGB图像，3通道）
MODEL_CONFIG = {
    'in_channels': 3,
    'time_emb_dim': 128,
    'base_channels': 64,
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 16,      # ImageNet建议16-32
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'save_every': 10,
    'sample_every': 5,
}

# 采样配置
SAMPLE_CONFIG = {
    'num_samples': 64,
    'num_steps': 100,
    'image_size': (128, 128),  # 建议128x128或64x64
}

# 数据配置
DATA_CONFIG = {
    'data_dir': './data/imagenet',
    'batch_size': 16,
    'image_size': 128,         # 图像会被调整为指定尺寸
    'dataset_type': 'imagenet', # 'imagenet', 'celeba', 'face_folder'
}
```

## 生成样本

从训练好的模型生成新样本：

```bash
# 使用默认设置
python generate_samples.py

# 自定义参数
python generate_samples.py --num_samples 100 --num_steps 200

# 指定检查点
python generate_samples.py --checkpoint results/checkpoint_epoch_50.pt
```

## 性能优化建议

### 内存优化

- **Batch Size**: ImageNet 图像较大，建议使用 8-16
- **Image Size**: 
  - 64x64: 内存友好，速度快
  - 128x128: 平衡质量和速度（推荐）
  - 224x224: 高质量但需要更多内存

### 训练加速

- **使用 GPU**: 强烈推荐使用 CUDA GPU
- **多进程加载**: `num_workers=4` 已在代码中设置
- **混合精度**: 可考虑使用 `torch.cuda.amp` 加速

### 质量提升

- **增加训练轮数**: `num_epochs` 设置为 100+ 
- **增加生成步数**: `num_steps` 设置为 200-500
- **更大模型**: 增加 `base_channels` (如 128 或 256)

## 在代码中使用

```python
from flowmatching import UNet, FlowMatching, load_imagenet_data
from flowmatching.utils import visualize_samples
import torch

# 加载 ImageNet 数据
train_loader = load_imagenet_data(
    batch_size=16,
    data_dir='./data/imagenet',
    image_size=128,
    split='train'
)

# 创建模型
model = UNet(in_channels=3, time_emb_dim=128, base_channels=64)

# 创建 Flow Matching 实例
flow_matching = FlowMatching(model, device='cuda')

# 生成样本
samples = flow_matching.sample(
    num_samples=64, 
    num_steps=100,
    image_size=(128, 128),
    channels=3
)

# 可视化并保存网格
visualize_samples(samples, save_path='imagenet_samples.png', nrow=8)
```

## 输出说明

所有生成的图像都以**网格形式**保存为 PNG 文件：
- 训练过程中的样本: `results/samples_epoch_N.png`
- 最终生成样本: `results/final_samples_grid.png`
- 自定义生成: `results/generated_samples_grid.png`

## 故障排除

### 1. ImageNet 数据集问题

**错误**: `ImageNet train directory not found`

**解决**:
- 确保目录结构正确: `data/imagenet/train/类别名/图像文件`
- 检查路径是否正确配置在 `config.py` 中
- 确保有足够的磁盘空间（ImageNet 约 150GB）

### 2. 内存不足

**解决**:
- 减小 `batch_size` 到 8 或更小
- 减小 `image_size` 到 64
- 使用梯度累积

### 3. 训练速度慢

**解决**:
- 确保使用 GPU
- 增加 `num_workers`（如果 CPU 核心数足够）
- 考虑使用较小的图像尺寸进行快速实验

### 4. 图像质量不好

**解决**:
- 增加训练轮数 `num_epochs`
- 生成时增加 `num_steps` (200-500)
- 使用更大的模型 (`base_channels=128`)

## ImageNet 数据集获取

ImageNet 数据集需要从官网获取：
- 官网: https://www.image-net.org/
- 需要注册账号
- 下载 ILSVRC2012 数据集
- 解压并按上述目录结构组织

**注意**: ImageNet 数据集很大，确保有足够的存储空间。

## 参考文献

- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- ImageNet: https://www.image-net.org/
