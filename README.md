# Flow Matching for Face Image Generation

使用 Flow Matching 算法生成人脸图像的完整实现。**默认使用 CelebA 数据集，自动下载，开箱即用！**

## 快速开始

### 一键启动（推荐）

```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 开始训练（自动下载CelebA数据集）
python train_main.py
```

就这么简单！CelebA 数据集会在首次运行时自动下载（约 1.3GB）。

## 项目结构

```
flowmatching/          # 主包目录
├── __init__.py       # 包初始化
├── models.py         # 模型架构 (UNet, ResidualBlock, TimeEmbedding)
├── flow_matching.py  # Flow Matching 算法实现
├── data.py           # 数据加载工具（支持CelebA、ImageNet和自定义文件夹）
├── train.py          # 训练函数
├── utils.py          # 工具函数（仅网格可视化）
└── config.py         # 配置参数

train_main.py         # 主训练脚本
generate_samples.py   # 生成样本脚本
requirements.txt      # 依赖包
README.md            # 本文件
```

## 数据集选项

### 1. CelebA（默认推荐）⭐

**特点：**
- ✅ **自动下载**：首次运行自动下载，无需手动设置
- ✅ **大规模**：超过 200,000 张名人图像
- ✅ **高质量**：218x178 像素，高分辨率人脸
- ✅ **开箱即用**：无需任何配置

**配置：**
```python
# flowmatching/config.py
DATA_CONFIG = {
    'dataset_type': 'celeba',  # 默认选项
    'data_dir': './data',      # 数据集将下载到这里
    'batch_size': 32,
    'image_size': 64,          # 可调整为 64, 128 等
}
```

**使用方法：**
```bash
python train_main.py
```

### 2. 自定义人脸图像文件夹

将你的人脸图像放在文件夹中：

**配置：**
```python
DATA_CONFIG = {
    'dataset_type': 'face_folder',
    'data_dir': './data/faces',  # 你的图像文件夹
    'batch_size': 32,
    'image_size': 64,
}
```

**目录结构：**
```
data/
└── faces/
    ├── img1.jpg
    ├── img2.jpg
    └── ...
```

### 3. ImageNet（高级用户）

需要手动下载和组织数据集，适合有 ImageNet 数据的用户。

## 配置参数

在 `flowmatching/config.py` 中修改配置：

```python
# 模型配置
MODEL_CONFIG = {
    'in_channels': 3,  # RGB
    'time_emb_dim': 128,
    'base_channels': 64,
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 32,      # 根据GPU内存调整
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'save_every': 10,      # 每10个epoch保存检查点
    'sample_every': 5,     # 每5个epoch生成样本
}

# 采样配置
SAMPLE_CONFIG = {
    'num_samples': 64,
    'num_steps': 100,      # 增加可提高质量（但更慢）
    'image_size': (64, 64), # 可设置为 (128, 128)
}

# 数据配置
DATA_CONFIG = {
    'dataset_type': 'celeba',  # 'celeba', 'face_folder', 'imagenet'
    'data_dir': './data',
    'batch_size': 32,
    'image_size': 64,
}
```

## 使用方法

### 训练模型

```bash
python train_main.py
```

训练过程会：
- ✅ 自动下载 CelebA 数据集（如果使用）
- ✅ 训练 Flow Matching 模型
- ✅ 每 5 个 epoch 自动生成并保存样本网格
- ✅ 每 10 个 epoch 自动保存检查点
- ✅ 训练结束后生成最终高质量样本

### 生成样本

```bash
# 使用默认设置
python generate_samples.py

# 自定义参数
python generate_samples.py --num_samples 100 --num_steps 200

# 指定检查点
python generate_samples.py --checkpoint results/checkpoint_epoch_50.pt
```

## 输出文件

所有结果保存在 `results/` 目录：

```
results/
├── checkpoint_epoch_10.pt      # 检查点
├── checkpoint_epoch_20.pt
├── checkpoint_final.pt         # 最终模型
├── samples_epoch_5.png         # 训练过程样本网格
├── samples_epoch_10.png
├── samples_epoch_15.png
└── final_samples_grid.png      # 最终样本网格
```

## 性能优化

### GPU 内存优化

- **小 GPU (8GB)**: `batch_size=16`, `image_size=64`
- **中等 GPU (16GB)**: `batch_size=32`, `image_size=64`
- **大 GPU (24GB+)**: `batch_size=64`, `image_size=128`

### 生成质量优化

- **快速生成**: `num_steps=50-100`
- **高质量**: `num_steps=200-500`
- **极致质量**: `num_steps=1000+` (较慢)

### 训练优化

- 使用 GPU 可显著加速训练
- 增加训练轮数可提高质量
- 使用更大的模型 (`base_channels=128`) 可提升效果

## 关于 CelebA 数据集

**CelebA (Celebrities Attributes)** 是一个大规模人脸属性数据集：
- **图像数量**: 202,599 张
- **分辨率**: 178x218 像素
- **类别**: 10,177 个名人
- **属性**: 40 个二值属性标注
- **大小**: 约 1.3 GB（压缩包）

**自动下载位置**: `./data/celeb/`

首次下载需要一些时间，但之后会直接使用本地数据。

## 故障排除

### 1. 下载失败

**问题**: CelebA 下载中断或失败

**解决**:
- 检查网络连接
- 确保有足够的磁盘空间（至少 2GB）
- 重新运行，下载会自动恢复

### 2. CUDA 内存不足

**问题**: `RuntimeError: CUDA out of memory`

**解决**:
```python
# 在 config.py 中减小
TRAIN_CONFIG['batch_size'] = 16  # 或更小
DATA_CONFIG['image_size'] = 64   # 使用更小的图像
```

### 3. 生成质量不好

**问题**: 生成的图像模糊或质量差

**解决**:
- 增加训练轮数: `num_epochs = 100`
- 增加生成步数: `num_steps = 200-500`
- 使用更大的模型: `base_channels = 128`

## 在代码中使用

```python
from flowmatching import UNet, FlowMatching, load_celeba_data
from flowmatching.utils import visualize_samples

# 加载 CelebA 数据（自动下载）
train_loader = load_celeba_data(
    batch_size=32,
    data_dir='./data',
    image_size=64,
    split='train'
)

# 创建模型
model = UNet(in_channels=3, time_emb_dim=128, base_channels=64)

# 创建 Flow Matching
flow_matching = FlowMatching(model, device='cuda')

# 生成样本
samples = flow_matching.sample(
    num_samples=64,
    num_steps=100,
    image_size=(64, 64),
    channels=3
)

# 可视化
visualize_samples(samples, save_path='faces.png', nrow=8)
```

## 参考文献

- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- CelebA Dataset: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
