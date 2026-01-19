# Flow Matching for MNIST Image Generation

使用 Flow Matching 算法生成 MNIST 数字图像的完整实现。

## 项目结构

```
flowmatching/          # 主包目录
├── __init__.py       # 包初始化
├── models.py         # 模型架构 (UNet, ResidualBlock, TimeEmbedding)
├── flow_matching.py  # Flow Matching 算法实现
├── data.py           # 数据加载工具
├── train.py          # 训练函数
├── utils.py          # 工具函数（可视化、保存等）
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

### 1. 训练模型

运行主训练脚本：

```bash
python train_main.py
```

训练过程会：
- 自动下载 MNIST 数据集
- 训练 Flow Matching 模型
- 每 5 个 epoch 生成并保存样本图像
- 每 10 个 epoch 保存模型检查点
- 训练结束后生成最终的高质量样本

**训练结果保存在 `results/` 目录：**
- `checkpoint_epoch_N.pt`: 训练检查点
- `checkpoint_final.pt`: 最终模型
- `samples_epoch_N.png`: 训练过程中的样本网格
- `final_samples/final_samples_grid.png`: 最终样本网格图
- `final_samples/individual/`: 最终样本的单独图像文件
- `final_samples/final_samples.npy`: 样本的 numpy 数组
- `individual_samples/epoch_N/`: 各 epoch 的单独样本图像

### 2. 从训练好的模型生成样本

```bash
# 使用默认设置生成样本
python generate_samples.py

# 指定参数生成样本
python generate_samples.py --num_samples 100 --num_steps 200

# 指定检查点文件
python generate_samples.py --checkpoint results/checkpoint_epoch_50.pt

# 保存单独图像和 numpy 数组
python generate_samples.py --save_individual --save_numpy

# 指定输出目录
python generate_samples.py --output_dir my_samples
```

**生成参数说明：**
- `--checkpoint`: 检查点文件路径（默认使用最新的检查点）
- `--num_samples`: 生成样本数量（默认 64）
- `--num_steps`: ODE 积分步数，越多质量越好但越慢（默认 100）
- `--output_dir`: 输出目录（默认 `results/generated_samples`）
- `--save_individual`: 保存单独的图像文件
- `--save_numpy`: 保存为 numpy 数组文件

### 3. 在代码中使用

```python
from flowmatching import UNet, FlowMatching, load_mnist_data
from flowmatching.utils import visualize_samples, save_samples_as_images
import torch

# 加载数据
train_loader, test_loader = load_mnist_data(batch_size=128)

# 创建模型
model = UNet(in_channels=1, time_emb_dim=128, base_channels=64)

# 创建 Flow Matching 实例
flow_matching = FlowMatching(model, device='cuda')

# 生成样本
samples = flow_matching.sample(num_samples=64, num_steps=100)

# 可视化并保存
visualize_samples(samples, save_path='my_samples.png', nrow=8)
save_samples_as_images(samples, 'individual_images', prefix='img')
```

## 配置参数

可以在 `flowmatching/config.py` 中修改配置：

- **模型配置**: 网络架构参数
- **训练配置**: 学习率、批次大小、训练轮数等
- **采样配置**: 生成样本的数量和步数
- **路径配置**: 数据目录和结果目录

## 保存功能

项目支持多种保存方式：

1. **网格图像** (`visualize_samples`): 将所有样本排列成网格并保存为 PNG
2. **单独图像** (`save_samples_as_images`): 将每个样本保存为单独的图像文件
3. **Numpy 数组** (`save_samples_as_numpy`): 保存为 `.npy` 文件供后续处理

## 性能优化建议

- 使用 GPU 训练会快很多（自动检测 CUDA）
- 增加 `num_steps` 可以提高生成质量，但会增加生成时间
- 调整 `batch_size` 以适应你的 GPU 内存

## 文件说明

- **训练过程中的保存**:
  - 每 5 个 epoch 自动生成样本并保存到 `results/samples_epoch_N.png`
  - 同时保存单独图像到 `results/individual_samples/epoch_N/`

- **训练完成后的保存**:
  - 最终模型: `results/checkpoint_final.pt`
  - 最终样本网格: `results/final_samples/final_samples_grid.png`
  - 最终单独图像: `results/final_samples/individual/`
  - 最终 numpy 数组: `results/final_samples/final_samples.npy`

## 故障排除

1. **CUDA 内存不足**: 减小 `batch_size` 或 `num_samples`
2. **找不到检查点**: 确保先运行 `python train_main.py` 训练模型
3. **数据集下载失败**: 检查网络连接，MNIST 会自动下载到 `data/` 目录

## 参考文献

- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
