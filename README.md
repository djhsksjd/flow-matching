# Flow Matching for Face Image Generation

使用 Flow Matching 算法生成人脸图像的完整实现。支持 CelebA、ImageNet 和自定义数据集。

## 特性

- ✅ **Flow Matching 算法**：实现完整的 Flow Matching 生成模型
- ✅ **多种数据集支持**：CelebA（自动下载）、ImageNet、自定义数据集
- ✅ **模块化设计**：清晰的代码结构，易于扩展
- ✅ **可视化工具**：GIF 动画和 Web 界面展示生成过程
- ✅ **开箱即用**：默认配置即可开始训练

## 快速开始

### 安装

```bash
# 克隆或下载项目
cd flowmatching/code

# 安装依赖
pip install -r requirements.txt
```

### 训练模型

```bash
# 使用默认配置（CelebA数据集）
python -m scripts.train

# 或直接运行
python scripts/train.py
```

首次运行会自动下载 CelebA 数据集（约 1.3GB）。

### 生成样本

```bash
# 使用训练好的模型生成样本
python -m scripts.generate

# 自定义参数
python -m scripts.generate --num_samples 100 --num_steps 200
```

### 可视化生成过程

```bash
# 生成 GIF 动画
python -m scripts.visualize

# 启动 Web 界面（交互式）
python -m scripts.serve
# 然后在浏览器打开 http://127.0.0.1:5000
```

## 项目结构

```
flowmatching/
├── flowmatching/          # 核心包
│   ├── __init__.py
│   ├── models.py          # 模型架构 (UNet, ResidualBlock, TimeEmbedding)
│   ├── flow_matching.py   # Flow Matching 算法实现
│   ├── data.py            # 数据加载工具
│   ├── train.py           # 训练函数
│   ├── utils.py           # 工具函数
│   └── config.py          # 配置参数
│
├── scripts/               # 可执行脚本
│   ├── train.py          # 训练脚本
│   ├── generate.py       # 生成样本脚本
│   ├── visualize.py      # 可视化脚本
│   └── serve.py          # Web 服务器
│
├── data/                  # 数据目录（自动创建）
├── results/               # 结果目录（自动创建）
│
├── requirements.txt       # 依赖包
├── setup.py              # 安装脚本
├── README.md             # 本文件
└── .gitignore           # Git 忽略文件
```

## 配置

在 `flowmatching/config.py` 中修改配置：

```python
# 数据配置
DATA_CONFIG = {
    'dataset_type': 'celeba',  # 'celeba', 'face_folder', 'imagenet'
    'data_dir': './data',
    'batch_size': 32,
    'image_size': 64,
}

# 训练配置
TRAIN_CONFIG = {
    'batch_size': 32,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'save_every': 10,
    'sample_every': 5,
}
```

## 数据集

### CelebA（默认）

- **自动下载**：首次运行自动下载
- **规模**：200,000+ 张人脸图像
- **配置**：`dataset_type: 'celeba'`

### 自定义数据集

将图像放在 `data/faces/` 目录，设置 `dataset_type: 'face_folder'`

### ImageNet

需要手动下载和组织，设置 `dataset_type: 'imagenet'`

## 使用示例

### 训练

```python
from flowmatching import UNet, FlowMatching, load_celeba_data, train_flow_matching

# 加载数据
train_loader = load_celeba_data(batch_size=32, image_size=64)

# 创建模型
model = UNet(in_channels=3, time_emb_dim=128, base_channels=64)

# 训练
flow_matching = train_flow_matching(model, train_loader, num_epochs=50)
```

### 生成

```python
from flowmatching import UNet, FlowMatching
from flowmatching.utils import load_checkpoint, visualize_samples

# 加载模型
model = UNet(**MODEL_CONFIG)
load_checkpoint('results/checkpoint_final.pt', model)

# 生成样本
flow_matching = FlowMatching(model)
samples = flow_matching.sample(num_samples=64, num_steps=100)

# 可视化
visualize_samples(samples, save_path='samples.png')
```

## 命令行工具

安装后可以使用命令行工具：

```bash
# 训练
flowmatching-train

# 生成
flowmatching-generate --num_samples 100

# 可视化
flowmatching-visualize --num_steps 200

# Web 服务器
flowmatching-serve --port 5000
```

## 输出文件

训练结果保存在 `results/` 目录：

- `checkpoint_epoch_N.pt` - 训练检查点
- `checkpoint_final.pt` - 最终模型
- `samples_epoch_N.png` - 训练过程样本
- `final_samples_grid.png` - 最终样本网格
- `animations/generation_animation.gif` - 生成过程动画

## 性能优化

- **GPU 加速**：自动检测并使用 CUDA
- **批次大小**：根据 GPU 内存调整 `batch_size`
- **图像尺寸**：64x64（快速）或 128x128（高质量）
- **生成步数**：50-100（快速）或 200-500（高质量）

## 故障排除

### Google Drive 下载限制

如果遇到下载限制，可以：
1. 等待 24 小时后重试
2. 手动下载数据集（见文档）
3. 切换到自定义数据集

### 内存不足

减小 `batch_size` 或 `image_size` 在配置文件中。

## 参考文献

- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
- CelebA Dataset: https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html

## 许可证

MIT License
