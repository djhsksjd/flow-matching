# 项目结构说明

## 目录结构

```
flowmatching/
├── flowmatching/          # 核心包（Python包）
│   ├── __init__.py       # 包初始化，导出主要接口
│   ├── models.py         # 神经网络模型
│   │   ├── TimeEmbedding
│   │   ├── ResidualBlock
│   │   └── UNet
│   ├── flow_matching.py  # Flow Matching 算法核心
│   │   └── FlowMatching
│   ├── data.py           # 数据加载模块
│   │   ├── load_celeba_data()
│   │   ├── load_imagenet_data()
│   │   └── load_face_data_from_folder()
│   ├── train.py          # 训练功能
│   │   └── train_flow_matching()
│   ├── utils.py          # 工具函数
│   │   ├── visualize_samples()
│   │   ├── count_parameters()
│   │   └── load_checkpoint()
│   └── config.py         # 配置管理
│       ├── MODEL_CONFIG
│       ├── TRAIN_CONFIG
│       ├── SAMPLE_CONFIG
│       └── DATA_CONFIG
│
├── scripts/              # 可执行脚本
│   ├── __init__.py
│   ├── train.py         # 训练脚本
│   ├── generate.py      # 生成样本脚本
│   ├── visualize.py     # 可视化脚本（GIF动画）
│   └── serve.py         # Web服务器脚本
│
├── docs/                 # 文档目录
│   ├── STRUCTURE.md     # 本文件
│   ├── QUICK_START.md   # 快速开始指南
│   └── download_celeba_manual.md  # 手动下载指南
│
├── data/                 # 数据目录（自动创建）
│   ├── celeb/           # CelebA数据集
│   └── faces/           # 自定义人脸数据集
│
├── results/              # 结果目录（自动创建）
│   ├── checkpoint_*.pt  # 模型检查点
│   ├── samples_*.png    # 生成的样本
│   └── animations/      # 动画文件
│
├── requirements.txt      # Python依赖
├── setup.py             # 安装脚本
├── .gitignore           # Git忽略文件
└── README.md            # 主文档
```

## 模块说明

### flowmatching/ - 核心包

**models.py**
- `TimeEmbedding`: 时间步嵌入
- `ResidualBlock`: 残差块（带时间条件）
- `UNet`: U-Net架构，用于预测速度场

**flow_matching.py**
- `FlowMatching`: Flow Matching算法主类
  - `train_step()`: 训练步骤
  - `sample()`: 生成样本（Euler方法）
  - `sample_ode()`: 生成样本（ODE求解器）

**data.py**
- `load_celeba_data()`: 加载CelebA数据集（自动下载）
- `load_imagenet_data()`: 加载ImageNet数据集
- `load_face_data_from_folder()`: 从文件夹加载自定义数据

**train.py**
- `train_flow_matching()`: 完整训练流程

**utils.py**
- `visualize_samples()`: 可视化样本网格
- `count_parameters()`: 统计模型参数
- `load_checkpoint()`: 加载检查点

**config.py**
- 集中管理所有配置参数

### scripts/ - 可执行脚本

**train.py**
- 主训练脚本
- 支持多种数据集
- 自动保存检查点和样本

**generate.py**
- 从训练好的模型生成样本
- 支持命令行参数

**visualize.py**
- 生成GIF动画展示生成过程
- 可视化从噪声到图像的演变

**serve.py**
- 启动Web服务器
- 交互式可视化界面

## 使用方式

### 作为模块使用

```python
from flowmatching import UNet, FlowMatching, load_celeba_data
from flowmatching.utils import visualize_samples

# 使用核心功能
```

### 作为脚本使用

```bash
# 方式1: 作为模块运行
python -m scripts.train
python -m scripts.generate

# 方式2: 直接运行
python scripts/train.py
python scripts/generate.py

# 方式3: 安装后使用命令行工具
pip install -e .
flowmatching-train
flowmatching-generate
```

## 设计原则

1. **模块化**: 核心功能与脚本分离
2. **可扩展**: 易于添加新模型或数据集
3. **配置化**: 所有参数集中在config.py
4. **标准化**: 遵循Python包标准结构
