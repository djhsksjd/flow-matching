# 使用指南

## 基本使用流程

### 1. 训练模型

```bash
# 使用默认配置（CelebA数据集）
python -m scripts.train
```

### 2. 生成样本

```bash
# 使用默认设置
python -m scripts.generate

# 自定义参数
python -m scripts.generate --num_samples 100 --num_steps 200
```

### 3. 可视化

```bash
# 生成GIF动画
python -m scripts.visualize

# 启动Web界面
python -m scripts.serve
```

## 配置修改

编辑 `flowmatching/config.py` 修改配置：

```python
# 修改数据集类型
DATA_CONFIG['dataset_type'] = 'face_folder'  # 或 'imagenet'

# 修改训练参数
TRAIN_CONFIG['num_epochs'] = 100
TRAIN_CONFIG['batch_size'] = 16

# 修改模型参数
MODEL_CONFIG['base_channels'] = 128  # 更大的模型
```

## 数据集切换

### 使用CelebA（默认）

无需修改，直接运行：
```bash
python -m scripts.train
```

### 使用自定义数据集

1. 修改配置：
```python
DATA_CONFIG['dataset_type'] = 'face_folder'
```

2. 准备数据：
```bash
mkdir data/faces
# 将图像放入 data/faces/
```

3. 开始训练：
```bash
python -m scripts.train
```

### 使用ImageNet

1. 下载并组织ImageNet数据集
2. 修改配置：
```python
DATA_CONFIG['dataset_type'] = 'imagenet'
DATA_CONFIG['data_dir'] = './data/imagenet'
```

3. 开始训练

## 高级用法

### 从检查点继续训练

修改 `scripts/train.py` 添加加载检查点的功能。

### 自定义模型架构

修改 `flowmatching/models.py` 中的 `UNet` 类。

### 添加新数据集

在 `flowmatching/data.py` 中添加新的加载函数。
