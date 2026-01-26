# API 文档

## 核心类

### FlowMatching

Flow Matching 算法主类。

```python
from flowmatching import FlowMatching

flow_matching = FlowMatching(model, device='cuda')
```

#### 方法

- `train_step(x1, optimizer)`: 执行一步训练
- `sample(num_samples, num_steps, image_size, channels)`: 生成样本
- `sample_ode(num_samples, num_steps, image_size, channels)`: 使用ODE求解器生成样本

### UNet

U-Net 模型架构。

```python
from flowmatching import UNet

model = UNet(in_channels=3, time_emb_dim=128, base_channels=64)
```

## 数据加载

### load_celeba_data

加载 CelebA 数据集。

```python
from flowmatching import load_celeba_data

train_loader = load_celeba_data(
    batch_size=32,
    data_dir='./data',
    image_size=64,
    split='train'
)
```

### load_imagenet_data

加载 ImageNet 数据集。

```python
from flowmatching import load_imagenet_data

train_loader = load_imagenet_data(
    batch_size=16,
    data_dir='./data/imagenet',
    image_size=128,
    split='train'
)
```

### load_face_data_from_folder

从文件夹加载自定义数据集。

```python
from flowmatching import load_face_data_from_folder

train_loader = load_face_data_from_folder(
    batch_size=32,
    data_dir='./data/faces',
    image_size=64
)
```

## 工具函数

### visualize_samples

可视化生成的样本。

```python
from flowmatching.utils import visualize_samples

visualize_samples(samples, save_path='samples.png', nrow=8)
```

### load_checkpoint

加载模型检查点。

```python
from flowmatching.utils import load_checkpoint

epoch, loss = load_checkpoint('checkpoint.pt', model, device='cuda')
```

### count_parameters

统计模型参数数量。

```python
from flowmatching.utils import count_parameters

num_params = count_parameters(model)
```

## 训练函数

### train_flow_matching

训练 Flow Matching 模型。

```python
from flowmatching import train_flow_matching

flow_matching = train_flow_matching(
    model,
    train_loader,
    num_epochs=50,
    lr=1e-4,
    device='cuda'
)
```
