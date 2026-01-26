# 快速启动指南

## 🚀 一键开始训练

根据你的检查结果，CelebA 数据集还没有下载。不用担心，运行训练脚本时会自动下载！

### 步骤 1: 直接开始训练（推荐）

```bash
python train_main.py
```

这会：
1. ✅ 自动下载 CelebA 数据集（约 1.3GB，首次需要一些时间）
2. ✅ 创建模型
3. ✅ 开始训练
4. ✅ 自动保存检查点和生成的样本

### 步骤 2: 或者先快速测试

如果想先快速验证一切是否正常：

```bash
# 快速测试（使用少量样本）
python quick_start.py
```

### 步骤 3: 检查环境

如果想检查所有组件：

```bash
# 完整测试
python test_model.py
```

## 📋 当前状态

根据你的检查结果：
- ✅ 配置正确：使用 CelebA 数据集
- ⏳ 数据集：将在首次运行时自动下载
- ✅ 代码：已准备就绪

## 🎯 下一步

直接运行：

```bash
python train_main.py
```

首次运行时会看到类似这样的输出：

```
Loading CelebA dataset (split: train)...
This may download ~1.3GB of data if not already present.
Downloading/loading CelebA dataset...
✅ Successfully loaded CelebA train set
   Total images: 202,599
```

然后训练就会自动开始！

## 💡 提示

- **下载时间**：首次下载 CelebA 需要一些时间（取决于网速）
- **磁盘空间**：确保有至少 2GB 可用空间
- **中断恢复**：如果下载中断，重新运行会自动恢复
- **GPU 加速**：如果有 GPU，训练会快很多

## 🎨 训练完成后

训练完成后，你可以：

1. **查看生成的样本**：
   ```bash
   python generate_samples.py
   ```

2. **可视化生成过程**：
   ```bash
   # GIF 动画
   python visualize_generation.py
   
   # 或 Web 界面
   python web_visualizer.py
   ```

现在就可以开始训练了！🎉
