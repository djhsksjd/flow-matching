# 手动下载 CelebA 数据集指南

如果遇到 Google Drive 下载限制，可以手动下载 CelebA 数据集。

## 方法 1: 从官方链接下载

1. **访问下载页面**：
   - 官方页面: http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html
   - 或直接访问: https://drive.google.com/drive/folders/0B7EVK8r0v71pZjFTYXZWM3FlRnM

2. **下载文件**：
   - 下载 `img_align_celeba.zip` (约 1.3GB)
   - 或下载整个 `img_align_celeba` 文件夹

3. **解压和组织**：
   ```bash
   # 创建目录
   mkdir -p data/celeb
   
   # 解压到正确位置
   # Windows: 解压 img_align_celeba.zip 到 data/celeb/
   # 最终结构应该是: data/celeb/img_align_celeba/*.jpg
   ```

4. **验证结构**：
   ```
   data/
   └── celeb/
       └── img_align_celeba/
           ├── 000001.jpg
           ├── 000002.jpg
           └── ... (202,599 images)
   ```

## 方法 2: 使用其他下载源

如果 Google Drive 不可用，可以尝试：

1. **Kaggle**: https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
2. **其他镜像站点**

## 方法 3: 使用自定义数据集

如果下载困难，可以切换到自定义人脸数据集：

1. **修改配置** (`flowmatching/config.py`):
   ```python
   DATA_CONFIG = {
       'dataset_type': 'face_folder',  # 改为 face_folder
       'data_dir': './data',
       'batch_size': 32,
       'image_size': 64,
   }
   ```

2. **准备数据**：
   ```bash
   # 创建目录
   mkdir -p data/faces
   
   # 将你的人脸图像放入 data/faces/
   # 支持 .jpg, .png, .jpeg 格式
   ```

3. **开始训练**：
   ```bash
   python train_main.py
   ```

## 验证下载

下载完成后，运行检查脚本：

```bash
python check_dataset.py
```

应该显示：
```
✅ CelebA directory exists
✅ Found 202,599 images
```

## 注意事项

- **磁盘空间**: 确保有至少 2GB 可用空间
- **解压时间**: 解压可能需要几分钟
- **文件数量**: 应该有 202,599 张图像文件
