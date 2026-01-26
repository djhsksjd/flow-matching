"""
Data Loading Utilities

This module provides functions for loading and preprocessing datasets.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
import os


def load_celeba_data(batch_size=32, data_dir='./data', image_size=64, split='train'):
    """
    Load CelebA dataset for face generation.
    
    CelebA is a large-scale face attributes dataset with more than 200K celebrity images.
    The dataset will be automatically downloaded (~1.3GB) if not present.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/download CelebA dataset
        image_size: Size to resize images to (default: 64x64)
        split: 'train', 'valid', or 'test'
    
    Returns:
        data_loader: DataLoader for the specified split
    """
    print(f"Loading CelebA dataset (split: {split})...")
    print(f"This may download ~1.3GB of data if not already present.")
    print(f"Dataset will be stored in: {os.path.abspath(data_dir)}")
    
    # Data augmentation for training
    if split == 'train':
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.1)),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    
    try:
        # Use torchvision's CelebA dataset (auto-downloads if needed)
        print("Downloading/loading CelebA dataset...")
        dataset = datasets.CelebA(
            root=data_dir,
            split=split,
            download=True,  # Automatically download if not present
            transform=transform
        )
        print(f"✅ Successfully loaded CelebA {split} set")
        print(f"   Total images: {len(dataset):,}")
    except Exception as e:
        error_msg = str(e)
        print(f"❌ Error loading CelebA dataset: {error_msg}")
        
        # Check for specific error messages
        if 'gdown' in error_msg.lower() and 'required' in error_msg.lower():
            print("\n" + "="*60)
            print("⚠️  Missing dependency: gdown")
            print("="*60)
            print("CelebA dataset requires 'gdown' to download from Google Drive.")
            print("\nTo fix this, run:")
            print("  pip install gdown")
            print("\nOr install all requirements:")
            print("  pip install -r requirements.txt")
        elif 'too many users' in error_msg.lower() or 'viewed or downloaded' in error_msg.lower():
            print("\n" + "="*60)
            print("⚠️  Google Drive Download Limit Reached")
            print("="*60)
            print("The CelebA dataset file has reached Google Drive's download limit.")
            print("This is a temporary restriction that usually resolves within 24 hours.")
            print("\nAlternative Solutions:")
            print("\n1. Wait and Retry (Recommended):")
            print("   - Wait 24 hours and try again")
            print("   - The download limit resets automatically")
            print("\n2. Manual Download:")
            print("   - Visit: https://drive.google.com/drive/folders/0B7EVK8r0v71pZjFTYXZWM3FlRnM")
            print("   - Download the 'img_align_celeba' folder manually")
            print("   - Extract to: data/celeb/img_align_celeba/")
            print("\n3. Use Alternative Dataset:")
            print("   - Switch to 'face_folder' dataset type in config.py")
            print("   - Place your own face images in data/faces/")
            print("\n4. Try Alternative Download Method:")
            print("   - Use a VPN or different network")
            print("   - Try downloading at a different time")
        else:
            print("\nTroubleshooting:")
            print("1. Check your internet connection (first download requires ~1.3GB)")
            print("2. Ensure you have enough disk space")
            print("3. Try again - download may resume if interrupted")
        raise
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,  # More workers for faster loading
        pin_memory=True,
        drop_last=True  # Drop last incomplete batch
    )
    
    return data_loader


def load_face_data_from_folder(batch_size=32, data_dir='./data/faces', image_size=64):
    """
    Load face images from a folder structure.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory containing face images
        image_size: Size to resize images to (default: 64x64)
    
    Returns:
        data_loader: DataLoader for face images
    """
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
    ])
    
    # Use ImageFolder which expects subdirectories, or create a simple dataset
    if os.path.isdir(data_dir):
        # Check if it's a structured folder (with subdirs) or flat folder
        subdirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        
        if subdirs:
            # Structured folder (ImageFolder format)
            dataset = ImageFolder(root=data_dir, transform=transform)
        else:
            # Flat folder - create a simple dataset
            from torch.utils.data import Dataset
            from PIL import Image
            import glob
            
            class FlatImageDataset(Dataset):
                def __init__(self, folder_path, transform=None):
                    self.image_paths = glob.glob(os.path.join(folder_path, '*.jpg')) + \
                                     glob.glob(os.path.join(folder_path, '*.png')) + \
                                     glob.glob(os.path.join(folder_path, '*.jpeg'))
                    self.transform = transform
                
                def __len__(self):
                    return len(self.image_paths)
                
                def __getitem__(self, idx):
                    img_path = self.image_paths[idx]
                    image = Image.open(img_path).convert('RGB')
                    if self.transform:
                        image = self.transform(image)
                    return image, 0  # Return dummy label
            
            dataset = FlatImageDataset(data_dir, transform=transform)
    else:
        raise ValueError(f"Data directory not found: {data_dir}")
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True,
        drop_last=True
    )
    
    return data_loader


def load_imagenet_data(batch_size=32, data_dir='./data/imagenet', image_size=128, split='train'):
    """
    Load ImageNet dataset.
    
    ImageNet directory structure should be:
        imagenet/
            train/
                class1/
                    img1.JPEG
                    img2.JPEG
                    ...
                class2/
                    ...
            val/
                class1/
                    ...
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory containing ImageNet dataset (should have 'train' and/or 'val' subdirectories)
        image_size: Size to resize images to (default: 128x128)
        split: 'train' or 'val'
    
    Returns:
        data_loader: DataLoader for the specified split
    """
    # ImageNet standard preprocessing with data augmentation for training
    if split == 'train':
        transform = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=(0.8, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05),
            transforms.ToTensor(),
        ])
    else:
        transform = transforms.Compose([
            transforms.Resize(int(image_size * 1.15)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
        ])
    
    split_dir = os.path.join(data_dir, split)
    
    if not os.path.isdir(split_dir):
        raise ValueError(
            f"ImageNet {split} directory not found: {split_dir}\n"
            f"Expected structure: {data_dir}/train/ and/or {data_dir}/val/\n"
            "Each subdirectory should contain class folders with images."
        )
    
    dataset = ImageFolder(root=split_dir, transform=transform)
    
    print(f"Found {len(dataset.classes)} classes in ImageNet {split} set")
    print(f"Total images: {len(dataset)}")
    
    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=(split == 'train'),
        num_workers=4,  # More workers for ImageNet
        pin_memory=True,
        drop_last=True,
        persistent_workers=True  # Keep workers alive between epochs
    )
    
    return data_loader


def load_face_data(batch_size=32, data_dir='./data', image_size=64, use_celeba=True):
    """
    Load face dataset (CelebA by default, or from folder).
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/download dataset
        image_size: Size to resize images to
        use_celeba: If True, use CelebA dataset; if False, load from folder
    
    Returns:
        train_loader: DataLoader for training set
    """
    if use_celeba:
        train_loader = load_celeba_data(
            batch_size=batch_size,
            data_dir=data_dir,
            image_size=image_size,
            split='train'
        )
    else:
        face_dir = os.path.join(data_dir, 'faces') if not os.path.exists(data_dir + '/celeb') else data_dir
        train_loader = load_face_data_from_folder(
            batch_size=batch_size,
            data_dir=face_dir,
            image_size=image_size
        )
    
    return train_loader
