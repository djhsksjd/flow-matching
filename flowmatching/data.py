"""
Data Loading Utilities

This module provides functions for loading and preprocessing datasets.
"""

from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_mnist_data(batch_size=128, data_dir='./data'):
    """
    Load MNIST dataset.
    
    Args:
        batch_size: Batch size for data loaders
        data_dir: Directory to store/download MNIST dataset
    
    Returns:
        train_loader: DataLoader for training set
        test_loader: DataLoader for test set
    """
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    
    train_dataset = datasets.MNIST(
        root=data_dir,
        train=True,
        download=True,
        transform=transform
    )
    
    test_dataset = datasets.MNIST(
        root=data_dir,
        train=False,
        download=True,
        transform=transform
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True
    )
    
    return train_loader, test_loader
