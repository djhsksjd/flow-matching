"""
Utility Functions

This module contains utility functions for visualization and other helper functions.
"""

import matplotlib.pyplot as plt
import torch
import numpy as np
import os
from PIL import Image


def visualize_samples(samples, save_path=None, nrow=8, figsize=(12, 12), title=None):
    """
    Visualize generated samples and save as image.
    
    Args:
        samples: Tensor of shape [num_samples, channels, height, width]
        save_path: Path to save the figure (optional)
        nrow: Number of images per row
        figsize: Figure size (width, height)
        title: Optional title for the plot
    """
    samples = samples.cpu()
    num_samples = samples.shape[0]
    
    ncol = (num_samples + nrow - 1) // nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    
    # Handle both 2D and 1D axes arrays
    if nrow == 1:
        axes = axes.reshape(1, -1)
    if ncol == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()
    
    for i in range(num_samples):
        axes[i].imshow(samples[i].squeeze(), cmap='gray')
        axes[i].axis('off')
    
    for i in range(num_samples, len(axes)):
        axes[i].axis('off')
    
    if title:
        fig.suptitle(title, fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved samples grid to {save_path}")
    
    plt.close()
    return save_path


def save_samples_as_images(samples, save_dir, prefix='sample', start_idx=0, format='png'):
    """
    Save individual generated samples as separate image files.
    
    Args:
        samples: Tensor of shape [num_samples, channels, height, width]
        save_dir: Directory to save individual images
        prefix: Prefix for image filenames
        start_idx: Starting index for filenames
        format: Image format ('png', 'jpg', etc.)
    
    Returns:
        List of saved file paths
    """
    os.makedirs(save_dir, exist_ok=True)
    samples = samples.cpu().numpy()
    saved_paths = []
    
    for i in range(samples.shape[0]):
        # Convert to uint8 [0, 255]
        img = samples[i].squeeze()
        img = np.clip(img * 255, 0, 255).astype(np.uint8)
        
        # Create PIL Image
        if len(img.shape) == 2:
            img_pil = Image.fromarray(img, mode='L')
        else:
            img_pil = Image.fromarray(img.transpose(1, 2, 0), mode='RGB')
        
        # Save
        filename = f"{prefix}_{start_idx + i:04d}.{format}"
        filepath = os.path.join(save_dir, filename)
        img_pil.save(filepath)
        saved_paths.append(filepath)
    
    print(f"Saved {len(saved_paths)} individual images to {save_dir}")
    return saved_paths


def save_samples_as_numpy(samples, save_path):
    """
    Save generated samples as numpy array file.
    
    Args:
        samples: Tensor of shape [num_samples, channels, height, width]
        save_path: Path to save numpy file (.npy)
    """
    samples = samples.cpu().numpy()
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    np.save(save_path, samples)
    print(f"Saved samples array to {save_path}")


def count_parameters(model):
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
    
    Returns:
        num_params: Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def load_checkpoint(checkpoint_path, model, optimizer=None, device='cpu'):
    """
    Load model checkpoint.
    
    Args:
        checkpoint_path: Path to checkpoint file
        model: Model to load weights into
        optimizer: Optional optimizer to load state
        device: Device to load checkpoint on
    
    Returns:
        epoch: Epoch number from checkpoint
        loss: Loss value from checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    if optimizer is not None and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    loss = checkpoint.get('loss', 0.0)
    
    return epoch, loss
