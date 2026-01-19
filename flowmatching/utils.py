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
    Visualize generated samples and save as grid image.
    
    Args:
        samples: Tensor of shape [num_samples, channels, height, width]
        save_path: Path to save the figure (optional)
        nrow: Number of images per row
        figsize: Figure size (width, height)
        title: Optional title for the plot
    """
    samples = samples.cpu()
    num_samples = samples.shape[0]
    
    # Convert to numpy and handle channels
    if samples.dim() == 4:
        if samples.shape[1] == 3:  # RGB
            # Convert from [B, C, H, W] to [B, H, W, C] for imshow
            samples_np = samples.permute(0, 2, 3, 1).numpy()
            cmap = None
        else:  # Grayscale
            samples_np = samples.squeeze(1).numpy()  # [B, H, W]
            cmap = 'gray'
    else:
        samples_np = samples.numpy()
        cmap = 'gray'
    
    ncol = (num_samples + nrow - 1) // nrow
    fig, axes = plt.subplots(nrow, ncol, figsize=figsize)
    
    # Handle both 2D and 1D axes arrays
    if nrow == 1:
        axes = axes.reshape(1, -1)
    if ncol == 1:
        axes = axes.reshape(-1, 1)
    
    axes = axes.flatten()
    
    for i in range(num_samples):
        if cmap is None:  # RGB
            axes[i].imshow(samples_np[i])
        else:  # Grayscale
            axes[i].imshow(samples_np[i], cmap=cmap)
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
