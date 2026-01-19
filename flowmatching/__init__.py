"""
Flow Matching Package

A modular implementation of Flow Matching for generative modeling.
"""

from .models import UNet, ResidualBlock, TimeEmbedding
from .flow_matching import FlowMatching
from .data import load_mnist_data
from .train import train_flow_matching
from .utils import (
    visualize_samples,
    save_samples_as_images,
    save_samples_as_numpy,
    count_parameters,
    load_checkpoint
)

__version__ = "1.0.0"
__all__ = [
    'UNet',
    'ResidualBlock',
    'TimeEmbedding',
    'FlowMatching',
    'load_mnist_data',
    'train_flow_matching',
    'visualize_samples',
    'save_samples_as_images',
    'save_samples_as_numpy',
    'count_parameters',
    'load_checkpoint',
]
