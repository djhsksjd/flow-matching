"""
Flow Matching Package

A modular implementation of Flow Matching for generative modeling.
"""

from .models import UNet, ResidualBlock, TimeEmbedding
from .flow_matching import FlowMatching
from .data import load_face_data, load_celeba_data, load_face_data_from_folder, load_imagenet_data
from .train import train_flow_matching
from .utils import (
    visualize_samples,
    count_parameters,
    load_checkpoint
)

__version__ = "1.0.0"
__all__ = [
    'UNet',
    'ResidualBlock',
    'TimeEmbedding',
    'FlowMatching',
    'load_face_data',
    'load_celeba_data',
    'load_face_data_from_folder',
    'load_imagenet_data',
    'train_flow_matching',
    'visualize_samples',
    'count_parameters',
    'load_checkpoint',
]
