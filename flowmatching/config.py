"""
Configuration File

This module contains default configuration parameters for Flow Matching.
"""

# Model configuration
MODEL_CONFIG = {
    'in_channels': 3,  # RGB images
    'time_emb_dim': 128,
    'base_channels': 64,
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 32,  # Batch size for face images
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'save_every': 10,
    'sample_every': 5,
    'num_sample_steps': 50,
}

# Sampling configuration
SAMPLE_CONFIG = {
    'num_samples': 64,
    'num_steps': 100,
    'image_size': (64, 64),  # Face image size (can use 64x64, 128x128)
}

# Data configuration
DATA_CONFIG = {
    'data_dir': './data',  # Data directory (CelebA will be downloaded here automatically)
    'batch_size': 32,
    'image_size': 64,  # Resize face images to 64x64 (CelebA supports up to 218x178)
    'dataset_type': 'celeba',  # Options: 'celeba' (auto-download), 'face_folder', 'imagenet'
}

# Paths
PATHS = {
    'results_dir': 'results',
    'checkpoints_dir': 'results',
}
