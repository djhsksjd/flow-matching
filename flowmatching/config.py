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
    'batch_size': 16,  # Smaller batch size for ImageNet (larger images)
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
    'image_size': (128, 128),  # ImageNet image size (can use 64x64, 128x128, or 256x256)
}

# Data configuration
DATA_CONFIG = {
    'data_dir': './data/imagenet',  # ImageNet directory
    'batch_size': 16,
    'image_size': 128,  # Resize ImageNet images to 128x128 (standard is 224, but 128 is more memory efficient)
    'dataset_type': 'imagenet',  # Options: 'imagenet', 'celeba', 'face_folder'
}

# Paths
PATHS = {
    'results_dir': 'results',
    'checkpoints_dir': 'results',
}
