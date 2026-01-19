"""
Configuration File

This module contains default configuration parameters for Flow Matching.
"""

# Model configuration
MODEL_CONFIG = {
    'in_channels': 1,
    'time_emb_dim': 128,
    'base_channels': 64,
}

# Training configuration
TRAIN_CONFIG = {
    'batch_size': 128,
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
    'image_size': (28, 28),
}

# Data configuration
DATA_CONFIG = {
    'data_dir': './data',
    'batch_size': 128,
}

# Paths
PATHS = {
    'results_dir': 'results',
    'checkpoints_dir': 'results',
}
