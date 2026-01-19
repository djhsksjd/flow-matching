# Flow Matching Package

Modular implementation of Flow Matching for generative modeling on MNIST dataset.

## Package Structure

```
flowmatching/
├── __init__.py          # Package initialization and exports
├── models.py            # Neural network architectures (UNet, ResidualBlock, TimeEmbedding)
├── flow_matching.py     # Flow Matching algorithm implementation
├── data.py              # Data loading utilities
├── train.py             # Training functions
├── utils.py             # Visualization and helper utilities
└── config.py            # Configuration parameters
```

## Module Descriptions

### models.py
Contains the neural network architectures:
- `TimeEmbedding`: Sinusoidal time step embedding
- `ResidualBlock`: Residual block with time conditioning
- `UNet`: U-Net architecture for velocity field prediction

### flow_matching.py
Implements the Flow Matching algorithm:
- `FlowMatching`: Main class for flow matching training and sampling
  - `train_step()`: Perform one training step
  - `sample()`: Generate samples using Euler method
  - `sample_ode()`: Generate samples using ODE solver

### data.py
Data loading utilities:
- `load_mnist_data()`: Load and preprocess MNIST dataset

### train.py
Training utilities:
- `train_flow_matching()`: Complete training loop with checkpointing and sampling

### utils.py
Utility functions:
- `visualize_samples()`: Visualize generated samples
- `count_parameters()`: Count model parameters
- `load_checkpoint()`: Load model checkpoints

### config.py
Default configuration parameters for models, training, sampling, and data.

## Usage

### Basic Usage

```python
from flowmatching import UNet, FlowMatching, load_mnist_data, train_flow_matching

# Load data
train_loader, test_loader = load_mnist_data(batch_size=128)

# Create model
model = UNet(in_channels=1, time_emb_dim=128, base_channels=64)

# Train
flow_matching = train_flow_matching(
    model,
    train_loader,
    num_epochs=50,
    lr=1e-4
)

# Generate samples
samples = flow_matching.sample(num_samples=64, num_steps=100)
```

### Using Configuration

```python
from flowmatching import UNet
from flowmatching.config import MODEL_CONFIG

model = UNet(**MODEL_CONFIG)
```

### Training from Command Line

```bash
python train_main.py
```

## Installation

Install required packages:

```bash
pip install -r requirements.txt
```

## References

- "Flow Matching for Generative Modeling" (Lipman et al., 2023)
