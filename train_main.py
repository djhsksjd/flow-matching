"""
Main Training Script for Flow Matching on MNIST

This script trains a Flow Matching model to generate MNIST digit images.
Usage: python train_main.py
"""

import torch
import os
from flowmatching import UNet, FlowMatching, load_mnist_data, train_flow_matching
from flowmatching.utils import (
    visualize_samples,
    save_samples_as_images,
    save_samples_as_numpy,
    count_parameters
)
from flowmatching.config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, PATHS, SAMPLE_CONFIG


def main():
    """Main training function."""
    print("=" * 60)
    print("Flow Matching for MNIST Image Generation")
    print("=" * 60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create results directory
    results_dir = PATHS['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(os.path.join(results_dir, 'final_samples'), exist_ok=True)
    print(f"Results will be saved to: {results_dir}/")
    
    # Load data
    print("\nLoading MNIST dataset...")
    train_loader, test_loader = load_mnist_data(
        batch_size=DATA_CONFIG['batch_size'],
        data_dir=DATA_CONFIG['data_dir']
    )
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Test samples: {len(test_loader.dataset)}")
    
    # Create model
    print("\nCreating model...")
    model = UNet(**MODEL_CONFIG)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    
    # Train
    print("\nStarting training...")
    print(f"Training configuration:")
    print(f"  Epochs: {TRAIN_CONFIG['num_epochs']}")
    print(f"  Learning rate: {TRAIN_CONFIG['learning_rate']}")
    print(f"  Batch size: {TRAIN_CONFIG['batch_size']}")
    print(f"  Save checkpoint every: {TRAIN_CONFIG['save_every']} epochs")
    print(f"  Generate samples every: {TRAIN_CONFIG['sample_every']} epochs")
    
    flow_matching = train_flow_matching(
        model,
        train_loader,
        num_epochs=TRAIN_CONFIG['num_epochs'],
        lr=TRAIN_CONFIG['learning_rate'],
        device=device,
        save_dir=results_dir,
        save_every=TRAIN_CONFIG['save_every'],
        sample_every=TRAIN_CONFIG['sample_every'],
        num_sample_steps=TRAIN_CONFIG['num_sample_steps']
    )
    
    # Final sampling and saving
    print("\n" + "=" * 60)
    print("Generating final samples...")
    print("=" * 60)
    
    # Generate samples with higher quality (more steps)
    num_samples = SAMPLE_CONFIG['num_samples']
    num_steps = SAMPLE_CONFIG['num_steps']
    
    print(f"Generating {num_samples} samples with {num_steps} steps...")
    samples = flow_matching.sample(num_samples, num_steps=num_steps)
    
    # Save grid visualization
    grid_path = os.path.join(results_dir, 'final_samples', 'final_samples_grid.png')
    visualize_samples(
        samples, 
        save_path=grid_path, 
        nrow=8,
        title=f"Final Generated Samples (Flow Matching)"
    )
    
    # Save individual images
    individual_dir = os.path.join(results_dir, 'final_samples', 'individual')
    save_samples_as_images(
        samples, 
        individual_dir, 
        prefix='generated',
        start_idx=0
    )
    
    # Save as numpy array
    numpy_path = os.path.join(results_dir, 'final_samples', 'final_samples.npy')
    save_samples_as_numpy(samples, numpy_path)
    
    # Save final checkpoint
    final_checkpoint_path = os.path.join(results_dir, 'checkpoint_final.pt')
    torch.save({
        'epoch': TRAIN_CONFIG['num_epochs'],
        'model_state_dict': model.state_dict(),
        'config': MODEL_CONFIG,
    }, final_checkpoint_path)
    print(f"Saved final checkpoint to {final_checkpoint_path}")
    
    print("\n" + "=" * 60)
    print("Training completed!")
    print("=" * 60)
    print(f"\nResults saved to: {results_dir}/")
    print("\nGenerated files:")
    print(f"  - {grid_path}")
    print(f"  - {numpy_path}")
    print(f"  - {individual_dir}/ (individual images)")
    print(f"  - {final_checkpoint_path}")
    print("\nTo generate more samples, use:")
    print("  python generate_samples.py")


if __name__ == '__main__':
    main()
