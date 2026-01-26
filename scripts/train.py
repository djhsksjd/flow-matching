"""
Train Flow Matching Model

Main training script for Flow Matching generative model.
Supports CelebA, ImageNet, and custom face datasets.

Usage:
    python -m scripts.train
    python scripts/train.py
"""

import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from flowmatching import UNet, FlowMatching, load_celeba_data, load_imagenet_data, load_face_data, train_flow_matching
from flowmatching.utils import visualize_samples, count_parameters
from flowmatching.config import MODEL_CONFIG, TRAIN_CONFIG, DATA_CONFIG, PATHS, SAMPLE_CONFIG


def main():
    """Main training function."""
    print("=" * 60)
    print("Flow Matching for Face Image Generation")
    print("=" * 60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Create results directory
    results_dir = PATHS['results_dir']
    os.makedirs(results_dir, exist_ok=True)
    print(f"Results will be saved to: {results_dir}/")
    
    # Load data based on dataset type
    dataset_type = DATA_CONFIG.get('dataset_type', 'celeba')
    print(f"\n{'='*60}")
    print(f"Loading {dataset_type.upper()} dataset...")
    print(f"{'='*60}")
    
    try:
        if dataset_type == 'celeba':
            train_loader = load_celeba_data(
                batch_size=DATA_CONFIG['batch_size'],
                data_dir=DATA_CONFIG['data_dir'],
                image_size=DATA_CONFIG['image_size'],
                split='train'
            )
            print(f"✅ Training samples: {len(train_loader.dataset):,}")
            print(f"✅ Image size: {DATA_CONFIG['image_size']}x{DATA_CONFIG['image_size']}")
        elif dataset_type == 'imagenet':
            train_loader = load_imagenet_data(
                batch_size=DATA_CONFIG['batch_size'],
                data_dir=DATA_CONFIG['data_dir'],
                image_size=DATA_CONFIG['image_size'],
                split='train'
            )
            print(f"✅ Training samples: {len(train_loader.dataset):,}")
            print(f"✅ Number of classes: {len(train_loader.dataset.classes)}")
            print(f"✅ Image size: {DATA_CONFIG['image_size']}x{DATA_CONFIG['image_size']}")
        elif dataset_type == 'face_folder':
            train_loader = load_face_data(
                batch_size=DATA_CONFIG['batch_size'],
                data_dir=DATA_CONFIG['data_dir'],
                image_size=DATA_CONFIG['image_size'],
                use_celeba=False
            )
            print(f"✅ Training samples: {len(train_loader.dataset):,}")
            print(f"✅ Image size: {DATA_CONFIG['image_size']}x{DATA_CONFIG['image_size']}")
        else:
            raise ValueError(f"Unknown dataset_type: {dataset_type}. Use 'celeba', 'face_folder', or 'imagenet'")
        
    except Exception as e:
        error_msg = str(e)
        print(f"\n❌ Error loading dataset: {error_msg}")
        
        if dataset_type == 'imagenet':
            print("\nImageNet setup instructions:")
            print("  1. Download ImageNet dataset")
            print("  2. Organize as: data/imagenet/train/class1/..., class2/..., etc.")
        elif dataset_type == 'celeba':
            if 'too many users' in error_msg.lower() or 'viewed or downloaded' in error_msg.lower():
                print("\n💡 Quick Fix: Switch to custom face folder")
                print("   Edit flowmatching/config.py:")
                print("     DATA_CONFIG['dataset_type'] = 'face_folder'")
                print("   Then place face images in: data/faces/")
        return
    
    # Create model
    print("\nCreating model...")
    model = UNet(**MODEL_CONFIG)
    num_params = count_parameters(model)
    print(f"Model parameters: {num_params:,}")
    print(f"Input channels: {MODEL_CONFIG['in_channels']} (RGB)")
    
    # Train
    print("\nStarting training...")
    print(f"Training configuration:")
    print(f"  Dataset: {dataset_type}")
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
    
    num_samples = SAMPLE_CONFIG['num_samples']
    num_steps = SAMPLE_CONFIG['num_steps']
    image_size = SAMPLE_CONFIG['image_size']
    
    print(f"Generating {num_samples} samples with {num_steps} steps...")
    samples = flow_matching.sample(
        num_samples, 
        num_steps=num_steps,
        image_size=image_size,
        channels=MODEL_CONFIG['in_channels']
    )
    
    # Save grid visualization only
    grid_path = os.path.join(results_dir, 'final_samples_grid.png')
    visualize_samples(
        samples, 
        save_path=grid_path, 
        nrow=8,
        title=f"Final Generated Face Samples ({dataset_type.upper()}) - Flow Matching"
    )
    
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
    print(f"  - {final_checkpoint_path}")
    print("\nTo generate more samples, use:")
    print("  python -m scripts.generate")


if __name__ == '__main__':
    main()
