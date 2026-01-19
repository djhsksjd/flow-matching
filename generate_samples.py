"""
Generate Samples from Trained Model

This script loads a trained Flow Matching model and generates new samples.
Usage: python generate_samples.py [--checkpoint PATH] [--num_samples N] [--num_steps N]
"""

import torch
import os
import argparse
from flowmatching import UNet, FlowMatching
from flowmatching.utils import (
    visualize_samples,
    save_samples_as_images,
    save_samples_as_numpy,
    load_checkpoint
)
from flowmatching.config import MODEL_CONFIG, SAMPLE_CONFIG, PATHS


def main():
    parser = argparse.ArgumentParser(description='Generate samples from trained Flow Matching model')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (default: latest checkpoint in results/)')
    parser.add_argument('--num_samples', type=int, default=SAMPLE_CONFIG['num_samples'],
                       help=f'Number of samples to generate (default: {SAMPLE_CONFIG["num_samples"]})')
    parser.add_argument('--num_steps', type=int, default=SAMPLE_CONFIG['num_steps'],
                       help=f'Number of integration steps (default: {SAMPLE_CONFIG["num_steps"]})')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory (default: results/generated_samples)')
    parser.add_argument('--save_individual', action='store_true',
                       help='Save individual images')
    parser.add_argument('--save_numpy', action='store_true',
                       help='Save as numpy array')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Generate Samples from Flow Matching Model")
    print("=" * 60)
    
    # Device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Find checkpoint
    if args.checkpoint is None:
        results_dir = PATHS['results_dir']
        # Look for final checkpoint first
        final_checkpoint = os.path.join(results_dir, 'checkpoint_final.pt')
        if os.path.exists(final_checkpoint):
            args.checkpoint = final_checkpoint
            print(f"Found final checkpoint: {args.checkpoint}")
        else:
            # Find latest checkpoint
            checkpoints = [f for f in os.listdir(results_dir) if f.startswith('checkpoint_epoch_')]
            if checkpoints:
                epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoints]
                latest_idx = epochs.index(max(epochs))
                args.checkpoint = os.path.join(results_dir, checkpoints[latest_idx])
                print(f"Found latest checkpoint: {args.checkpoint}")
            else:
                raise FileNotFoundError(f"No checkpoint found in {results_dir}. Please train a model first or specify --checkpoint.")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(PATHS['results_dir'], 'generated_samples')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create model
    print("\nCreating model...")
    model = UNet(**MODEL_CONFIG)
    
    # Load checkpoint
    print(f"\nLoading checkpoint: {args.checkpoint}")
    epoch, loss = load_checkpoint(args.checkpoint, model, device=device)
    print(f"Loaded checkpoint from epoch {epoch} (loss: {loss:.6f})")
    
    # Create FlowMatching instance
    flow_matching = FlowMatching(model, device)
    
    # Generate samples
    print(f"\nGenerating {args.num_samples} samples with {args.num_steps} steps...")
    samples = flow_matching.sample(args.num_samples, num_steps=args.num_steps)
    
    # Save grid visualization
    grid_path = os.path.join(args.output_dir, 'generated_samples_grid.png')
    visualize_samples(
        samples,
        save_path=grid_path,
        nrow=8,
        title=f"Generated Samples (Epoch {epoch})"
    )
    
    # Save individual images
    if args.save_individual:
        individual_dir = os.path.join(args.output_dir, 'individual')
        save_samples_as_images(
            samples,
            individual_dir,
            prefix='generated',
            start_idx=0
        )
    
    # Save as numpy array
    if args.save_numpy:
        numpy_path = os.path.join(args.output_dir, 'generated_samples.npy')
        save_samples_as_numpy(samples, numpy_path)
    
    print("\n" + "=" * 60)
    print("Generation completed!")
    print("=" * 60)
    print(f"\nResults saved to: {args.output_dir}/")
    print(f"  - Grid: {grid_path}")
    if args.save_individual:
        print(f"  - Individual images: {os.path.join(args.output_dir, 'individual')}/")
    if args.save_numpy:
        print(f"  - Numpy array: {numpy_path}")


if __name__ == '__main__':
    main()
