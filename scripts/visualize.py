"""
Visualize Generation Process

Create animations showing the generation process from noise to final image.

Usage:
    python -m scripts.visualize
    python scripts/visualize.py [--checkpoint PATH] [--num_samples N] [--num_steps N]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from flowmatching import UNet, FlowMatching
from flowmatching.utils import load_checkpoint
from flowmatching.config import MODEL_CONFIG, SAMPLE_CONFIG, PATHS


def generate_intermediate_steps(flow_matching, num_samples=1, num_steps=100, image_size=(64, 64), channels=3):
    """Generate intermediate steps during sampling."""
    flow_matching.model.eval()
    x = torch.randn(num_samples, channels, image_size[0], image_size[1], device=flow_matching.device)
    steps = [x.cpu().clone()]
    dt = 1.0 / num_steps
    
    with torch.no_grad():
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=flow_matching.device)
            v = flow_matching.model(x, t)
            x = x + dt * v
            x = torch.clamp(x, 0, 1)
            if (i + 1) % max(1, num_steps // 50) == 0 or i == num_steps - 1:
                steps.append(x.cpu().clone())
    
    return steps


def create_animation(steps, save_path='generation_animation.gif', fps=10, title="Flow Matching Generation Process"):
    """Create an animation from intermediate generation steps."""
    num_samples = steps[0].shape[0]
    num_steps = len(steps)
    
    frames = []
    for step in steps:
        step_np = step.numpy()
        if step_np.shape[1] == 3:  # RGB
            step_np = step_np.transpose(0, 2, 3, 1)
            frames.append(step_np)
        else:  # Grayscale
            step_np = step_np.squeeze(1)
            frames.append(step_np)
    
    fig, axes = plt.subplots(1, num_samples, figsize=(4 * num_samples, 4))
    if num_samples == 1:
        axes = [axes]
    
    ims = []
    for idx, ax in enumerate(axes):
        ax.axis('off')
        if frames[0].ndim == 3:  # RGB
            im = ax.imshow(frames[0][idx], animated=True)
        else:  # Grayscale
            im = ax.imshow(frames[0][idx], cmap='gray', animated=True)
        ims.append(im)
    
    fig.suptitle(title, fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    def animate(frame):
        for idx, (ax, im) in enumerate(zip(axes, ims)):
            if frames[frame].ndim == 3:  # RGB
                im.set_array(frames[frame][idx])
            else:  # Grayscale
                im.set_array(frames[frame][idx])
        return ims
    
    anim = animation.FuncAnimation(
        fig, animate, frames=num_steps, interval=1000/fps, blit=True, repeat=True
    )
    
    print(f"Saving animation to {save_path}...")
    anim.save(save_path, writer='pillow', fps=fps)
    print(f"✅ Animation saved!")
    
    plt.close()
    return anim


def main():
    parser = argparse.ArgumentParser(description='Visualize Flow Matching generation process')
    parser.add_argument('--checkpoint', type=str, default=None,
                       help='Path to checkpoint file (default: latest checkpoint)')
    parser.add_argument('--num_samples', type=int, default=1,
                       help='Number of samples to generate (default: 1)')
    parser.add_argument('--num_steps', type=int, default=100,
                       help='Number of integration steps (default: 100)')
    parser.add_argument('--output_dir', type=str, default='results/animations',
                       help='Output directory for animations')
    parser.add_argument('--fps', type=int, default=10,
                       help='Frames per second for animation (default: 10)')
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Flow Matching Generation Visualization")
    print("=" * 60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Find checkpoint
    if args.checkpoint is None:
        results_dir = PATHS['results_dir']
        final_checkpoint = os.path.join(results_dir, 'checkpoint_final.pt')
        if os.path.exists(final_checkpoint):
            args.checkpoint = final_checkpoint
        else:
            checkpoints = [f for f in os.listdir(results_dir) if f.startswith('checkpoint_epoch_')]
            if checkpoints:
                epochs = [int(f.split('_')[2].split('.')[0]) for f in checkpoints]
                latest_idx = epochs.index(max(epochs))
                args.checkpoint = os.path.join(results_dir, checkpoints[latest_idx])
            else:
                raise FileNotFoundError(f"No checkpoint found in {results_dir}")
    
    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = UNet(**MODEL_CONFIG)
    epoch, loss = load_checkpoint(args.checkpoint, model, device=device)
    print(f"✅ Loaded checkpoint from epoch {epoch} (loss: {loss:.6f})")
    
    flow_matching = FlowMatching(model, device=device)
    
    # Generate intermediate steps
    print(f"\nGenerating {args.num_samples} sample(s) with {args.num_steps} steps...")
    print("Capturing intermediate steps for visualization...")
    
    image_size = SAMPLE_CONFIG['image_size']
    channels = MODEL_CONFIG['in_channels']
    
    steps = generate_intermediate_steps(
        flow_matching,
        num_samples=args.num_samples,
        num_steps=args.num_steps,
        image_size=image_size,
        channels=channels
    )
    
    print(f"✅ Captured {len(steps)} intermediate steps")
    
    # Create animation
    anim_path = os.path.join(args.output_dir, 'generation_animation.gif')
    create_animation(steps, save_path=anim_path, fps=args.fps)
    
    print("\n" + "=" * 60)
    print("✅ Visualization Complete!")
    print("=" * 60)
    print(f"\nGenerated files:")
    print(f"  - Animation: {anim_path}")
    print(f"\nOpen the GIF file to see the generation process!")


if __name__ == '__main__':
    main()
