"""
Training Utilities

This module provides functions for training Flow Matching models.
"""

import torch
import torch.optim as optim
from tqdm import tqdm
import os
from .flow_matching import FlowMatching
from .utils import visualize_samples


def train_flow_matching(
    model,
    train_loader,
    num_epochs=50,
    lr=1e-4,
    device='cuda' if torch.cuda.is_available() else 'cpu',
    save_dir='results',
    save_every=10,
    sample_every=5,
    num_sample_steps=50
):
    """
    Train flow matching model.
    
    Args:
        model: UNet model to train
        train_loader: DataLoader for training data
        num_epochs: Number of training epochs
        lr: Learning rate
        device: Device to run training on
        save_dir: Directory to save checkpoints and samples
        save_every: Save checkpoint every N epochs
        sample_every: Generate samples every N epochs
        num_sample_steps: Number of steps for sampling during training
    
    Returns:
        flow_matching: Trained FlowMatching instance
    """
    os.makedirs(save_dir, exist_ok=True)
    
    flow_matching = FlowMatching(model, device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Training loop
    for epoch in range(num_epochs):
        epoch_loss = 0.0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        for batch_idx, (x, _) in enumerate(pbar):
            x = x.to(device)
            
            loss = flow_matching.train_step(x, optimizer)
            epoch_loss += loss
            num_batches += 1
            
            pbar.set_postfix({'loss': loss, 'avg_loss': epoch_loss / num_batches})
        
        avg_loss = epoch_loss / num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % save_every == 0:
            checkpoint_path = os.path.join(save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, checkpoint_path)
            print(f"Saved checkpoint to {checkpoint_path}")
        
        # Generate and save samples (grid only)
        if (epoch + 1) % sample_every == 0:
            # Use config image size
            from .config import SAMPLE_CONFIG
            img_size = SAMPLE_CONFIG.get('image_size', (64, 64))
            samples = flow_matching.sample(64, num_steps=num_sample_steps, 
                                         image_size=img_size, channels=3)
            save_path = os.path.join(save_dir, f"samples_epoch_{epoch+1}.png")
            visualize_samples(samples, save_path=save_path, nrow=8, 
                            title=f"Generated Face Samples - Epoch {epoch+1}")
    
    return flow_matching
