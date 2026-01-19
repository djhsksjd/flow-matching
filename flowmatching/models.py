"""
Neural Network Models for Flow Matching

This module contains the model architectures including:
- TimeEmbedding: Time step embedding
- ResidualBlock: Residual block with time conditioning
- UNet: U-Net architecture for velocity field prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class TimeEmbedding(nn.Module):
    """Time embedding using sinusoidal encoding."""
    
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, time):
        device = time.device
        time = time.float()
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device, dtype=torch.float32) * -emb)
        emb = time[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        if self.dim % 2 == 1:  # Zero pad
            emb = F.pad(emb, (0, 1), mode='constant')
        return emb


class ResidualBlock(nn.Module):
    """Residual block with time conditioning."""
    
    def __init__(self, in_channels, out_channels, time_emb_dim, dropout=0.1):
        super().__init__()
        self.time_mlp = nn.Linear(time_emb_dim, out_channels)
        self.block1 = nn.Sequential(
            nn.GroupNorm(8, in_channels),
            nn.SiLU(),
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
        )
        self.block2 = nn.Sequential(
            nn.GroupNorm(8, out_channels),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
        )
        self.res_conv = nn.Conv2d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x, time_emb):
        h = self.block1(x)
        time_emb = self.time_mlp(time_emb)
        h = h + time_emb[:, :, None, None]
        h = self.block2(h)
        return h + self.res_conv(x)


class UNet(nn.Module):
    """U-Net architecture for flow matching velocity field prediction."""
    
    def __init__(self, in_channels=1, time_emb_dim=128, base_channels=64):
        super().__init__()
        self.time_emb_dim = time_emb_dim
        
        # Time embedding
        self.time_mlp = nn.Sequential(
            TimeEmbedding(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # Encoder (downsampling)
        self.enc1 = nn.Conv2d(in_channels, base_channels, 3, padding=1)
        self.enc2 = ResidualBlock(base_channels, base_channels, time_emb_dim)
        self.down1 = nn.Conv2d(base_channels, base_channels, 3, stride=2, padding=1)  # 28->14
        
        self.enc3 = ResidualBlock(base_channels, base_channels * 2, time_emb_dim)
        self.down2 = nn.Conv2d(base_channels * 2, base_channels * 2, 3, stride=2, padding=1)  # 14->7
        
        self.enc4 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        
        # Bottleneck
        self.bottleneck = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        
        # Decoder (upsampling)
        self.dec1 = ResidualBlock(base_channels * 2, base_channels * 2, time_emb_dim)
        self.up1 = nn.ConvTranspose2d(base_channels * 2, base_channels * 2, 4, stride=2, padding=1)  # 7->14
        
        self.dec2 = ResidualBlock(base_channels * 4, base_channels, time_emb_dim)  # 4x because of skip connection
        self.up2 = nn.ConvTranspose2d(base_channels, base_channels, 4, stride=2, padding=1)  # 14->28
        
        self.dec3 = ResidualBlock(base_channels * 2, base_channels, time_emb_dim)  # 2x because of skip connection
        
        # Output
        self.output = nn.Sequential(
            nn.GroupNorm(8, base_channels),
            nn.SiLU(),
            nn.Conv2d(base_channels, in_channels, 3, padding=1),
        )
    
    def forward(self, x, t):
        # Time embedding
        t_emb = self.time_mlp(t)
        
        # Encoder
        e1 = self.enc1(x)
        e2 = self.enc2(e1, t_emb)
        e3 = self.down1(e2)
        
        e4 = self.enc3(e3, t_emb)
        e5 = self.down2(e4)
        
        e6 = self.enc4(e5, t_emb)
        
        # Bottleneck
        b = self.bottleneck(e6, t_emb)
        
        # Decoder with skip connections
        d1 = self.dec1(b, t_emb)
        d2 = self.up1(d1)
        d2 = torch.cat([d2, e4], dim=1)  # Skip connection
        
        d3 = self.dec2(d2, t_emb)
        d4 = self.up2(d3)
        d4 = torch.cat([d4, e2], dim=1)  # Skip connection
        
        d5 = self.dec3(d4, t_emb)
        
        # Output
        return self.output(d5)
