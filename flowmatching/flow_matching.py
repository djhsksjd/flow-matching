"""
Flow Matching Algorithm Implementation

This module implements the Flow Matching algorithm for generative modeling.
Flow Matching learns a velocity field that defines an ODE to transform noise into data.

Based on: "Flow Matching for Generative Modeling" (Lipman et al., 2023)
"""

import torch
import torch.nn.functional as F
import numpy as np
from scipy import integrate


class FlowMatching:
    """Flow Matching for generative modeling."""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        """
        Initialize Flow Matching.
        
        Args:
            model: Neural network model that predicts velocity field v_theta(x, t)
            device: Device to run computation on ('cuda' or 'cpu')
        """
        self.model = model.to(device)
        self.device = device
    
    def sample_t(self, batch_size):
        """Sample time steps uniformly from [0, 1]."""
        return torch.rand(batch_size, device=self.device)
    
    def sample_noise_and_data(self, x1):
        """
        Sample x0 (noise) and x1 (data).
        
        Args:
            x1: Data samples (shape: [batch, channels, height, width])
        
        Returns:
            x0: Noise samples (same shape as x1)
            x1: Data samples
        """
        batch_size = x1.shape[0]
        x0 = torch.randn_like(x1)
        return x0, x1
    
    def compute_flow_matching_loss(self, x0, x1, t):
        """
        Compute flow matching loss.
        
        The flow matching objective:
        - x_t = (1-t) * x0 + t * x1  (linear interpolation)
        - u_t = x1 - x0  (target velocity field)
        - Loss: ||v_theta(x_t, t) - u_t||^2
        
        Args:
            x0: Noise samples
            x1: Data samples
            t: Time steps (shape: [batch])
        
        Returns:
            loss: MSE loss between predicted and target velocity field
        """
        # Linear interpolation
        x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * x1
        
        # Target velocity field
        u_t = x1 - x0
        
        # Predicted velocity field
        v_t = self.model(x_t, t)
        
        # Loss
        loss = F.mse_loss(v_t, u_t)
        return loss
    
    def train_step(self, x1, optimizer):
        """
        Perform one training step.
        
        Args:
            x1: Data samples (shape: [batch, channels, height, width])
            optimizer: Optimizer for model parameters
        
        Returns:
            loss: Training loss value
        """
        self.model.train()
        optimizer.zero_grad()
        
        # Sample time steps
        t = self.sample_t(x1.shape[0])
        
        # Sample noise and data
        x0, x1 = self.sample_noise_and_data(x1)
        
        # Compute loss
        loss = self.compute_flow_matching_loss(x0, x1, t)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        return loss.item()
    
    @torch.no_grad()
    def sample(self, num_samples, num_steps=100, image_size=(64, 64), channels=3):
        """
        Sample images using Euler method for ODE integration.
        
        The ODE is: dx/dt = v_theta(x, t)
        We solve it from t=0 (noise) to t=1 (data).
        
        Args:
            num_samples: Number of samples to generate
            num_steps: Number of integration steps
            image_size: Tuple of (height, width) for generated images
            channels: Number of channels (3 for RGB, 1 for grayscale)
        
        Returns:
            x: Generated samples (shape: [num_samples, channels, height, width])
        """
        self.model.eval()
        
        # Start from noise
        x = torch.randn(num_samples, channels, image_size[0], image_size[1], device=self.device)
        
        # Time steps
        dt = 1.0 / num_steps
        
        # Euler integration
        for i in range(num_steps):
            t = torch.full((num_samples,), i * dt, device=self.device)
            v = self.model(x, t)
            x = x + dt * v
        
        # Clip to [0, 1] range
        x = torch.clamp(x, 0, 1)
        
        return x
    
    @torch.no_grad()
    def sample_ode(self, num_samples, num_steps=100, image_size=(64, 64), channels=3):
        """
        Sample using ODE solver (more accurate but slower).
        
        Args:
            num_samples: Number of samples to generate
            num_steps: Number of evaluation points for ODE solver
            image_size: Tuple of (height, width) for generated images
            channels: Number of channels (3 for RGB, 1 for grayscale)
        
        Returns:
            x: Generated samples (shape: [num_samples, channels, height, width])
        """
        self.model.eval()
        
        # Start from noise
        x0 = torch.randn(num_samples, channels, image_size[0], image_size[1], device=self.device)
        x0_flat = x0.view(num_samples, -1).cpu().numpy()
        
        # Define ODE function
        def ode_func(t, x_flat):
            x = torch.from_numpy(x_flat).float().view(num_samples, channels, image_size[0], image_size[1]).to(self.device)
            t_tensor = torch.full((num_samples,), t, device=self.device)
            v = self.model(x, t_tensor)
            return v.view(num_samples, -1).cpu().numpy()
        
        # Solve ODE from t=0 to t=1
        t_eval = np.linspace(0, 1, num_steps + 1)
        sol = integrate.solve_ivp(ode_func, [0, 1], x0_flat.flatten(), t_eval=t_eval, method='RK45')
        
        # Get final state
        x_final = sol.y[:, -1].reshape(num_samples, channels, image_size[0], image_size[1])
        x_final = torch.from_numpy(x_final).float().to(self.device)
        x_final = torch.clamp(x_final, 0, 1)
        
        return x_final
