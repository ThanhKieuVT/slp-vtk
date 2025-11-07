"""
Flow Matching Components: Scheduler, Loss, Block
"""
import torch
import torch.nn as nn
import math


class FlowMatchingScheduler:
    """
    Flow Matching Scheduler: Sample timesteps và add noise
    """
    
    def __init__(self, sigma_min=0.0, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sample_timesteps(self, batch_size, device):
        """Sample timesteps t ~ U(0, 1)"""
        return torch.rand(batch_size, device=device)
    
    def get_path(self, x0, x1, t):
        """
        Linear interpolation path: z_t = (1-t) * x0 + t * x1
        
        Args:
            x0: [B, T, D] - noise (z_0)
            x1: [B, T, D] - data (z_1)
            t: [B] - timesteps
        
        Returns:
            z_t: [B, T, D] - interpolated
        """
        t = t.view(-1, 1, 1)  # [B, 1, 1]
        return (1 - t) * x0 + t * x1
    
    def get_velocity(self, x0, x1, t):
        """
        Velocity field: v_t = x1 - x0 (constant for linear path)
        
        Args:
            x0: [B, T, D] - noise
            x1: [B, T, D] - data
            t: [B] - timesteps
        
        Returns:
            v_t: [B, T, D] - velocity
        """
        return x1 - x0
    
    def add_noise(self, x1, t):
        """
        Add noise to data (for training)
        
        Args:
            x1: [B, T, D] - data
            t: [B] - timesteps
        
        Returns:
            z_t: [B, T, D] - noisy data
            v_gt: [B, T, D] - ground truth velocity
            x0: [B, T, D] - noise
        """
        x0 = torch.randn_like(x1)  # Sample noise
        z_t = self.get_path(x0, x1, t)
        v_gt = self.get_velocity(x0, x1, t)
        return z_t, v_gt, x0


class FlowMatchingLoss(nn.Module):
    """Flow Matching Loss: MSE between predicted and ground truth velocity"""
    
    def __init__(self):
        super().__init__()
    
    def forward(self, v_pred, v_gt, mask=None):
        """
        Args:
            v_pred: [B, T, D] - predicted velocity
            v_gt: [B, T, D] - ground truth velocity
            mask: [B, T] - valid mask (True = valid)
        
        Returns:
            loss: scalar
        """
        loss = (v_pred - v_gt) ** 2
        
        if mask is not None:
            loss = loss * mask.unsqueeze(-1).float()
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        
        return loss


class FlowMatchingBlock(nn.Module):
    """
    Flow Matching Block: Predict velocity v(z_t, t, condition)
    """
    
    def __init__(
        self,
        data_dim=256,
        condition_dim=512,
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.data_dim = data_dim
        self.condition_dim = condition_dim
        
        # Input projection
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition projection
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z_t, t, condition, mask=None):
        """
        Args:
            z_t: [B, T, data_dim] - noisy latent
            t: [B] - timesteps
            condition: [B, L, condition_dim] - text features
            mask: [B, T] - valid mask
        
        Returns:
            v_pred: [B, T, data_dim] - predicted velocity
        """
        B, T, D = z_t.shape
        
        # Project input
        x = self.input_proj(z_t)  # [B, T, hidden_dim]
        
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))  # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, hidden_dim]
        x = x + t_emb
        
        # Condition: average pooling over sequence length
        condition_pooled = condition.mean(dim=1)  # [B, condition_dim]
        condition_emb = self.condition_proj(condition_pooled)  # [B, hidden_dim]
        condition_emb = condition_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, hidden_dim]
        x = x + condition_emb
        
        # Attention mask
        if mask is not None:
            attn_mask = ~mask  # [B, T] - True = ignore
        else:
            attn_mask = None
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Output
        v_pred = self.output_proj(x)  # [B, T, data_dim]
        
        return v_pred

