# models/flow_matching.py
"""
Core Flow Matching utilities
Implements Optimal Transport Conditional Flow Matching
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FlowMatchingScheduler:
    """
    Manages timesteps and noise schedules for flow matching
    """
    def __init__(self, num_timesteps=1000, sigma_min=0.001):
        self.num_timesteps = num_timesteps
        self.sigma_min = sigma_min
        
    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps for training"""
        # Uniform sampling in [0, 1]
        t = torch.rand(batch_size, device=device)
        return t
    
    def add_noise(self, x_0, t):
        """
        Add noise to data according to flow schedule
        x_t = (1-t)·x_0 + t·x_1
        where x_1 ~ N(0, I) is target noise
        """
        batch_size = x_0.shape[0]
        
        # Sample target noise
        x_1 = torch.randn_like(x_0)
        
        # Interpolate: x_t = (1-t)·x_0 + t·x_1
        t_expanded = t.view(-1, 1, 1)  # [B, 1, 1]
        x_t = (1 - t_expanded) * x_0 + t_expanded * x_1
        
        # Ground truth velocity: v = x_1 - x_0
        velocity_gt = x_1 - x_0
        
        return x_t, velocity_gt, x_1
    
    def generate_timesteps(self, num_steps):
        """
        Generate inference timesteps
        Using linear schedule from 0 to 1
        """
        return torch.linspace(0, 1, num_steps)

class SinusoidalPosEmb(nn.Module):
    """
    Sinusoidal positional embeddings for timestep encoding
    Same as in Transformers, but for continuous time t ∈ [0,1]
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        
    def forward(self, t):
        """
        t: [B] timesteps in [0, 1]
        Returns: [B, dim] embeddings
        """
        device = t.device
        half_dim = self.dim // 2
        
        # Frequencies
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        
        # Compute embeddings
        emb = t[:, None] * emb[None, :]  # [B, half_dim]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)  # [B, dim]
        
        return emb

class FlowMatchingBlock(nn.Module):
    """
    Basic building block for flow matching
    Predicts velocity field v(x_t, t, condition)
    
    Architecture:
        Input: noisy_data [B, T, D] + timestep [B] + condition [B, L, C]
        Output: velocity [B, T, D]
    """
    def __init__(
        self,
        data_dim,           # Dimension of data (e.g., 66 for body poses)
        condition_dim,      # Dimension of conditioning (e.g., 768 from BERT)
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        self.data_dim = data_dim
        self.condition_dim = condition_dim
        self.hidden_dim = hidden_dim
        
        # Timestep embedding
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Data projection
        self.data_proj = nn.Linear(data_dim, hidden_dim)
        
        # Condition projection
        self.condition_proj = nn.Linear(condition_dim, hidden_dim)
        
        # Transformer blocks for temporal modeling
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )
        
        # Cross-attention to condition on text
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        self.norm_cross = nn.LayerNorm(hidden_dim)
        
        # Output projection to velocity
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, data_dim)
        )
        
    def forward(self, x_t, t, condition, mask=None):
        """
        x_t: [B, T, data_dim] - noisy data at time t
        t: [B] - timestep
        condition: [B, L, condition_dim] - text features
        mask: [B, T] - padding mask (optional)
        
        Returns: velocity [B, T, data_dim]
        """
        B, T, D = x_t.shape
        
        # 1. Embed timestep
        t_emb = self.time_embed(t)  # [B, hidden_dim]
        
        # 2. Project data
        h = self.data_proj(x_t)  # [B, T, hidden_dim]
        
        # 3. Add timestep information (broadcast)
        h = h + t_emb[:, None, :]  # [B, T, hidden_dim]
        
        # 4. Self-attention (temporal modeling)
        h = self.transformer(h, src_key_padding_mask=mask)
        
        # 5. Cross-attention to text condition
        h_cross, _ = self.cross_attn(
            query=h,
            key=condition,
            value=condition
        )
        h = self.norm_cross(h + h_cross)  # Residual
        
        # 6. Predict velocity
        velocity = self.output_proj(h)  # [B, T, data_dim]
        
        return velocity

class FlowMatchingLoss(nn.Module):
    """
    Loss function for flow matching training
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, pred_velocity, gt_velocity, mask=None):
        """
        pred_velocity: [B, T, D] - predicted by model
        gt_velocity: [B, T, D] - ground truth (x_1 - x_0)
        mask: [B, T] - padding mask
        
        Returns: scalar loss
        """
        # MSE loss
        loss = F.mse_loss(pred_velocity, gt_velocity, reduction='none')
        
        # Apply mask if provided
        if mask is not None:
            # mask: True for valid, False for padding
            loss = loss * mask.unsqueeze(-1)
            loss = loss.sum() / mask.sum()
        else:
            loss = loss.mean()
        
        return loss

def euler_integrate(velocity_fn, x_0, t_span, num_steps):
    """
    Euler integration for ODE: dx/dt = v(x, t)
    
    velocity_fn: function that computes velocity given (x, t, condition)
    x_0: [B, T, D] - initial condition (noise)
    t_span: (t_start, t_end) - integration interval
    num_steps: number of integration steps
    
    Returns: x at t_end
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / num_steps
    
    x = x_0
    t = t_start
    
    for _ in range(num_steps):
        # Compute velocity
        v = velocity_fn(x, t)
        
        # Euler step: x_{n+1} = x_n + dt * v_n
        x = x + dt * v
        t = t + dt
    
    return x

def rk4_integrate(velocity_fn, x_0, t_span, num_steps):
    """
    Runge-Kutta 4th order integration (more accurate than Euler)
    
    RK4 formula:
        k1 = f(x_n, t_n)
        k2 = f(x_n + dt/2 * k1, t_n + dt/2)
        k3 = f(x_n + dt/2 * k2, t_n + dt/2)
        k4 = f(x_n + dt * k3, t_n + dt)
        x_{n+1} = x_n + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
    """
    t_start, t_end = t_span
    dt = (t_end - t_start) / num_steps
    
    x = x_0
    t = t_start
    
    for _ in range(num_steps):
        k1 = velocity_fn(x, t)
        k2 = velocity_fn(x + dt/2 * k1, t + dt/2)
        k3 = velocity_fn(x + dt/2 * k2, t + dt/2)
        k4 = velocity_fn(x + dt * k3, t + dt)
        
        x = x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        t = t + dt
    
    return x