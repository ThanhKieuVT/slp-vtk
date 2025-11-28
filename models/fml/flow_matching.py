import torch
import torch.nn as nn
import math

class FlowMatchingScheduler:
    """Flow Matching Scheduler"""
    def __init__(self, sigma_min=0.0, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sample_timesteps(self, batch_size, device):
        return torch.rand(batch_size, device=device)
    
    def get_path(self, x0, x1, t):
        t = t.view(-1, 1, 1)
        return (1 - t) * x0 + t * x1
    
    def get_velocity(self, x0, x1):
        # OT-Flow: velocity is constant, independent of t
        return x1 - x0
    
    def add_noise(self, x1, t):
        x0 = torch.randn_like(x1)
        z_t = self.get_path(x0, x1, t)
        v_gt = self.get_velocity(x0, x1)
        return z_t, v_gt, x0

class FlowMatchingLoss(nn.Module):
    """Normalized Flow Matching Loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, v_pred, v_gt, mask=None):
        loss = (v_pred - v_gt) ** 2
        
        if mask is not None:
            loss = loss * mask.unsqueeze(-1).float()
            num_elements = mask.sum() * v_pred.shape[-1]
            return loss.sum() / num_elements.clamp(min=1)
        
        return loss.mean()

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, x):
        device = x.device
        T = x.shape[1]
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        pos = torch.arange(T, device=device).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return emb[None, :, :]

class TimestepEmbedding(nn.Module):
    """Sinusoidal Timestep Embedding (DDPM-style)"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.SiLU(),
            nn.Linear(dim * 4, dim)
        )
    
    def forward(self, t):
        # Sinusoidal encoding
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat([emb.sin(), emb.cos()], dim=-1)
        return self.mlp(emb)

class CrossAttentionLayer(nn.Module):
    """Pre-LN Cross-Attention with FFN"""
    def __init__(self, d_model, nhead, dim_feedforward=None, dropout=0.1):
        super().__init__()
        if dim_feedforward is None:
            dim_feedforward = d_model * 4
            
        # Cross-Attention
        self.multihead_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        
        # Feed-Forward Network
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(dropout)
        )
        self.norm2 = nn.LayerNorm(d_model)
    
    def forward(self, query, key, value, key_padding_mask=None):
        # Pre-LN Cross-Attention
        q_norm = self.norm1(query)
        attn_output, attn_weights = self.multihead_attn(
            q_norm, key, value, key_padding_mask=key_padding_mask
        )
        query = query + self.dropout1(attn_output)
        
        # Pre-LN FFN
        query = query + self.ffn(self.norm2(query))
        
        return query, attn_weights

class FlowMatchingBlock(nn.Module):
    def __init__(self, data_dim=256, condition_dim=512, hidden_dim=512, 
                 num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        
        # Input Projections
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        
        # Improved Time Embedding
        self.time_embed = TimestepEmbedding(hidden_dim)
        
        # Positional Embedding
        self.pos_emb = SinusoidalPosEmb(hidden_dim)
        
        # Transformer Layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn_block': nn.TransformerEncoderLayer(
                    hidden_dim, num_heads, hidden_dim * 4, dropout, 
                    activation='gelu', batch_first=True, norm_first=True
                ),
                'cross_attn': CrossAttentionLayer(
                    hidden_dim, num_heads, hidden_dim * 4, dropout
                ),
            }))
        
        # Output Projection
        self.output_proj = nn.Linear(hidden_dim, data_dim)
    
    def forward(self, z_t, t, condition, text_attn_mask=None, 
                pose_attn_mask=None, return_attn=False):
        B, T, _ = z_t.shape
        
        # Time embedding (improved with sinusoidal encoding)
        t_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, H]
        
        # Initial embeddings
        x = self.input_proj(z_t) + t_emb + self.pos_emb(z_t)[:, :T, :]
        
        # Transformer blocks
        all_attn = []
        for layer in self.layers:
            # Self-Attention (pose sequence)
            x = layer['self_attn_block'](x, src_key_padding_mask=pose_attn_mask)
            
            # Cross-Attention (pose ← text)
            x, attn = layer['cross_attn'](
                x, condition, condition, key_padding_mask=text_attn_mask
            )
            if return_attn:
                all_attn.append(attn.cpu())
        
        # Output
        out = self.output_proj(x)
        return (out, all_attn) if return_attn else out