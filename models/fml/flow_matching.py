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
    
    def get_velocity(self, x0, x1, t):
        return x1 - x0
    
    def add_noise(self, x1, t):
        x0 = torch.randn_like(x1)
        z_t = self.get_path(x0, x1, t)
        v_gt = self.get_velocity(x0, x1, t)
        return z_t, v_gt, x0

class FlowMatchingLoss(nn.Module):
    """Flow Matching Loss"""
    def __init__(self):
        super().__init__()
    
    def forward(self, v_pred, v_gt, mask=None):
        loss = (v_pred - v_gt) ** 2
        if mask is not None:
            loss = loss * mask.unsqueeze(-1).float()
            loss = loss.sum() / mask.sum().clamp(min=1)
        else:
            loss = loss.mean()
        return loss

class CrossAttentionLayer(nn.Module):
    """Cross-Attention: Pose (Query) -> Text (Key/Value)"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None):
        attn_output, attn_weights = self.multihead_attn(
            query=query, key=key, value=value, key_padding_mask=key_padding_mask
        )
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights

class FlowMatchingBlock(nn.Module):
    """SOTA Flow Matching Block"""
    def __init__(self, data_dim=256, condition_dim=512, hidden_dim=512, num_layers=6, num_heads=8, dropout=0.1):
        super().__init__()
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                'self_attn_block': nn.TransformerEncoderLayer(
                    d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                    dropout=dropout, activation='gelu', batch_first=True
                ),
                'cross_attn': CrossAttentionLayer(hidden_dim, num_heads, dropout),
            }))
            
        self.output_proj = nn.Linear(hidden_dim, data_dim)
    
    def forward(self, z_t, t, condition, text_attn_mask=None, pose_attn_mask=None, return_attn=False):
        B, T, D = z_t.shape
        x = self.input_proj(z_t)
        t_emb = self.time_embed(t.unsqueeze(-1)).unsqueeze(1).expand(-1, T, -1)
        x = x + t_emb
        
        all_attn_weights = []
        for layer_dict in self.layers:
            x = layer_dict['self_attn_block'](x, src_key_padding_mask=pose_attn_mask)
            x, attn_weights = layer_dict['cross_attn'](
                query=x, key=condition, value=condition, key_padding_mask=text_attn_mask
            )
            if return_attn: all_attn_weights.append(attn_weights.cpu())

        v_pred = self.output_proj(x)
        if return_attn: return v_pred, all_attn_weights
        return v_pred