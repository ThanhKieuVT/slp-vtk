"""
Flow Matching Components: Scheduler, Loss, Enhanced Block with Cross-Attention
"""
import torch
import torch.nn as nn
import math


class FlowMatchingScheduler:
    """Flow Matching Scheduler: Sample timesteps và add noise (Giữ nguyên)"""
    
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
    """Flow Matching Loss: MSE between predicted and ground truth velocity (Giữ nguyên)"""
    
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
    """Lớp Cross-Attention chuyên biệt cho Pose (Query) -> Text (Key/Value)"""
    def __init__(self, d_model, nhead, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=dropout,
            batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query, key, value, key_padding_mask=None):
        # query: Latent Pose [B, T_pose, D]
        # key/value: Text Features [B, L_text, D]
        
        # Cross-Attention
        attn_output, attn_weights = self.multihead_attn(
            query=query, 
            key=key, 
            value=value, 
            key_padding_mask=key_padding_mask # Mask áp dụng lên K/V (text padding)
        )
        
        # Residual connection + Norm
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights # Trả về weights để debug/log


class EnhancedFlowMatchingBlock(nn.Module):
    """
    SOTA Flow Matching Block: Predict velocity v(z_t, t, condition) với CROSS-ATTENTION
    """
    
    def __init__(
        self,
        data_dim=256,
        condition_dim=512, # Text Feature Dim (sau khi proj)
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.data_dim = data_dim
        self.condition_dim = condition_dim
        
        # Input projection (z_t)
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            # 1. Self-Attention (Pose -> Pose)
            self_attn_layer = nn.TransformerEncoderLayer(
                d_model=hidden_dim, nhead=num_heads, dim_feedforward=hidden_dim * 4,
                dropout=dropout, activation='gelu', batch_first=True
            )
            
            # 2. Cross-Attention (Pose -> Text)
            cross_attn_layer = CrossAttentionLayer(hidden_dim, num_heads, dropout)
            
            # 3. Feed-Forward
            ffn = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.GELU(),
                nn.Linear(hidden_dim * 4, hidden_dim)
            )
            
            self.layers.append(nn.ModuleDict({
                'self_attn': self_attn_layer,
                'cross_attn': cross_attn_layer,
                'ffn': ffn
            }))
            
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, data_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z_t, t, condition, text_attn_mask=None, pose_attn_mask=None, return_attn=False):
        """
        Args:
            z_t: [B, T, data_dim] - noisy latent
            t: [B] - timesteps
            condition: [B, L, condition_dim] - text features (TOKEN LEVEL)
            text_attn_mask: [B, L] - Mask cho text tokens (True = ignore)
            pose_attn_mask: [B, T] - Mask cho pose frames (True = ignore)
        """
        B, T, D = z_t.shape
        
        # Project input
        x = self.input_proj(z_t)  # [B, T, hidden_dim]
        
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))  # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, hidden_dim]
        x = x + t_emb
        
        # Text projection (giả sử đã được project trong LatentFlowMatcher)
        text_features = condition 
        
        all_attn_weights = []
        
        for layer_dict in self.layers:
            # 1. Self-Attention (Pose)
            x = layer_dict['self_attn'](x, src_key_padding_mask=pose_attn_mask)
            
            # 2. Cross-Attention (Pose -> Text)
            x_cross, attn_weights = layer_dict['cross_attn'](
                query=x, key=text_features, value=text_features, 
                key_padding_mask=text_attn_mask
            )
            x = x_cross # Output của cross-attention đã bao gồm residual
            
            if return_attn:
                all_attn_weights.append(attn_weights.cpu()) # [B, H, T, L]

            # 3. Feed-Forward (Residual + Norm đã có trong layer)
            x = layer_dict['ffn'](x) # Thường nằm trong TransformerEncoderLayer
            
        # Output
        v_pred = self.output_proj(x)
        
        if return_attn:
            return v_pred, all_attn_weights
        
        return v_pred