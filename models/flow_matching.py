import torch
import torch.nn as nn
import math

# ============================================
# 1. FLOW MATCHING SCHEDULER
# ============================================
class FlowMatchingScheduler:
    """Flow Matching Scheduler - Straight-line interpolation"""
    def __init__(self, sigma_min=0.0, sigma_max=1.0):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max
    
    def sample_timesteps(self, batch_size, device):
        """Sample random timesteps t ~ U(0,1)"""
        return torch.rand(batch_size, device=device)
    
    def get_path(self, x0, x1, t):
        """Interpolation path: z_t = (1-t)*x0 + t*x1"""
        t = t.view(-1, 1, 1)
        return (1 - t) * x0 + t * x1
    
    def get_velocity(self, x0, x1, t):
        """Target velocity: v = x1 - x0 (constant for straight line)"""
        return x1 - x0
    
    def add_noise(self, x1, t):
        """
        Add noise to clean data x1 at timestep t
        Returns: z_t, v_gt, x0
        """
        x0 = torch.randn_like(x1)  # Sample from standard Gaussian
        z_t = self.get_path(x0, x1, t)
        v_gt = self.get_velocity(x0, x1, t)
        return z_t, v_gt, x0


# ============================================
# 2. FLOW MATCHING LOSS (FIXED SCALING)
# ============================================
class FlowMatchingLoss(nn.Module):
    """
    Flow Matching Loss - FIXED SCALING
    
    FIX: Chia cho số SAMPLES thay vì (samples × dimensions)
    để tránh loss quá nhỏ làm yếu gradient
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, v_pred, v_gt, mask=None):
        """
        Args:
            v_pred: [B, T, D] - Predicted velocity
            v_gt: [B, T, D] - Ground truth velocity
            mask: [B, T] - Valid frame mask (True = valid)
        """
        # MSE Loss
        loss = (v_pred - v_gt) ** 2  # [B, T, D]
        
        if mask is not None:
            # 🔥 FIX: Sum over D first, then average over valid frames
            # Cách cũ: loss.sum() / (mask.sum() * D) → Quá nhỏ
            # Cách mới: loss.sum(D) / mask.sum() → Vừa phải
            
            loss_per_frame = loss.sum(dim=-1)  # [B, T] - Sum over D
            masked_loss = loss_per_frame * mask.float()  # Apply mask
            
            total_valid_frames = mask.sum()
            loss = masked_loss.sum() / total_valid_frames.clamp(min=1)
        else:
            loss = loss.mean()
            
        return loss


# ============================================
# 3. CROSS-ATTENTION LAYER
# ============================================
class CrossAttentionLayer(nn.Module):
    """Cross-Attention: Pose (Query) -> Text (Key/Value)"""
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
        """
        Args:
            query: [B, T_pose, D] - Pose features
            key: [B, T_text, D] - Text features
            value: [B, T_text, D] - Text features
            key_padding_mask: [B, T_text] - True = ignore
        
        Returns:
            output: [B, T_pose, D]
            attn_weights: [B, num_heads, T_pose, T_text]
        """
        attn_output, attn_weights = self.multihead_attn(
            query=query, 
            key=key, 
            value=value, 
            key_padding_mask=key_padding_mask
        )
        
        # Residual + LayerNorm
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights


# ============================================
# 4. FLOW MATCHING BLOCK (DENOISER)
# ============================================
class FlowMatchingBlock(nn.Module):
    """
    SOTA Flow Matching Block
    - Self-attention for temporal modeling
    - Cross-attention for text conditioning
    - Time embedding injection
    """
    def __init__(
        self, 
        data_dim=256,           # Latent dimension
        condition_dim=512,      # Text feature dimension
        hidden_dim=512,         # Internal hidden dimension
        num_layers=6,           # Number of transformer layers
        num_heads=8,            # Number of attention heads
        dropout=0.1
    ):
        super().__init__()
        
        # Input projection: latent -> hidden
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        
        # Time embedding (sinusoidal + MLP)
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer layers
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(nn.ModuleDict({
                # Self-attention on pose sequence
                'self_attn_block': nn.TransformerEncoderLayer(
                    d_model=hidden_dim,
                    nhead=num_heads,
                    dim_feedforward=hidden_dim * 4,
                    dropout=dropout,
                    activation='gelu',
                    batch_first=True
                ),
                # Cross-attention to text
                'cross_attn': CrossAttentionLayer(
                    d_model=hidden_dim, 
                    nhead=num_heads, 
                    dropout=dropout
                ),
            }))
        
        # Output projection: hidden -> velocity (data_dim)
        self.output_proj = nn.Linear(hidden_dim, data_dim)
    
    def forward(
        self, 
        z_t,                    # [B, T, data_dim] - Noisy latent
        t,                      # [B] - Timesteps
        condition,              # [B, T_text, condition_dim] - Text features
        text_attn_mask=None,    # [B, T_text] - Text padding mask
        pose_attn_mask=None,    # [B, T] - Pose padding mask
        return_attn=False
    ):
        """
        Args:
            z_t: Noisy latent at timestep t
            t: Timesteps (0 to 1)
            condition: Text condition features
            text_attn_mask: Mask for text (True = ignore)
            pose_attn_mask: Mask for pose (True = ignore)
            return_attn: Return attention weights
        
        Returns:
            v_pred: Predicted velocity [B, T, data_dim]
            (optional) attn_weights: List of attention maps
        """
        B, T, D = z_t.shape
        
        # Project input
        x = self.input_proj(z_t)  # [B, T, hidden_dim]
        
        # Add time embedding (broadcast to all frames)
        t_emb = self.time_embed(t.unsqueeze(-1))  # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, hidden_dim]
        x = x + t_emb
        
        # Process through transformer layers
        all_attn_weights = []
        for layer_dict in self.layers:
            # Self-attention (temporal modeling)
            x = layer_dict['self_attn_block'](
                x, 
                src_key_padding_mask=pose_attn_mask
            )
            
            # Cross-attention (text conditioning)
            x, attn_weights = layer_dict['cross_attn'](
                query=x,
                key=condition,
                value=condition,
                key_padding_mask=text_attn_mask
            )
            
            if return_attn:
                all_attn_weights.append(attn_weights.cpu())
        
        # Predict velocity
        v_pred = self.output_proj(x)  # [B, T, data_dim]
        
        if return_attn:
            return v_pred, all_attn_weights
        return v_pred