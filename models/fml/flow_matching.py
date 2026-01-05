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
        # OT-Flow: velocity is constant
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
            # Mask: True=Valid, False=Padding
            loss = loss * mask.unsqueeze(-1).float()
            num_elements = mask.sum() * v_pred.shape[-1]
            return loss.sum() / num_elements.clamp(min=1)
        else:
            return loss.mean()

class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb

class CrossAttentionLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.multihead_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        self.norm_ffn = nn.LayerNorm(d_model)

    def forward(self, tgt, memory, memory_key_padding_mask=None, key_padding_mask=None):
        # Cross Attention
        attn_out, attn_weights = self.multihead_attn(
            tgt, memory, memory, 
            key_padding_mask=key_padding_mask 
        )
        tgt = self.norm(tgt + self.dropout(attn_out))
        
        # FFN
        ffn_out = self.ffn(tgt)
        tgt = self.norm_ffn(tgt + self.dropout(ffn_out))
        return tgt, attn_weights

class FlowMatchingBlock(nn.Module):
    def __init__(self, data_dim, hidden_dim, condition_dim, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        
        # Embeddings
        self.time_embed = nn.Sequential(
            SinusoidalPosEmb(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.input_proj = nn.Linear(data_dim, hidden_dim)
        self.pos_emb = SinusoidalPosEmb(hidden_dim) # Simple fixed pos emb
        
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
        
        # Time embedding
        t_emb = self.time_embed(t).unsqueeze(1)  # [B, 1, H]
        
        # Initial embeddings
        pos_emb = self.pos_emb(torch.arange(T, device=z_t.device).float()) # [T, H]
        x = self.input_proj(z_t) + t_emb + pos_emb.unsqueeze(0)
        
        # ✅ FIX QUAN TRỌNG: Đảo Mask cho PyTorch Transformer
        # PyTorch quy ước: True = Padding (Ignore), False = Keep
        # Dataset quy ước: True = Valid, False = Padding
        # => Cần lấy phủ định (~) của mask
        
        torch_pose_mask = None
        if pose_attn_mask is not None:
            torch_pose_mask = ~pose_attn_mask.bool() # True ở vị trí Padding
            
        torch_text_mask = None
        if text_attn_mask is not None:
            torch_text_mask = ~text_attn_mask.bool() # True ở vị trí Padding

        # Transformer blocks
        for layer in self.layers:
            # Self-Attention
            x = layer['self_attn_block'](x, src_key_padding_mask=torch_pose_mask)
            
            # Cross-Attention
            x, attn = layer['cross_attn'](
                x, condition, 
                key_padding_mask=torch_text_mask 
            )
            
        return self.output_proj(x)