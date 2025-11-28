import torch
import torch.nn as nn
import math

class FlowMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, v_pred, v_gt, mask=None):
        loss = (v_pred - v_gt) ** 2
        if mask is not None:
            mask = mask.float().unsqueeze(-1) # [B, T, 1]
            loss = loss * mask
            return loss.sum() / mask.sum().clamp(min=1)
        return loss.mean()

class FlowMatchingScheduler:
    def __init__(self, sigma_min=1e-4):
        self.sigma_min = sigma_min

    def sample_timesteps(self, batch_size, device):
        return torch.rand(batch_size, device=device)

    def add_noise(self, x1, t):
        x0 = torch.randn_like(x1)
        t_b = t.view(-1, 1, 1)
        xt = t_b * x1 + (1 - t_b) * x0
        v_target = x1 - x0 
        return xt, v_target, x0

# ==========================================
# 🔥 NEW: POSITIONAL ENCODING & TRANSFORMER
# ==========================================
class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        # x: [B, T, D] -> Ta chỉ cần T
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        pos = torch.arange(x.shape[1], device=device).float()
        emb = pos[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb[None, :, :] # [1, T, D]

class FlowMatchingBlock(nn.Module):
    """
    Nâng cấp: Dùng Transformer thay vì MLP để học Temporal (thời gian)
    """
    def __init__(self, data_dim, condition_dim, hidden_dim, num_layers=4, num_heads=4, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Input Projection
        self.data_proj = nn.Linear(data_dim, hidden_dim)
        self.cond_proj = nn.Linear(condition_dim, hidden_dim)
        
        # Time Embedding
        self.time_mlp = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Positional Embedding (Bắt buộc cho Transformer)
        self.pos_emb = SinusoidalPosEmb(hidden_dim)
        
        # Transformer Blocks
        # batch_first=True là QUAN TRỌNG
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=num_layers)
        
        # Final Norm & Output
        self.final_norm = nn.LayerNorm(hidden_dim)
        self.out_proj = nn.Linear(hidden_dim, data_dim)

    def forward(self, x, t, condition, text_attn_mask=None, pose_attn_mask=None, return_attn=False):
        # x: [B, T, D_latent]
        B, T, _ = x.shape
        
        # 1. Embeddings
        h = self.data_proj(x)
        # Cộng vị trí (Positional Encoding)
        h = h + self.pos_emb(x)[:, :T, :] 
        
        # 2. Time Embedding (Cộng vào tất cả các frame)
        t_emb = self.time_mlp(t.view(-1, 1)).unsqueeze(1) # [B, 1, H]
        h = h + t_emb
        
        # 3. Condition (Text) Embedding
        # Pooling text condition
        if condition.dim() == 3:
            if text_attn_mask is not None:
                # text_attn_mask: True=Keep, False=Pad (logic BERT cũ) -> cần check lại input
                # Giả sử mask input vào đây là 1=valid, 0=pad
                mask = text_attn_mask.unsqueeze(-1).float()
                cond_vec = (condition * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
            else:
                cond_vec = condition.mean(dim=1)
        else:
            cond_vec = condition
            
        c_emb = self.cond_proj(cond_vec).unsqueeze(1) # [B, 1, H]
        h = h + c_emb
        
        # 4. Transformer Forward
        # pose_attn_mask: True là padding (cần ignore). 
        h = self.transformer(h, src_key_padding_mask=pose_attn_mask)
        
        h = self.final_norm(h)
        output = self.out_proj(h)
        
        if return_attn:
            return output, None
        return output