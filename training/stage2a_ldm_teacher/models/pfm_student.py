# Tên file: models/pfm_student.py
# === PHIÊN BẢN STUDENT: LÕI MAMBA NHƯNG DÙNG TIME EMBEDDING LIÊN TỤC (0-1) ===

import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba

# ---------------------------------
# CÁC MODULE HELPER
# ---------------------------------

class PositionalEncoding(nn.Module):
    # Giữ nguyên như phiên bản Mamba-Teacher (batch_first)
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(1)].unsqueeze(0)
        return self.dropout(x)

class ContinuousTimeEmbedding(nn.Module):
    """
    Time Embedding cho Flow Matching (thời gian t liên tục 0-1).
    Sử dụng Gaussian Fourier Features.
    """
    def __init__(self, d_model, scale=30.0):
        super().__init__()
        self.d_model = d_model
        half_dim = d_model // 2
        self.W = nn.Parameter(torch.randn(half_dim) * scale, requires_grad=False)

    def forward(self, t):
        """
        Args:
            t: Tensor, shape [B] (giá trị float từ 0 đến 1)
        """
        t = t.float().unsqueeze(1) # [B, 1]
        t_proj = t * self.W.unsqueeze(0) # [B, half_dim]
        emb = torch.cat([torch.sin(t_proj), torch.cos(t_proj)], dim=-1) # [B, d_model]
        return emb

# ---------------------------------
# LÕI KIẾN TRÚC (Giống hệt Teacher)
# ---------------------------------

class MambaCrossAttnLayer(nn.Module):
    # Giữ nguyên 100% như file ldm_denoiser_mamba.py
    def __init__(self, d_model, nhead, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.norm_mamba = nn.LayerNorm(d_model)
        self.mamba = Mamba(d_model=d_model, d_state=d_state, d_conv=d_conv, expand=expand)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=nhead, dropout=0.1, batch_first=True
        )
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4), nn.GELU(), nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, z_t, text_embeddings, text_mask=None, pose_mask=None):
        # Mamba (Self-Attn)
        z_t_res = z_t
        z_t = self.norm_mamba(z_t)
        z_t_mamba = self.mamba(z_t)
        if pose_mask is not None:
             z_t_mamba = z_t_mamba.masked_fill(pose_mask.unsqueeze(-1), 0.0)
        z_t = z_t_res + z_t_mamba
        
        # Cross-Attention (Conditioning)
        z_t_res = z_t
        z_t = self.norm_cross(z_t)
        attn_output, _ = self.cross_attn(
            query=z_t, key=text_embeddings, value=text_embeddings,
            key_padding_mask=text_mask
        )
        z_t = z_t_res + attn_output
        
        # FFN
        z_t_res = z_t
        z_t = self.norm_ffn(z_t)
        z_t = z_t_res + self.ffn(z_t)
        
        return z_t

# ---------------------------------
# MÔ HÌNH STUDENT CHÍNH
# ---------------------------------

class PFM_Student(nn.Module):
    def __init__(
        self,
        latent_dim=256,
        text_embed_dim=768,
        hidden_dim=768,     # Phải = text_embed_dim
        num_layers=6,
        num_heads=12,
        dropout=0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim 
        
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        # THAY ĐỔI: Output là velocity, nên tên là 'velocity_pred_proj'
        self.velocity_pred_proj = nn.Linear(hidden_dim, latent_dim)
        
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # === THAY ĐỔI QUAN TRỌNG: Time Embedding ===
        self.time_embed = nn.Sequential(
            ContinuousTimeEmbedding(hidden_dim), # Dùng module mới
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        self.layers = nn.ModuleList([
            MambaCrossAttnLayer(
                d_model=hidden_dim, nhead=num_heads
            ) for _ in range(num_layers)
        ])
        
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, z_t, t, text_embeddings, text_mask=None, pose_mask=None):
        """
        Args:
            z_t: [B, T, latent_dim] (Latent ở thời gian t)
            t: [B] (Thời gian t liên tục, float 0-1)
            text_embeddings: [B, L, text_embed_dim]
            text_mask: [B, L] (True = ignore)
            pose_mask: [B, T] (True = ignore)
        """
        
        z_t_embed = self.input_proj(z_t)
        
        # Cộng Time embedding
        t_embed = self.time_embed(t) # t là float [B]
        z_t_embed = z_t_embed + t_embed.unsqueeze(1)
        
        z_t_embed = self.pos_encoder(z_t_embed)
        
        output = z_t_embed
        for layer in self.layers:
            output = layer(
                z_t=output,
                text_embeddings=text_embeddings,
                text_mask=text_mask,
                pose_mask=pose_mask
            )
        
        output = self.norm_out(output)
        
        # THAY ĐỔI: Output là dự đoán vận tốc (velocity)
        velocity_pred = self.velocity_pred_proj(output)
        return velocity_pred