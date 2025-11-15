# Tên file: models/ldm_denoiser_mamba.py
# === PHIÊN BẢN NÂNG CẤP: LÕI LAI MAMBA + TRANSFORMER CROSS-ATTENTION ===

import torch
import torch.nn as nn
import math
from mamba_ssm import Mamba

# ---------------------------------
# CÁC MODULE CŨ (Hơi điều chỉnh cho batch_first)
# ---------------------------------

class PositionalEncoding(nn.Module):
    """
    Phiên bản Positional Encoding hỗ trợ batch_first=True.
    Mong đợi input x có shape [B, T, D]
    """
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model) # Bỏ 1 dimension ở giữa
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe) # pe shape [max_len, d_model]

    def forward(self, x):
        """
        Args:
            x: Tensor, shape [B, T, D]
        """
        # Thêm PE vào [B, T, D]
        x = x + self.pe[:x.size(1)].unsqueeze(0) # [1, T, D]
        return self.dropout(x)

class TimeEmbed(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq)

    def forward(self, t):
        t = t.float()
        freqs = torch.einsum("b, d -> bd", t, self.inv_freq)
        emb = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        return emb

# ---------------------------------
# KHỐI KIẾN TRÚC MỚI (LÕI LAI)
# ---------------------------------

class MambaCrossAttnLayer(nn.Module):
    """
    Một khối (Block) lai kết hợp:
    1. Mamba: Làm Self-Attention trên chuỗi pose (z_t)
    2. Cross-Attention: Làm Conditioning với text_embeddings
    """
    def __init__(self, d_model, nhead, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        
        # 1. Khối Mamba (cho Self-Attention)
        self.norm_mamba = nn.LayerNorm(d_model)
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        # 2. Khối Cross-Attention (cho Text Conditioning)
        self.norm_cross = nn.LayerNorm(d_model)
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=nhead,
            dropout=0.1,
            batch_first=True # Rất quan trọng!
        )
        
        # 3. Khối FFN (như Transformer)
        self.norm_ffn = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 4, d_model)
        )

    def forward(self, z_t, text_embeddings, text_mask=None, pose_mask=None):
        """
        Args:
            z_t: [B, T_pose, D] (Pose latents)
            text_embeddings: [B, L_text, D] (Text memory)
            text_mask: [B, L_text] (True = ignore)
            pose_mask: [B, T_pose] (True = ignore)
        """
        
        # 1. Mamba (Self-Attention)
        z_t_res = z_t
        z_t = self.norm_mamba(z_t)
        z_t_mamba = self.mamba(z_t)
        
        # Xử lý padding cho Mamba (Mamba không tự làm)
        if pose_mask is not None:
             z_t_mamba = z_t_mamba.masked_fill(pose_mask.unsqueeze(-1), 0.0)
        
        z_t = z_t_res + z_t_mamba
        
        # 2. Cross-Attention (Conditioning)
        z_t_res = z_t
        z_t = self.norm_cross(z_t)
        
        # query=z_t, key=text, value=text
        attn_output, _ = self.cross_attn(
            query=z_t,
            key=text_embeddings,
            value=text_embeddings,
            key_padding_mask=text_mask # Mask cho text
            # Không cần attn_mask (causal mask)
        )
        z_t = z_t_res + attn_output
        
        # 3. FFN
        z_t_res = z_t
        z_t = self.norm_ffn(z_t)
        z_t = z_t_res + self.ffn(z_t)
        
        return z_t

# ---------------------------------
# MÔ HÌNH DENOISER CHÍNH (DÙNG LÕI MỚI)
# ---------------------------------

class LDM_Mamba_Denoiser(nn.Module):
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
        
        # 1. Input/Output Projections
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        # 2. Positional Encoding (phiên bản batch_first)
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # 3. Time Embedding
        self.time_embed = nn.Sequential(
            TimeEmbed(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 4. Kiến trúc lõi (Một chồng các khối MambaCrossAttnLayer)
        self.layers = nn.ModuleList([
            MambaCrossAttnLayer(
                d_model=hidden_dim,
                nhead=num_heads
            ) for _ in range(num_layers)
        ])
        
        # 5. LayerNorm cuối cùng
        self.norm_out = nn.LayerNorm(hidden_dim)

    def forward(self, z_t, timesteps, text_embeddings, text_mask=None, pose_mask=None):
        """
        Args:
            z_t: [B, T, latent_dim]
            timesteps: [B]
            text_embeddings: [B, L, text_embed_dim] (Từ BERT)
            text_mask: [B, L] (True = ignore)
            pose_mask: [B, T] (True = ignore)
        """
        
        # 1. Chuẩn bị Pose Latent (tgt) [B, T, D]
        z_t_embed = self.input_proj(z_t)
        
        # Cộng Time embedding [B, D] -> [B, 1, D]
        t_embed = self.time_embed(timesteps)
        z_t_embed = z_t_embed + t_embed.unsqueeze(1)
        
        # Cộng Positional Encoding
        z_t_embed = self.pos_encoder(z_t_embed) # Đã là [B, T, D]
        
        # 2. Chuẩn bị Text (memory)
        # `text_embeddings` đã sẵn sàng [B, L, D]
        
        # 3. Chạy qua các khối "lai"
        output = z_t_embed
        for layer in self.layers:
            output = layer(
                z_t=output,
                text_embeddings=text_embeddings,
                text_mask=text_mask,
                pose_mask=pose_mask
            )
        
        output = self.norm_out(output)
        
        # 4. Lấy output
        noise_pred = self.output_proj(output)
        return noise_pred