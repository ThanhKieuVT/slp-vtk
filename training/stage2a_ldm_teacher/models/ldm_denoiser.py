# Tên file: models/ldm_denoiser.py
# (Đã sửa tham số cho mCLIP)
import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """Positional encoding (sin/cos)"""
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        """x shape: [SeqLen, Batch, Dim]"""
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)

class TimeEmbed(nn.Module):
    """Sinusoidal time embedding"""
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

class LDM_TransformerDenoiser(nn.Module):
    """
    Mô hình Denoiser LDM dựa trên Transformer.
    Kiến trúc này đã tích hợp Cross-Attention.
    """
    def __init__(
        self,
        latent_dim=256,
        # === SỬA LỖI 1: Sửa tham số cho mCLIP ===
        text_embed_dim=1024, # (mCLIP-Large là 1024)
        hidden_dim=1024,     # (Dim nội bộ = text_embed_dim)
        num_layers=6,
        num_heads=16,        # (XLM-R Large dùng 16 heads)
        dropout=0.1
    ):
        super().__init__()
        # === SỬA LỖI 2: Thêm dòng này ===
        self.latent_dim = latent_dim 
        
        # 1. Input/Output Projections
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        # 2. Positional Encoding cho pose
        self.pos_encoder = PositionalEncoding(hidden_dim, dropout)
        
        # 3. Time Embedding
        self.time_embed = nn.Sequential(
            TimeEmbed(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # 4. Kiến trúc Transformer Decoder (có Cross-Attention)
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True # Quan trọng!
        )
        self.transformer_decoder = nn.TransformerDecoder(
            decoder_layer, 
            num_layers=num_layers
        )

    def forward(self, z_t, timesteps, text_embeddings, text_mask=None, pose_mask=None):
        """
        Args:
            z_t: [B, T, latent_dim] - Latent pose bị nhiễu
            timesteps: [B] - Timesteps (số nguyên)
            text_embeddings: [B, L, text_embed_dim] - Embeddings từ mCLIP
            text_mask: [B, L] - Mask cho text (True = ignore)
            pose_mask: [B, T] - Mask cho pose (True = ignore)
        """
        
        # 1. Chuẩn bị Pose Latent (tgt)
        z_t_embed = self.input_proj(z_t) # [B, T, hidden_dim]
        
        # Thêm Time embedding
        t_embed = self.time_embed(timesteps) # [B, hidden_dim]
        z_t_embed = z_t_embed + t_embed.unsqueeze(1) # [B, T, hidden_dim]
        
        # Thêm Positional encoding (cần đổi sang [T, B, D])
        z_t_embed = z_t_embed.transpose(0, 1) # [T, B, hidden_dim]
        z_t_embed = self.pos_encoder(z_t_embed)
        z_t_embed = z_t_embed.transpose(0, 1) # [B, T, hidden_dim]
        
        # 2. Chuẩn bị Text (memory)
        # `text_embeddings` đã sẵn sàng, shape [B, L, text_embed_dim]
        
        # 3. Chạy qua Transformer Decoder
        output = self.transformer_decoder(
            tgt=z_t_embed,
            memory=text_embeddings,
            tgt_key_padding_mask=pose_mask,
            memory_key_padding_mask=text_mask
        )
        
        # 4. Lấy output
        noise_pred = self.output_proj(output) # [B, T, latent_dim]
        return noise_pred