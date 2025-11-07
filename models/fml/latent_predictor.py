"""
Stage 2: Latent Predictor
Text → Latent [B,T,256]
"""
import torch
import torch.nn as nn
import math
from transformers import BertModel


class PositionalEncoding(nn.Module):
    """Positional encoding cho Transformer"""
    
    def __init__(self, d_model, max_len=5000, dropout=0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LatentPredictor(nn.Module):
    """
    Predictor: Text → Latent Space
    Sử dụng Transformer Decoder với learnable queries (giống DETR)
    """
    
    def __init__(
        self,
        latent_dim=256,
        text_encoder_name='bert-base-multilingual-cased',
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        num_queries=100,  # Số lượng learnable queries (sẽ pad/truncate theo target length)
        dropout=0.1,
        max_len=5000
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.num_queries = num_queries
        
        # Text encoder (BERT)
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        text_dim = self.text_encoder.config.hidden_size  # 768
        
        # Project text features
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Learnable queries (queries để query từ text features)
        self.query_embed = nn.Parameter(torch.randn(num_queries, hidden_dim))
        
        # Positional encoding cho queries
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len, dropout)
        
        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        
        # Output projection to latent
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, text_tokens, attention_mask, target_length=None, mask=None):
        """
        Args:
            text_tokens: [B, L] token IDs
            attention_mask: [B, L] attention mask
            target_length: int hoặc [B] - độ dài target sequence (nếu None, dùng num_queries)
            mask: [B, T] mask cho output (nếu có)
        
        Returns:
            latent: [B, T, 256]
        """
        B = text_tokens.shape[0]
        
        # Encode text
        text_outputs = self.text_encoder(text_tokens, attention_mask=attention_mask)
        text_features = text_outputs.last_hidden_state  # [B, L, 768]
        text_features = self.text_proj(text_features)  # [B, L, hidden_dim]
        
        # Determine target length
        if target_length is None:
            T = self.num_queries
        elif isinstance(target_length, int):
            T = target_length
        else:
            T = target_length.max().item()
        
        # Create queries (learnable + positional encoding)
        queries = self.query_embed[:T].unsqueeze(0).expand(B, -1, -1)  # [B, T, hidden_dim]
        queries = self.pos_encoding(queries)
        
        # Create text attention mask (True = ignore padding)
        text_attn_mask = ~attention_mask.bool()  # [B, L]
        
        # Decoder: queries attend to text_features
        decoded = self.decoder(
            queries,
            text_features,
            tgt_key_padding_mask=None,  # Queries không có padding
            memory_key_padding_mask=text_attn_mask  # Text có padding
        )  # [B, T, hidden_dim]
        
        # Project to latent
        latent = self.output_proj(decoded)  # [B, T, 256]
        
        # Apply mask nếu có (zero out padding positions)
        if mask is not None:
            latent = latent * mask.unsqueeze(-1).float()
        
        return latent

