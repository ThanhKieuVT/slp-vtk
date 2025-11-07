"""
Mamba/SSM Prior: Học drift có cấu trúc thời gian dài hạn
"""
import torch
import torch.nn as nn

# Try import mamba_ssm, fallback to SimpleSSM nếu không có
try:
    from mamba_ssm import Mamba
    HAS_MAMBA = True
except ImportError:
    HAS_MAMBA = False
    print("⚠️  mamba-ssm not installed, using SimpleSSMPrior instead")


class MambaPrior(nn.Module):
    """
    Mamba Prior: Học drift v_prior(z, t) từ latent sequence
    """
    
    def __init__(
        self,
        latent_dim=256,
        hidden_dim=512,
        state_dim=16,
        num_layers=4,
        dropout=0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Mamba layers (chỉ dùng nếu có mamba-ssm)
        if HAS_MAMBA:
            self.mamba_layers = nn.ModuleList([
                Mamba(
                    d_model=hidden_dim,
                    d_state=state_dim,
                    d_conv=4,
                    expand=2,
                    dropout=dropout
                ) for _ in range(num_layers)
            ])
        else:
            raise ValueError("MambaPrior requires mamba-ssm. Use SimpleSSMPrior instead.")
        
        # Output projection to velocity
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z, t, mask=None):
        """
        Predict prior velocity v_prior(z, t)
        
        Args:
            z: [B, T, latent_dim] - latent sequence
            t: [B] - timesteps
            mask: [B, T] - valid mask
        
        Returns:
            v_prior: [B, T, latent_dim] - prior velocity
        """
        B, T, D = z.shape
        
        # Project input
        x = self.input_proj(z)  # [B, T, hidden_dim]
        
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))  # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, hidden_dim]
        x = x + t_emb
        
        # Mamba layers
        for mamba_layer in self.mamba_layers:
            x = mamba_layer(x)  # [B, T, hidden_dim]
            if mask is not None:
                x = x * mask.unsqueeze(-1).float()
        
        # Output
        v_prior = self.output_proj(x)  # [B, T, latent_dim]
        
        return v_prior


class SimpleSSMPrior(nn.Module):
    """
    Simple SSM Prior (nếu không có mamba-ssm, dùng Transformer thay thế)
    """
    
    def __init__(
        self,
        latent_dim=256,
        hidden_dim=512,
        num_layers=4,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Input projection
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Transformer (thay cho Mamba)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output projection
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z, t, mask=None):
        """
        Predict prior velocity v_prior(z, t)
        
        Args:
            z: [B, T, latent_dim] - latent sequence
            t: [B] - timesteps
            mask: [B, T] - valid mask
        
        Returns:
            v_prior: [B, T, latent_dim] - prior velocity
        """
        B, T, D = z.shape
        
        # Project input
        x = self.input_proj(z)  # [B, T, hidden_dim]
        
        # Time embedding
        t_emb = self.time_embed(t.unsqueeze(-1))  # [B, hidden_dim]
        t_emb = t_emb.unsqueeze(1).expand(-1, T, -1)  # [B, T, hidden_dim]
        x = x + t_emb
        
        # Attention mask
        if mask is not None:
            attn_mask = ~mask  # [B, T]
        else:
            attn_mask = None
        
        # Transformer
        x = self.transformer(x, src_key_padding_mask=attn_mask)
        
        # Output
        v_prior = self.output_proj(x)  # [B, T, latent_dim]
        
        return v_prior

