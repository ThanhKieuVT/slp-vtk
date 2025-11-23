"""
Mamba/SSM Prior: Học drift có cấu trúc thời gian dài hạn
FIXED: Added 'condition' argument to forward() to match LatentFlowMatcher call signature.
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
    Mamba Prior: Học drift v_prior(z, t, condition) từ latent sequence
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
        
        # Condition projection (Text features -> Hidden)
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim) # Giả sử text_feature cũng dim=hidden_dim

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
                    # dropout=dropout # Mamba gốc có thể không có tham số dropout trong init, tuỳ version
                ) for _ in range(num_layers)
            ])
        else:
            raise ValueError("MambaPrior requires mamba-ssm. Use SimpleSSMPrior instead.")
        
        self.norm = nn.LayerNorm(hidden_dim)
        # Output projection to velocity
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z, t, condition, mask=None):
        """
        Predict prior velocity v_prior(z, t, condition)
        
        Args:
            z: [B, T, latent_dim] - latent sequence
            t: [B] - timesteps
            condition: [B, L, hidden_dim] - text features (Tên biến này quan trọng để khớp code gọi)
            mask: [B, T] - valid mask
        
        Returns:
            v_prior: [B, T, latent_dim] - prior velocity
        """
        B, T, D = z.shape
        
        # 1. Project Input & Time
        x = self.input_proj(z)  # [B, T, hidden_dim]
        t_emb = self.time_embed(t.unsqueeze(-1)).unsqueeze(1) # [B, 1, hidden_dim]
        
        # 2. Inject Condition (Text)
        # Cách đơn giản nhất cho Prior: Global Average Pooling text features rồi cộng vào
        # (Để giữ Mamba hoạt động trên chuỗi thời gian T của pose)
        if condition is not None:
            # condition: [B, L, H] -> [B, H] (Mean Pooling)
            cond_global = condition.mean(dim=1).unsqueeze(1) # [B, 1, H]
            cond_emb = self.cond_proj(cond_global)
            x = x + cond_emb
            
        x = x + t_emb
        
        # 3. Mamba Layers
        for mamba_layer in self.mamba_layers:
            x_res = x
            x = mamba_layer(x)  
            x = x + x_res # Residual connection thủ công nếu Mamba block không có
            
            if mask is not None:
                x = x * mask.unsqueeze(-1).float()
        
        x = self.norm(x)
        v_prior = self.output_proj(x)
        
        return v_prior


class SimpleSSMPrior(nn.Module):
    """
    Simple SSM Prior (Fallback dùng Transformer Encoder)
    FIXED: Thêm tham số 'condition' vào forward để khớp gọi hàm.
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
        
        self.input_proj = nn.Linear(latent_dim, hidden_dim)
        
        # Time embedding
        self.time_embed = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Condition handling (Cross-Attention like mechanism or simple addition)
        # Ở đây dùng Transformer Encoder thì ta cộng Global Condition vào Input
        self.cond_proj = nn.Linear(hidden_dim, hidden_dim)

        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(hidden_dim, latent_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, z, t, condition, mask=None):
        """
        Args:
            z: [B, T, latent_dim]
            t: [B]
            condition: [B, L, hidden_dim] (Text features)
            mask: [B, T]
        """
        B, T, D = z.shape
        
        # 1. Embeddings
        x = self.input_proj(z)
        t_emb = self.time_embed(t.unsqueeze(-1)).unsqueeze(1) # [B, 1, H]
        x = x + t_emb
        
        # 2. Condition Injection
        if condition is not None:
            # Mean pooling text features cho đơn giản & nhẹ
            cond_global = condition.mean(dim=1).unsqueeze(1) # [B, 1, H]
            x = x + self.cond_proj(cond_global)

        # 3. Masking
        if mask is not None:
            # Transformer nhận mask: True = ignore (padding), False = keep
            # mask đầu vào của mình: True = keep, False = ignore (padding)
            # => Cần đảo ngược: ~mask
            src_key_padding_mask = ~mask.bool()
        else:
            src_key_padding_mask = None
        
        # 4. Transformer
        x = self.transformer(x, src_key_padding_mask=src_key_padding_mask)
        
        # 5. Output
        v_prior = self.output_proj(x)
        
        return v_prior