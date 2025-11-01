# models/temporal_sync.py
"""
Temporal Synchronization Module
Aligns manual signs with non-manual markers
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalSynchronizer(nn.Module):
    """
    Synchronize manual and NMM sequences
    
    Input:
        - Manual features: [B, T, 150]
        - NMM features: [B, T, 64]
    
    Output:
        - Synchronized manual: [B, T, 150]
        - Synchronized NMM: [B, T, 64]
    
    Architecture:
        - Bidirectional GRU for temporal modeling
        - Cross-attention between manual and NMM
        - Residual connections
    """
    def __init__(
        self,
        manual_dim=150,
        nmm_dim=64,
        hidden_dim=512,
        num_gru_layers=3,
        num_heads=8,
        dropout=0.1
    ):
        super().__init__()
        
        self.manual_dim = manual_dim
        self.nmm_dim = nmm_dim
        self.hidden_dim = hidden_dim
        
        # Project to common dimension
        self.manual_proj = nn.Linear(manual_dim, hidden_dim)
        self.nmm_proj = nn.Linear(nmm_dim, hidden_dim)
        
        # Bidirectional GRU for temporal modeling
        self.manual_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0
        )
        
        self.nmm_gru = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0
        )
        
        # After bidirectional, hidden_dim → 2*hidden_dim
        # Project back to hidden_dim
        self.manual_gru_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        self.nmm_gru_proj = nn.Linear(hidden_dim * 2, hidden_dim)
        
        # Cross-attention: manual attends to NMM
        self.manual_to_nmm_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention: NMM attends to manual
        self.nmm_to_manual_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.manual_norm1 = nn.LayerNorm(hidden_dim)
        self.manual_norm2 = nn.LayerNorm(hidden_dim)
        self.nmm_norm1 = nn.LayerNorm(hidden_dim)
        self.nmm_norm2 = nn.LayerNorm(hidden_dim)
        
        # Output projections
        self.manual_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, manual_dim)
        )
        
        self.nmm_out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, nmm_dim)
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        print(f"TemporalSynchronizer: {total_params/1e6:.1f}M parameters")
    
    def forward(self, manual_seq, nmm_seq, mask=None):
        """
        Forward pass
        
        Args:
            manual_seq: [B, T, 150] manual poses
            nmm_seq: [B, T, 64] NMMs
            mask: [B, T] padding mask (True = padding)
        
        Returns:
            synced_manual: [B, T, 150]
            synced_nmm: [B, T, 64]
        """
        # 1. Project to common space
        manual_feat = self.manual_proj(manual_seq)  # [B, T, hidden_dim]
        nmm_feat = self.nmm_proj(nmm_seq)           # [B, T, hidden_dim]
        
        # 2. Temporal modeling with GRU
        manual_gru_out, _ = self.manual_gru(manual_feat)  # [B, T, 2*hidden_dim]
        nmm_gru_out, _ = self.nmm_gru(nmm_feat)          # [B, T, 2*hidden_dim]
        
        manual_gru_out = self.manual_gru_proj(manual_gru_out)  # [B, T, hidden_dim]
        nmm_gru_out = self.nmm_gru_proj(nmm_gru_out)          # [B, T, hidden_dim]
        
        # Residual
        manual_feat = self.manual_norm1(manual_feat + manual_gru_out)
        nmm_feat = self.nmm_norm1(nmm_feat + nmm_gru_out)
        
        # 3. Cross-attention
        # Manual attends to NMM
        manual_cross, _ = self.manual_to_nmm_attn(
            query=manual_feat,
            key=nmm_feat,
            value=nmm_feat,
            key_padding_mask=mask  # [B, T]
        )
        manual_feat = self.manual_norm2(manual_feat + manual_cross)
        
        # NMM attends to manual
        nmm_cross, _ = self.nmm_to_manual_attn(
            query=nmm_feat,
            key=manual_feat,
            value=manual_feat,
            key_padding_mask=mask
        )
        nmm_feat = self.nmm_norm2(nmm_feat + nmm_cross)
        
        # 4. Project to output space
        synced_manual = self.manual_out(manual_feat)  # [B, T, 150]
        synced_nmm = self.nmm_out(nmm_feat)           # [B, T, 64]
        
        # Residual with input
        synced_manual = synced_manual + manual_seq
        synced_nmm = synced_nmm + nmm_seq
        
        return synced_manual, synced_nmm
    
    def compute_sync_loss(self, manual_seq, nmm_seq, gt_manual, gt_nmm, mask=None):
        """
        Compute synchronization loss
        
        Args:
            manual_seq, nmm_seq: Generated sequences
            gt_manual, gt_nmm: Ground truth
            mask: Padding mask
        
        Returns:
            Total sync loss
        """
        # Forward
        synced_manual, synced_nmm = self.forward(manual_seq, nmm_seq, mask)
        
        # MSE losses
        loss_manual = F.mse_loss(synced_manual, gt_manual, reduction='none')
        loss_nmm = F.mse_loss(synced_nmm, gt_nmm, reduction='none')
        
        # Apply mask
        if mask is not None:
            valid_mask = ~mask  # Invert: True = valid
            loss_manual = (loss_manual * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum()
            loss_nmm = (loss_nmm * valid_mask.unsqueeze(-1)).sum() / valid_mask.sum()
        else:
            loss_manual = loss_manual.mean()
            loss_nmm = loss_nmm.mean()
        
        # Cross-correlation loss (encourage alignment)
        # Compute correlation between manual and NMM velocities
        manual_vel = synced_manual[:, 1:] - synced_manual[:, :-1]
        nmm_vel = synced_nmm[:, 1:] - synced_nmm[:, :-1]
        
        manual_vel_norm = F.normalize(manual_vel, dim=-1)
        nmm_vel_norm = F.normalize(nmm_vel, dim=-1)
        
        # Cosine similarity (higher = better aligned)
        correlation = (manual_vel_norm * nmm_vel_norm).sum(dim=-1).mean()
        loss_correlation = 1.0 - correlation  # Convert to loss
        
        # Total loss
        total_loss = loss_manual + loss_nmm + 0.1 * loss_correlation
        
        return {
            'total': total_loss,
            'manual': loss_manual,
            'nmm': loss_nmm,
            'correlation': loss_correlation
        }