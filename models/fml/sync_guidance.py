"""
Sync Guidance: Head để tối ưu correlation tay-mặt
"""
import torch
import torch.nn as nn
import numpy as np


class SyncGuidanceHead(nn.Module):
    """
    Sync Guidance Head: Học correlation giữa manual (tay) và NMM (mặt) trong latent
    """
    
    def __init__(
        self,
        latent_dim=256,
        hidden_dim=256,
        dropout=0.1
    ):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Project latent to manual and NMM features
        self.manual_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        self.nmm_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2)
        )
        
        # Correlation predictor
        self.corr_predictor = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Tanh()  # Output [-1, 1]
        )
    
    def forward(self, latent, mask=None):
        """
        Predict sync quality (correlation) từ latent
        
        Args:
            latent: [B, T, latent_dim]
            mask: [B, T] - valid mask
        
        Returns:
            corr_pred: [B, T] - predicted correlation
        """
        # Project to manual and NMM features
        manual_feat = self.manual_proj(latent)  # [B, T, hidden_dim//2]
        nmm_feat = self.nmm_proj(latent)  # [B, T, hidden_dim//2]
        
        # Concatenate
        combined = torch.cat([manual_feat, nmm_feat], dim=-1)  # [B, T, hidden_dim]
        
        # Predict correlation
        corr_pred = self.corr_predictor(combined).squeeze(-1)  # [B, T]
        
        if mask is not None:
            corr_pred = corr_pred * mask.float()
        
        return corr_pred
    
    def compute_loss(self, latent, pose_gt, mask=None):
        """
        Compute sync loss: -correlation + lag_penalty
        
        Args:
            latent: [B, T, latent_dim] - predicted latent
            pose_gt: [B, T, 214] - ground truth pose
            mask: [B, T] - valid mask
        
        Returns:
            loss: scalar
        """
        # Decode latent to pose (cần decoder, sẽ pass từ outside)
        # Ở đây chỉ compute correlation từ pose_gt
        
        # Extract manual and NMM from pose
        manual_gt = pose_gt[:, :, :150]  # [B, T, 150]
        nmm_gt = pose_gt[:, :, 150:]  # [B, T, 64]
        
        # Flatten
        B, T, _ = manual_gt.shape
        if mask is not None:
            valid_mask = mask
        else:
            valid_mask = torch.ones(B, T, device=manual_gt.device, dtype=torch.bool)
        
        # Compute correlation per sample
        losses = []
        for i in range(B):
            valid = valid_mask[i]
            if valid.sum() < 2:
                continue
            
            # === SỬA LỖI Ở ĐÂY ===
            # Thay vì .flatten(), dùng .mean(dim=-1) để tạo 1D signal
            manual_signal = manual_gt[i, valid].mean(dim=-1) # [T_valid]
            nmm_signal = nmm_gt[i, valid].mean(dim=-1)    # [T_valid]
            
            # Normalize (Đổi tên biến)
            manual_flat = (manual_signal - manual_signal.mean()) / (manual_signal.std() + 1e-6)
            nmm_flat = (nmm_signal - nmm_signal.mean()) / (nmm_signal.std() + 1e-6)
            # === KẾT THÚC SỬA LỖI ===

            # Correlation (Giờ 2 tensor đã cùng kích thước [T_valid])
            corr = (manual_flat * nmm_flat).mean()
            
            # Loss: negative correlation (maximize correlation = minimize -corr)
            loss = -corr
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=latent.device)
        
        return torch.stack(losses).mean()
    
    def compute_gradient(self, latent, pose_gt, mask=None):
        """
        Compute gradient for guidance: ∇_z L_sync
        
        Args:
            latent: [B, T, latent_dim]
            pose_gt: [B, T, 214]
            mask: [B, T]
        
        Returns:
            grad: [B, T, latent_dim] - gradient
        """
        # Cần bật grad trên latent để tính
        latent_grad = latent.detach().requires_grad_(True)
        
        loss = self.compute_loss(latent_grad, pose_gt, mask)
        
        # Tính gradient
        grad = torch.autograd.grad(
            loss, 
            latent_grad, 
            grad_outputs=torch.ones_like(loss), # Cần cho scalar loss
            create_graph=False
        )[0]
        
        return grad