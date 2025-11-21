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
        """
        manual_feat = self.manual_proj(latent)  # [B, T, hidden_dim//2]
        nmm_feat = self.nmm_proj(latent)  # [B, T, hidden_dim//2]
        
        combined = torch.cat([manual_feat, nmm_feat], dim=-1)  # [B, T, hidden_dim]
        
        corr_pred = self.corr_predictor(combined).squeeze(-1)  # [B, T]
        
        if mask is not None:
            corr_pred = corr_pred * mask.float()
        
        return corr_pred
    
    def compute_loss(self, latent, pose_gt, mask=None):
        """
        Compute sync score (correlation) từ pose_gt
        
        Args:
            latent: [B, T, latent_dim] - (Dùng để lấy device)
            pose_gt: [B, T, 214] - ground truth pose
        
        Returns:
            losses: [B] - Tensor 1D chứa negative correlation (loss) cho mỗi sample
        """
        manual_gt = pose_gt[:, :, :150]
        nmm_gt = pose_gt[:, :, 150:]
        
        B, T, _ = manual_gt.shape
        device = latent.device
        
        if mask is None:
             mask = torch.ones(B, T, device=device, dtype=torch.bool)
        
        losses = []
        for i in range(B):
            valid = mask[i]
            if valid.sum() < 2:
                losses.append(torch.tensor(0.0, device=device))
                continue
            
            manual_signal = manual_gt[i, valid].mean(dim=-1)
            nmm_signal = nmm_gt[i, valid].mean(dim=-1)
            
            # Normalize (zero-mean)
            manual_flat = (manual_signal - manual_signal.mean()) / (manual_signal.std() + 1e-6)
            nmm_flat = (nmm_signal - nmm_signal.mean()) / (nmm_signal.std() + 1e-6)

            # Correlation
            corr = (manual_flat * nmm_flat).mean()
            
            # Loss: negative correlation (maximize correlation = minimize -corr)
            loss = -corr
            losses.append(loss)
        
        if len(losses) == 0:
            return torch.tensor(0.0, device=device).unsqueeze(0) # Đảm bảo shape [1]
        
        return torch.stack(losses) # Shape [B]
    
    def compute_gradient(self, latent, pose_gt, mask=None):
        """
        Compute gradient for guidance: ∇_z L_sync
        """
        # Cần bật grad trên latent để tính
        latent_grad = latent.detach().requires_grad_(True)
        
        # compute_loss trả về [B]
        # Loss ở đây là negative correlation (L_sync)
        loss_per_sample = self.compute_loss(latent_grad, pose_gt, mask)
        
        # Mean để lấy scalar loss cho autograd
        loss = loss_per_sample.mean()

        # Tính gradient
        grad = torch.autograd.grad(
            loss, 
            latent_grad, 
            grad_outputs=torch.ones_like(loss), 
            create_graph=False,
            retain_graph=False # Dùng False vì không cần cho backward tiếp
        )[0]
        
        return grad