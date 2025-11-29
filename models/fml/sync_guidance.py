import torch
import torch.nn as nn
import numpy as np


class SyncGuidanceHead(nn.Module):
    def __init__(self, latent_dim, hidden_dim, dropout=0.1):
        super().__init__()
        # Nhánh xử lý Latent Pose
        self.pose_proj = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Nhánh xử lý Text (Lưu ý: hidden_dim của Text Projector bên ngoài là 512)
        # Ta giả định input text_features đã có dim = 512 từ LatentFlowMatcher
        self.text_proj = nn.Sequential(
            nn.Linear(512, hidden_dim), # 512 khớp với hidden_dim của LatentFlowMatcher
            nn.GELU(),
            nn.Dropout(dropout)
        )
        
        # Fusion & Score
        self.score_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1) # Output score scalar tại mỗi timestep
        )

    def forward(self, latent, text, mask=None):
        """
        latent: [B, T, D]
        text:   [B, D] (Global text feature) hoặc [B, T, D] (nếu đã expand)
        mask:   [B, T] (True = Valid)
        """
        # 1. Project Features
        h_pose = self.pose_proj(latent) # [B, T, H]
        
        # Expand Text nếu cần để khớp với Time
        if text.dim() == 2:
            h_text = self.text_proj(text).unsqueeze(1).expand(-1, latent.size(1), -1) # [B, T, H]
        else:
            h_text = self.text_proj(text)

        # 2. Concat & Predict
        cat_feat = torch.cat([h_pose, h_text], dim=-1) # [B, T, 2H]
        scores = self.score_net(cat_feat).squeeze(-1)  # [B, T]
        
        # 3. Masking (Gán score cực thấp cho phần padding để không ảnh hưởng max/mean)
        if mask is not None:
            scores = scores * mask.float()
            
        return scores
    
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