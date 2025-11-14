# Tên file: models/losses.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class VelocityLoss(nn.Module):
    """
    Tính loss L1 hoặc L2 trên vận tốc (sự thay đổi giữa các frame) 
    để đảm bảo chuyển động mượt mà.
    
    loss = || (z_pred[t+1] - z_pred[t]) - (z_gt[t+1] - z_gt[t]) ||
    """
    def __init__(self, loss_type='l1'):
        super().__init__()
        self.loss_type = loss_type
        if loss_type == 'l1':
            self.loss_fn = F.l1_loss
        else:
            self.loss_fn = F.mse_loss
            
    def forward(self, pred_z0, gt_z0, mask=None):
        """
        Args:
            pred_z0: [B, T, D] - Latent z0 dự đoán (đã khử nhiễu)
            gt_z0: [B, T, D] - Latent z0 thực tế
            mask: [B, T] - Mask (True = valid)
        """
        
        # Tính vận tốc
        vel_pred = pred_z0[:, 1:, :] - pred_z0[:, :-1, :]
        vel_gt = gt_z0[:, 1:, :] - gt_z0[:, :-1, :]
        
        # Tính loss
        loss = self.loss_fn(vel_pred, vel_gt, reduction='none')
        
        if mask is not None:
            # Mask cho velocity (bị ngắn hơn 1 frame)
            vel_mask = mask[:, 1:]
            loss = loss * vel_mask.unsqueeze(-1).float()
            
            # Trung bình loss trên các frame hợp lệ
            return loss.sum() / vel_mask.sum().clamp(min=1)
        
        return loss.mean()