import torch
import torch.nn as nn

class FlowMatchingLoss(nn.Module):
    """
    Flow Matching Loss - FIXED SCALING
    
    FIX: Chia cho số SAMPLES thay vì (samples × dimensions)
    để tránh loss quá nhỏ làm yếu gradient
    """
    def __init__(self):
        super().__init__()
    
    def forward(self, v_pred, v_gt, mask=None):
        """
        Args:
            v_pred: [B, T, D] - Predicted velocity
            v_gt: [B, T, D] - Ground truth velocity
            mask: [B, T] - Valid frame mask
        """
        # MSE Loss
        loss = (v_pred - v_gt) ** 2  # [B, T, D]
        
        if mask is not None:
            # 🔥 FIX: Sum over D first, then average over valid T
            # Trước: loss.sum() / (mask.sum() * D) → Quá nhỏ
            # Sau: loss.sum(D) / mask.sum() → Vừa phải
            
            loss_per_frame = loss.sum(dim=-1)  # [B, T] - Sum over D
            masked_loss = loss_per_frame * mask.float()  # [B, T]
            
            total_valid_frames = mask.sum()
            loss = masked_loss.sum() / total_valid_frames.clamp(min=1)
        else:
            loss = loss.mean()
            
        return loss