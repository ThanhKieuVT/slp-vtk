import torch
import torch.nn as nn
from transformers import BertModel

from .flow_matching import FlowMatchingBlock, FlowMatchingScheduler, FlowMatchingLoss
from .mamba_prior import SimpleSSMPrior
from .sync_guidance import SyncGuidanceHead

# === MỚI: Class Dự Đoán Độ Dài ===
class LengthPredictor(nn.Module):
    """Mạng nhỏ để dự đoán số lượng frames từ text features"""
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) # Output ra 1 số thực (số frames)
        )

    def forward(self, text_features, mask=None):
        # text_features: [Batch, Seq_Len, Dim]
        # Gom thông tin cả câu lại để đoán độ dài
        if mask is not None:
            # Masking để không tính padding vào trung bình
            # mask: True = ignore (padding), False = keep
            # Đổi lại logic mask: 1 = keep, 0 = ignore
            keep_mask = (~mask).float().unsqueeze(-1) # [B, L, 1]
            pooled_text = (text_features * keep_mask).sum(dim=1) / keep_mask.sum(dim=1).clamp(min=1)
        else:
            pooled_text = text_features.mean(dim=1) 
            
        pred_length = self.net(pooled_text)
        return pred_length.squeeze(-1) # [Batch]


class LatentFlowMatcher(nn.Module):
    """
    Latent Flow Matcher SOTA: Kèm theo Length Predictor
    """
    
    def __init__(
        self,
        latent_dim=256,
        text_encoder_name='bert-base-multilingual-cased',
        hidden_dim=512,
        num_flow_layers=6,
        num_prior_layers=4,
        num_heads=8,
        dropout=0.1,
        use_ssm_prior=True,
        use_sync_guidance=True,
        lambda_prior=0.1,
        gamma_guidance=0.01,
        lambda_anneal=True,
        W_PRIOR=0.1,
        W_SYNC=0.5,
        W_LENGTH=1.0 # Trọng số Loss cho độ dài
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_ssm_prior = use_ssm_prior
        self.use_sync_guidance = use_sync_guidance
        self.lambda_prior = lambda_prior
        self.gamma_guidance = gamma_guidance
        self.lambda_anneal = lambda_anneal
        
        self.W_PRIOR = W_PRIOR
        self.W_SYNC = W_SYNC
        self.W_LENGTH = W_LENGTH
        
        # Text encoder
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, hidden_dim) 
        
        # === MỚI: Length Predictor ===
        self.length_predictor = LengthPredictor(input_dim=hidden_dim)
        self.length_loss_fn = nn.MSELoss()
        
        # Flow scheduler
        self.scheduler = FlowMatchingScheduler()
        
        # Flow Block
        self.flow_block = FlowMatchingBlock(
            data_dim=latent_dim,
            condition_dim=hidden_dim, 
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # SSM Prior
        if use_ssm_prior:
            self.ssm_prior = SimpleSSMPrior(latent_dim, hidden_dim, num_prior_layers, num_heads, dropout)
        else:
            self.ssm_prior = None
        
        # Sync Guidance
        if use_sync_guidance:
            self.sync_head = SyncGuidanceHead(latent_dim, hidden_dim // 2, dropout)
        else:
            self.sync_head = None
        
        # Losses
        self.flow_loss_fn = FlowMatchingLoss()
        self.prior_reg_fn = nn.MSELoss()
        self.sync_loss_fn = nn.MSELoss()
    
    def encode_text(self, text_tokens, attention_mask):
        self.text_encoder.eval()
        with torch.no_grad():
            outputs = self.text_encoder(text_tokens, attention_mask=attention_mask)
        text_features = self.text_proj(outputs.last_hidden_state) 
        text_mask = ~attention_mask.bool()
        return text_features, text_mask
    
    def get_lambda_t(self, t):
        t_float = t.float()
        if self.lambda_anneal:
            return self.lambda_prior * (1 - t_float.view(-1, 1, 1))
        else:
            return torch.full((t.shape[0],1,1), self.lambda_prior, device=t.device, dtype=t_float.dtype)
    
    def forward(self, batch, gt_latent, pose_gt=None, mode='train', num_inference_steps=50, return_attn_weights=False):
        text_features, text_mask = self.encode_text(batch['text_tokens'], batch['attention_mask'])
        
        if mode == 'train':
            return self._train_forward(batch, text_features, text_mask, gt_latent, pose_gt, return_attn_weights)
        else:
            return self._inference_forward(batch, text_features, text_mask, num_inference_steps)
    
    def _train_forward(self, batch, text_features, text_mask, gt_latent, pose_gt=None, return_attn_weights=False):
        device = text_features.device
        B, T, D = gt_latent.shape
        
        # === 1. TRAINING LENGTH PREDICTOR ===
        # text_mask đang là True=Padding, nên truyền nguyên vào LengthPredictor (đã xử lý logic bên trong)
        pred_length = self.length_predictor(text_features, text_mask)
        target_length = batch['seq_lengths'].float()
        length_loss = self.length_loss_fn(pred_length, target_length)
        
        # === 2. FLOW MATCHING LOGIC ===
        t = self.scheduler.sample_timesteps(B, device)
        seq_lengths = batch.get('seq_lengths', torch.full((B,), T, device=device))
        mask = torch.arange(T, device=device)[None, :] < seq_lengths[:, None]
        
        latent_t, v_gt, latent_0 = self.scheduler.add_noise(gt_latent, t)
        
        # Flow Prediction
        latent_t_input = latent_t.detach().requires_grad_(False) # Không cần grad cho Flow
        if self.use_sync_guidance: # Chỉ bật grad nếu cần tính cho Sync Head (Optional, ở đây ta train riêng nên False cũng dc)
             latent_t_input = latent_t.detach().requires_grad_(True)

        pose_attn_mask = ~mask
        
        flow_output = self.flow_block(
            latent_t_input, t, text_features, 
            text_attn_mask=text_mask, pose_attn_mask=pose_attn_mask, 
            return_attn=return_attn_weights
        )
        
        if return_attn_weights: v_flow, attn_weights = flow_output
        else: v_flow = flow_output
        
        # Prior
        v_prior = None
        if self.use_ssm_prior and self.ssm_prior is not None:
            with torch.no_grad():
                v_prior = self.ssm_prior(latent_t.detach(), t, text_features, mask=mask)
        
        lambda_t = self.get_lambda_t(t)
        v_pred_train = v_flow + lambda_t * v_prior if v_prior is not None else v_flow
        
        # Sync Guidance Training
        sync_loss_train = torch.tensor(0.0, device=device)
        if self.use_sync_guidance and self.sync_head is not None and pose_gt is not None:
            sync_score_pred = self.sync_head(latent_t_input, mask)
            with torch.no_grad():
                sync_loss_target = self.sync_head.compute_loss(latent_t.detach(), pose_gt, mask)
            
            masked_score_pred = sync_score_pred * mask.float()
            pred_mean = masked_score_pred.sum(dim=1) / mask.sum(dim=1).clamp(min=1).float()
            sync_loss_train = self.sync_loss_fn(pred_mean, sync_loss_target.detach())
        
        # Losses Combine
        flow_loss = self.flow_loss_fn(v_pred_train, v_gt, mask=mask)        
        prior_loss = self.prior_reg_fn(v_flow, v_prior.detach()) if v_prior is not None else torch.tensor(0.0, device=device)
        
        # Tổng loss bao gồm cả Length Loss
        total_loss = flow_loss + self.W_PRIOR * prior_loss + self.W_SYNC * sync_loss_train + self.W_LENGTH * length_loss
        
        result = {
            'total': total_loss, 'flow': flow_loss, 
            'prior': prior_loss, 'sync': sync_loss_train, 
            'length': length_loss # Log ra để xem
        }
        if return_attn_weights: result['attn_weights'] = attn_weights
            
        return result
    
    def _inference_forward(self, batch, text_features, text_mask, num_steps=50):
        device = text_features.device
        B = text_features.shape[0]
        
        # === QUAN TRỌNG: TỰ ĐỘNG DỰ ĐOÁN ĐỘ DÀI ===
        with torch.no_grad():
            pred_len = self.length_predictor(text_features, text_mask)
            # Clamp độ dài: tối thiểu 10 frames, tối đa 300 frames (tuỳ dataset)
            target_lengths = pred_len.round().long().clamp(min=10, max=400)
            
        # Lấy độ dài lớn nhất trong batch này làm chuẩn
        T = target_lengths.max().item()
        
        # Tạo mask thật sự dựa trên độ dài dự đoán
        mask = torch.arange(T, device=device)[None, :] < target_lengths[:, None]
        
        # Khởi tạo latent ngẫu nhiên với kích thước T vừa đoán
        latent = torch.randn(B, T, self.latent_dim, device=device)
        dt = 1.0 / num_steps
        pose_attn_mask = ~mask
        
        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((B,), t_val, device=device)
            latent_grad = latent.detach().requires_grad_(True)
            
            v_flow = self.flow_block(latent_grad, t, text_features, text_attn_mask=text_mask, pose_attn_mask=pose_attn_mask)
            
            v_prior = None
            if self.use_ssm_prior and self.ssm_prior is not None:
                with torch.no_grad():
                    v_prior = self.ssm_prior(latent_grad.detach(), t, text_features, mask=mask)
            
            lambda_t = self.get_lambda_t(t)
            v_pred_raw = v_flow + lambda_t * v_prior if v_prior is not None else v_flow
            v_pred = v_pred_raw
            
            # Sync guidance (inference)
            if self.use_sync_guidance and self.sync_head is not None and 0.2 < t_val < 0.8:
                sync_score_pred = self.sync_head(latent_grad, mask)
                guidance_loss = (sync_score_pred * mask.float()).sum(dim=1) / mask.sum(dim=1).clamp(min=1).float()
                guidance_loss = guidance_loss.mean()
                
                sync_grad = torch.autograd.grad(guidance_loss, latent_grad, create_graph=False)[0]
                v_pred = v_pred_raw - self.gamma_guidance * sync_grad.detach()
            
            with torch.no_grad():
                latent = latent + dt * v_pred
                latent = latent * mask.unsqueeze(-1).float() # Giữ sạch nền
        
        return latent