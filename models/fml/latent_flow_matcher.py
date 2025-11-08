"""
Latent Flow Matcher với SSM Prior và Sync Guidance
"""
import torch
import torch.nn as nn
from transformers import BertModel

from .flow_matching import FlowMatchingBlock, FlowMatchingScheduler, FlowMatchingLoss
from .mamba_prior import SimpleSSMPrior  # Hoặc MambaPrior nếu có mamba-ssm
from .sync_guidance import SyncGuidanceHead


class LatentFlowMatcher(nn.Module):
    """
    Latent Flow Matcher với SSM Prior và Sync Guidance
    
    Velocity: v(z,t|x) = v_flow(z,t|x) + λ(t) * v_prior(z,t) - γ * ∇_z L_sync
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
        lambda_prior=0.1,  # Weight cho SSM prior
        gamma_guidance=0.01,  # Weight cho sync guidance
        lambda_anneal=True,  # Anneal lambda theo t
        # (Thêm 2 hằng số weight cho loss)
        W_PRIOR = 0.01,
        W_SYNC = 0.1
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
        
        # Text encoder
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        text_dim = self.text_encoder.config.hidden_size  # 768
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Flow scheduler
        self.scheduler = FlowMatchingScheduler()
        
        # Flow Matching Block
        self.flow_block = FlowMatchingBlock(
            data_dim=latent_dim,
            condition_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # SSM Prior (nếu dùng)
        if use_ssm_prior:
            self.ssm_prior = SimpleSSMPrior(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim,
                num_layers=num_prior_layers,
                num_heads=num_heads,
                dropout=dropout
            )
        else:
            self.ssm_prior = None
        
        # Sync Guidance (nếu dùng)
        if use_sync_guidance:
            self.sync_head = SyncGuidanceHead(
                latent_dim=latent_dim,
                hidden_dim=hidden_dim // 2,
                dropout=dropout
            )
        else:
            self.sync_head = None
        
        # Loss functions
        self.flow_loss_fn = FlowMatchingLoss()
        self.prior_reg_fn = nn.MSELoss()
        self.sync_loss_fn = nn.MSELoss() # Loss cho SyncHead
        
        self.dropout = nn.Dropout(dropout)
    
    def encode_text(self, text_tokens, attention_mask):
        """Encode text to features"""
        outputs = self.text_encoder(text_tokens, attention_mask=attention_mask)
        text_features = outputs.last_hidden_state  # [B, L, 768]
        text_features = self.text_proj(text_features)  # [B, L, hidden_dim]
        return text_features
    
    def get_lambda_t(self, t):
        """
        Anneal lambda theo t: λ(t) = λ * (1 - t)
        Ưu tiên prior ở early timesteps
        """
        # (SỬA LỖI 6: Thêm .float() cho an toàn)
        t_float = t.float()
        
        if self.lambda_anneal:
            # Chuyển t sang [B, 1, 1] để broadcast
            return self.lambda_prior * (1 - t_float.view(-1, 1, 1))
        else:
            # (SỬA LỖI 6: Trả về tensor)
            return torch.full((t.shape[0],1,1), self.lambda_prior, device=t.device, dtype=t_float.dtype)
    
    def forward(self, batch, gt_latent, pose_gt=None, mode='train', num_inference_steps=50):
        """
        Args:
            batch: dict với text_tokens, attention_mask
            gt_latent: [B, T, 256] - ground truth latent (chỉ khi train)
            pose_gt: [B, T, 214] - ground truth pose (cho sync guidance)
            mode: 'train' hoặc 'inference'
            num_inference_steps: số bước ODE (khi inference)
        """
        text_features = self.encode_text(
            batch['text_tokens'],
            batch['attention_mask']
        )
        
        if mode == 'train':
            return self._train_forward(batch, text_features, gt_latent, pose_gt)
        else:
            return self._inference_forward(batch, text_features, num_inference_steps)
    
    def _train_forward(self, batch, text_features, gt_latent, pose_gt=None):
        """Training: Flow Matching loss với SSM prior và sync guidance"""
        device = text_features.device
        B, T, D = gt_latent.shape
        
        # Sample timesteps
        t = self.scheduler.sample_timesteps(B, device)  # [B]
        
        # Mask
        seq_lengths = batch.get('seq_lengths', torch.full((B,), T, device=device))
        mask = torch.arange(T, device=device)[None, :] < seq_lengths[:, None]  # [B, T]
        
        # Add noise to latent
        latent_t, v_gt, latent_0 = self.scheduler.add_noise(gt_latent, t)
        
        # --- Bật grad cho latent_t để tính guidance ---
        latent_t_grad = latent_t.detach().requires_grad_(True)

        # --- Tính v_flow (phải chạy trên latent_t_grad) ---
        v_flow = self.flow_block(latent_t_grad, t, text_features, mask=mask)  # [B, T, 256]
        
        # --- Tính v_prior (không cần grad) ---
        v_prior = None
        if self.use_ssm_prior and self.ssm_prior is not None:
            with torch.no_grad():
                v_prior = self.ssm_prior(latent_t.detach(), t, mask=mask)  # [B, T, 256]
        
        # --- Combine velocities (chưa có guidance) ---
        lambda_t = self.get_lambda_t(t)  # [B, 1, 1]
        
        if v_prior is not None:
            v_pred_no_guidance = v_flow + lambda_t * v_prior
        else:
            v_pred_no_guidance = v_flow
        
        v_pred = v_pred_no_guidance # Mặc định
        sync_loss_train = torch.tensor(0.0, device=device)

        # --- Tính SyncHead Loss & Guidance (nếu có) ---
        if self.use_sync_guidance and self.sync_head is not None and pose_gt is not None:
            
            # 1. Tính Sync Score (dùng cho cả loss và guidance)
            # (Phải chạy trên latent_t_grad để có grad path)
            sync_score_pred = self.sync_head(latent_t_grad, mask) # [B, T]

            # 2. Tính Sync Loss (để train SyncHead)
            with torch.no_grad():
                # (compute_loss dùng pose_gt, không liên quan đến latent_t_grad)
                sync_loss_target = self.sync_head.compute_loss(latent_t.detach(), pose_gt, mask) # [B]
            
            # (Loss là MSE giữa score dự đoán (lấy trung bình T) và score thật (target là -corr))
            # Cần mask_sum để tính mean cho đúng
            masked_score_pred = sync_score_pred * mask.float()
            
            # (SỬA LỖI 2: Tính loss per-sample [B] rồi .mean() -> scalar)
            per_frame_sum = masked_score_pred.sum(dim=1)                    # [B]
            frame_counts = mask.sum(dim=1).clamp(min=1).float()             # [B]
            pred_mean = per_frame_sum / frame_counts                         # [B]
            # (sync_loss_target đã là [B] từ file sync_guidance.py đã sửa)
            
            sync_loss_train = self.sync_loss_fn(pred_mean, sync_loss_target.detach())
            
            # 3. Tính Guidance Gradient
            # (SỬA LỖI 2: Tính guidance loss per-sample rồi .mean() -> scalar)
            per_sample_score = masked_score_pred.sum(dim=1) / mask.sum(dim=1).clamp(min=1) # [B]
            guidance_loss = - per_sample_score.mean()                                 # scalar
            
            # (Tính grad của guidance loss WRT latent_t_grad)
            sync_grad = torch.autograd.grad(
                guidance_loss, 
                latent_t_grad, 
                # (BỎ grad_outputs vì loss đã là scalar)
                create_graph=False,
                retain_graph=True # (GIỮ LẠI để fix lỗi backward() 2 lần)
            )[0]

            # 4. Áp dụng Guidance
            v_pred = v_pred_no_guidance - self.gamma_guidance * sync_grad.detach()

        
        # --- Tính Loss cuối cùng ---
        
        # 1. Flow matching loss (dùng v_pred ĐÃ ĐƯỢC GUIDE)
        flow_loss = self.flow_loss_fn(v_pred, v_gt, mask=mask)
        
        # 2. Prior regularization loss (ép v_flow giống v_prior)
        prior_loss = torch.tensor(0.0, device=device)
        if v_prior is not None:
            # (Dùng v_flow, KHÔNG phải v_pred, để regularize)
            prior_loss = self.prior_reg_fn(v_flow, v_prior.detach())
        
        # 3. Total Loss
        total_loss = flow_loss + self.W_PRIOR * prior_loss + self.W_SYNC * sync_loss_train
        
        return {
            'total': total_loss,
            'flow': flow_loss,
            'prior': prior_loss,
            'sync': sync_loss_train # (Giờ là loss thật, không phải giá trị monitor)
        }
    
    # (SỬA LỖI 1: Bỏ decorator @torch.no_grad() khỏi hàm)
    def _inference_forward(self, batch, text_features, num_steps=50):
        """Inference: Flow from noise to latent"""
        device = text_features.device
        B = text_features.shape[0]
        T = batch.get('target_length', 50)
        if isinstance(T, torch.Tensor):
            T = T.max().item()
        
        # Start from noise
        latent = torch.randn(B, T, self.latent_dim, device=device)  # [B, T, 256]
        
        # Create mask
        seq_lengths = batch.get('seq_lengths', torch.full((B,), T, device=device))
        mask = torch.arange(T, device=device)[None, :] < seq_lengths[:, None]  # [B, T]
        
        # ODE integration (Euler method)
        dt = 1.0 / num_steps
        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((B,), t_val, device=device)  # [B]
            
            # --- Logic tính v_pred (giống train) ---
            
            # (Bật grad cho latent để tính guidance)
            latent_grad = latent.detach().requires_grad_(True)

            v_flow = self.flow_block(latent_grad, t, text_features, mask=mask)
            
            v_prior = None
            if self.use_ssm_prior and self.ssm_prior is not None:
                # (SỬA LỖI 1: Tắt grad cho prior)
                with torch.no_grad():
                    v_prior = self.ssm_prior(latent_grad.detach(), t, mask=mask)
            
            lambda_t = self.get_lambda_t(t)
            if v_prior is not None:
                v_pred_no_guidance = v_flow + lambda_t * v_prior
            else:
                v_pred_no_guidance = v_flow
            
            v_pred = v_pred_no_guidance
            
            # --- Sync guidance (chỉ ở middle steps) ---
            if self.use_sync_guidance and self.sync_head is not None and 0.2 < t_val < 0.8:
                
                # Tính score dự đoán
                sync_score_pred = self.sync_head(latent_grad, mask)
                masked_score_pred = sync_score_pred * mask.float()
                
                # (SỬA LỖI 2: Tính loss per-sample rồi .mean() -> scalar)
                per_sample_score = masked_score_pred.sum(dim=1) / mask.sum(dim=1).clamp(min=1) # [B]
                guidance_loss = - per_sample_score.mean()                                 # scalar
                
                # Lấy grad
                sync_grad = torch.autograd.grad(
                    guidance_loss, 
                    latent_grad, 
                    # (BỎ grad_outputs)
                    create_graph=False # (Thêm create_graph=False cho an toàn)
                )[0]
                
                # Áp dụng guidance
                v_pred = v_pred_no_guidance - self.gamma_guidance * sync_grad.detach()
            
            # (SỬA LỖI 1: Tắt grad cho bước Euler)
            with torch.no_grad():
                # Euler step
                latent = latent + dt * v_pred
                
                # Apply mask
                latent = latent * mask.unsqueeze(-1).float()
        
        return latent  # [B, T, 256]