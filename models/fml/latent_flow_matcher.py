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
        lambda_anneal=True  # Anneal lambda theo t
    ):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_ssm_prior = use_ssm_prior
        self.use_sync_guidance = use_sync_guidance
        self.lambda_prior = lambda_prior
        self.gamma_guidance = gamma_guidance
        self.lambda_anneal = lambda_anneal
        
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
        if self.lambda_anneal:
            return self.lambda_prior * (1 - t.unsqueeze(-1).unsqueeze(-1))  # [B, 1, 1]
        else:
            return self.lambda_prior
    
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
        
        # Predict flow velocity
        v_flow = self.flow_block(latent_t, t, text_features, mask=mask)  # [B, T, 256]
        
        # SSM Prior velocity
        v_prior = None
        if self.use_ssm_prior and self.ssm_prior is not None:
            v_prior = self.ssm_prior(latent_t, t, mask=mask)  # [B, T, 256]
        
        # Combine velocities
        lambda_t = self.get_lambda_t(t)  # [B, 1, 1]
        
        if v_prior is not None:
            v_pred = v_flow + lambda_t * v_prior
        else:
            v_pred = v_flow
        
        # Sync guidance gradient (nếu có)
        if self.use_sync_guidance and self.sync_head is not None and pose_gt is not None:
            # Compute gradient
            with torch.enable_grad():
                latent_t_grad = latent_t.detach().requires_grad_(True)
                sync_loss = self.sync_head.compute_loss(latent_t_grad, pose_gt, mask)
                sync_grad = torch.autograd.grad(sync_loss, latent_t_grad, create_graph=False)[0]
            
            # Apply guidance: v = v - γ * ∇_z L_sync
            v_pred = v_pred - self.gamma_guidance * sync_grad
        
        # Flow matching loss
        flow_loss = self.flow_loss_fn(v_pred, v_gt, mask=mask)
        
        # Prior regularization loss
        prior_loss = torch.tensor(0.0, device=device)
        if v_prior is not None:
            prior_loss = self.prior_reg_fn(v_flow, v_prior) * 0.01  # Small weight
        
        # Sync loss (for monitoring)
        sync_loss_val = torch.tensor(0.0, device=device)
        if self.use_sync_guidance and self.sync_head is not None and pose_gt is not None:
            with torch.no_grad():
                sync_loss_val = self.sync_head.compute_loss(latent_t, pose_gt, mask)
        
        total_loss = flow_loss + prior_loss
        
        return {
            'total': total_loss,
            'flow': flow_loss,
            'prior': prior_loss,
            'sync': sync_loss_val
        }
    
    @torch.no_grad()
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
            
            # Predict velocity
            v_flow = self.flow_block(latent, t, text_features, mask=mask)
            
            # SSM Prior
            v_prior = None
            if self.use_ssm_prior and self.ssm_prior is not None:
                v_prior = self.ssm_prior(latent, t, mask=mask)
            
            # Combine
            lambda_t = self.get_lambda_t(t)
            if v_prior is not None:
                v = v_flow + lambda_t * v_prior
            else:
                v = v_flow
            
            # Sync guidance (nếu có, chỉ ở middle steps)
            if self.use_sync_guidance and self.sync_head is not None and 0.2 < t_val < 0.8:
                # Decode để compute sync (cần decoder, sẽ pass từ outside)
                # Tạm thời skip trong inference đơn giản
                pass
            
            # Euler step
            latent = latent + dt * v
            
            # Apply mask
            latent = latent * mask.unsqueeze(-1).float()
        
        return latent  # [B, T, 256]

