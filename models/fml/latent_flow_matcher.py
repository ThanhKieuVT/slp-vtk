import torch
import torch.nn as nn
from transformers import BertModel

from .flow_matching import FlowMatchingBlock, FlowMatchingScheduler, FlowMatchingLoss
from .mamba_prior import SimpleSSMPrior
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
        lambda_prior=0.1,
        gamma_guidance=0.01,
        lambda_anneal=True,
        W_PRIOR=0.01,
        W_SYNC=0.1
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
        text_dim = self.text_encoder.config.hidden_size
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
        
        # SSM Prior
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
        
        # Sync Guidance
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
        self.sync_loss_fn = nn.MSELoss()
    
    def encode_text(self, text_tokens, attention_mask):
        outputs = self.text_encoder(text_tokens, attention_mask=attention_mask)
        text_features = self.text_proj(outputs.last_hidden_state)
        return text_features
    
    def get_lambda_t(self, t):
        t_float = t.float()
        if self.lambda_anneal:
            return self.lambda_prior * (1 - t_float.view(-1, 1, 1))
        else:
            return torch.full((t.shape[0],1,1), self.lambda_prior, device=t.device, dtype=t_float.dtype)
    
    def forward(self, batch, gt_latent, pose_gt=None, mode='train', num_inference_steps=50):
        text_features = self.encode_text(batch['text_tokens'], batch['attention_mask'])
        if mode == 'train':
            return self._train_forward(batch, text_features, gt_latent, pose_gt)
        else:
            return self._inference_forward(batch, text_features, num_inference_steps)
    
    def _train_forward(self, batch, text_features, gt_latent, pose_gt=None):
        device = text_features.device
        B, T, D = gt_latent.shape
        
        # Sample timesteps
        t = self.scheduler.sample_timesteps(B, device)
        
        # Mask
        seq_lengths = batch.get('seq_lengths', torch.full((B,), T, device=device))
        mask = torch.arange(T, device=device)[None, :] < seq_lengths[:, None]
        
        # Add noise
        latent_t, v_gt, latent_0 = self.scheduler.add_noise(gt_latent, t)
        
        # Bật grad cho latent_t
        latent_t_grad = latent_t.detach().requires_grad_(True)
        
        # Tính v_flow
        v_flow = self.flow_block(latent_t_grad, t, text_features, mask=mask)
        
        # Tính v_prior
        v_prior = None
        if self.use_ssm_prior and self.ssm_prior is not None:
            with torch.no_grad():
                v_prior = self.ssm_prior(latent_t.detach(), t, mask=mask)
        
        lambda_t = self.get_lambda_t(t)
        if v_prior is not None:
            v_pred_no_guidance = v_flow + lambda_t * v_prior
        else:
            v_pred_no_guidance = v_flow
        
        v_pred = v_pred_no_guidance
        sync_loss_train = torch.tensor(0.0, device=device)
        
        # --- Sync Guidance ---
        if self.use_sync_guidance and self.sync_head is not None and pose_gt is not None:
            # Predict score
            sync_score_pred = self.sync_head(latent_t_grad, mask)
            
            # Compute target loss (không grad)
            with torch.no_grad():
                sync_loss_target = self.sync_head.compute_loss(latent_t.detach(), pose_gt, mask)
            
            masked_score_pred = sync_score_pred * mask.float()
            per_frame_sum = masked_score_pred.sum(dim=1)
            frame_counts = mask.sum(dim=1).clamp(min=1).float()
            pred_mean = per_frame_sum / frame_counts
            
            sync_loss_train = self.sync_loss_fn(pred_mean, sync_loss_target.detach())
            
            # Compute guidance gradient
            per_sample_score = masked_score_pred.sum(dim=1) / frame_counts
            guidance_loss = - per_sample_score.mean()
            
            sync_grad = torch.autograd.grad(
                guidance_loss,
                latent_t_grad,
                create_graph=False,
                retain_graph=True
            )[0]
            
            # Apply guidance
            v_pred = v_pred_no_guidance - self.gamma_guidance * sync_grad.detach()
        
        # --- Losses ---
        flow_loss = self.flow_loss_fn(v_pred, v_gt, mask=mask)
        prior_loss = torch.tensor(0.0, device=device)
        if v_prior is not None:
            prior_loss = self.prior_reg_fn(v_flow, v_prior.detach())
        
        total_loss = flow_loss + self.W_PRIOR * prior_loss + self.W_SYNC * sync_loss_train
        
        return {
            'total': total_loss,
            'flow': flow_loss,
            'prior': prior_loss,
            'sync': sync_loss_train
        }
    
    def _inference_forward(self, batch, text_features, num_steps=50):
        device = text_features.device
        B = text_features.shape[0]
        T = batch.get('target_length', 50)
        if isinstance(T, torch.Tensor):
            T = T.max().item()
        
        latent = torch.randn(B, T, self.latent_dim, device=device)
        seq_lengths = batch.get('seq_lengths', torch.full((B,), T, device=device))
        mask = torch.arange(T, device=device)[None, :] < seq_lengths[:, None]
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((B,), t_val, device=device)
            
            latent_grad = latent.detach().requires_grad_(True)
            v_flow = self.flow_block(latent_grad, t, text_features, mask=mask)
            
            v_prior = None
            if self.use_ssm_prior and self.ssm_prior is not None:
                with torch.no_grad():
                    v_prior = self.ssm_prior(latent_grad.detach(), t, mask=mask)
            
            lambda_t = self.get_lambda_t(t)
            if v_prior is not None:
                v_pred_no_guidance = v_flow + lambda_t * v_prior
            else:
                v_pred_no_guidance = v_flow
            
            v_pred = v_pred_no_guidance
            
            # Sync guidance (inference)
            if self.use_sync_guidance and self.sync_head is not None and 0.2 < t_val < 0.8:
                sync_score_pred = self.sync_head(latent_grad, mask)
                masked_score_pred = sync_score_pred * mask.float()
                per_sample_score = masked_score_pred.sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                guidance_loss = - per_sample_score.mean()
                
                sync_grad = torch.autograd.grad(
                    guidance_loss,
                    latent_grad,
                    create_graph=False
                )[0]
                v_pred = v_pred_no_guidance - self.gamma_guidance * sync_grad.detach()
            
            with torch.no_grad():
                latent = latent + dt * v_pred
                latent = latent * mask.unsqueeze(-1).float()
        
        return latent
