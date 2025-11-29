import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

from .flow_matching import FlowMatchingBlock, FlowMatchingScheduler, FlowMatchingLoss
from .mamba_prior import SimpleSSMPrior
from .sync_guidance import SyncGuidanceHead 

class LengthPredictor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1) 
        )

    def forward(self, text_features, mask=None):
        if mask is not None:
            valid_mask = (~mask).float().unsqueeze(-1) 
            pooled_text = (text_features * valid_mask).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
        else:
            pooled_text = text_features.mean(dim=1) 
        pred_length = self.net(pooled_text)
        return pred_length.squeeze(-1) 

class LatentFlowMatcher(nn.Module):
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
        W_SYNC=0.1,
        W_LENGTH=1.0
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
        self.LENGTH_SCALE = 120.0 

        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, hidden_dim) 
        
        self.length_predictor = LengthPredictor(input_dim=hidden_dim)
        self.length_loss_fn = nn.SmoothL1Loss()
        
        self.scheduler = FlowMatchingScheduler()
        
        self.flow_block = FlowMatchingBlock(
            data_dim=latent_dim,
            condition_dim=hidden_dim, 
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        if use_ssm_prior:
            self.ssm_prior = SimpleSSMPrior(latent_dim, hidden_dim, num_prior_layers, num_heads, dropout)
        else:
            self.ssm_prior = None
        
        if use_sync_guidance:
            self.sync_head = SyncGuidanceHead(latent_dim, hidden_dim // 2, dropout)
        else:
            self.sync_head = None
        
        self.flow_loss_fn = FlowMatchingLoss()

    def encode_text(self, text_tokens, attention_mask):
        self.text_encoder.eval()
        with torch.no_grad():
            outputs = self.text_encoder(text_tokens, attention_mask=attention_mask)
        text_features = self.text_proj(outputs.last_hidden_state) 
        text_padding_mask = ~attention_mask.bool()
        return text_features, text_padding_mask
    
    def get_lambda_t(self, t):
        if self.lambda_anneal:
            return self.lambda_prior * (1 - t.float().view(-1, 1, 1))
        else:
            return torch.full((t.shape[0], 1, 1), self.lambda_prior, device=t.device)
    
    def forward(self, batch, gt_latent, pose_gt=None, mode='train', num_inference_steps=50, return_attn_weights=False, prior_scale=1.0):
        text_features, text_mask = self.encode_text(batch['text_tokens'], batch['attention_mask'])
        
        if mode == 'train':
            return self._train_forward(batch, text_features, text_mask, gt_latent, pose_gt, return_attn_weights, prior_scale)
        else:
            return self._inference_forward(batch, text_features, text_mask, num_inference_steps)
    
    def _train_forward(self, batch, text_features, text_mask, gt_latent, pose_gt=None, return_attn_weights=False, prior_scale=1.0):
        device = text_features.device
        B, T, D = gt_latent.shape
        
        # 1. LENGTH LOSS
        pred_length = self.length_predictor(text_features.detach(), text_mask)
        target_length = batch['seq_lengths'].float()
        length_loss = self.length_loss_fn(pred_length / self.LENGTH_SCALE, target_length / self.LENGTH_SCALE)
        
        # 2. SETUP FLOW
        t = self.scheduler.sample_timesteps(B, device)
        seq_lengths = batch['seq_lengths']
        valid_mask = torch.arange(T, device=device)[None, :] < seq_lengths[:, None]
        pose_padding_mask = ~valid_mask
        
        latent_t, v_gt, latent_0 = self.scheduler.add_noise(gt_latent, t)
        
        # 3. PRIOR (With Warmup)
        v_prior = None
        prior_loss = torch.tensor(0.0, device=device)
        
        if self.use_ssm_prior and self.ssm_prior is not None:
            v_prior = self.ssm_prior(latent_t, t, text_features, mask=valid_mask)
            prior_loss_raw = (v_prior - v_gt.detach()) ** 2
            prior_loss = (prior_loss_raw * valid_mask.unsqueeze(-1)).sum() / (valid_mask.sum() * D).clamp(min=1)
        
        # 4. FLOW MATCHING
        flow_output = self.flow_block(
            latent_t, t, text_features, 
            text_attn_mask=text_mask, 
            pose_attn_mask=pose_padding_mask, 
            return_attn=return_attn_weights
        )
        if return_attn_weights: v_flow, attn_weights = flow_output
        else: v_flow = flow_output
        
        # Residual Learning Logic
        lambda_t = self.get_lambda_t(t) * prior_scale 
        if v_prior is not None:
            v_target = v_gt - lambda_t * v_prior.detach()
            v_pred_train = v_flow + lambda_t * v_prior.detach()
        else:
            v_target = v_gt
            v_pred_train = v_flow
        
        # 5. SYNC GUIDANCE
        sync_loss_train = torch.tensor(0.0, device=device)
        
        if self.use_sync_guidance and self.sync_head is not None:
            text_valid_mask = (~text_mask).unsqueeze(-1).float()
            text_pooled = (text_features * text_valid_mask).sum(1) / text_valid_mask.sum(1).clamp(min=1)

            sync_score_pos = self.sync_head(latent_t, text_pooled, valid_mask)
            
            with torch.no_grad():
                # FIX: Thêm device cho randperm
                shuffled_idx = torch.randperm(B, device=device)
                shuffled_text = text_pooled[shuffled_idx]
            
            sync_score_neg = self.sync_head(latent_t, shuffled_text, valid_mask)
            
            score_pos = (sync_score_pos * valid_mask.float()).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
            score_neg = (sync_score_neg * valid_mask.float()).sum(dim=1) / valid_mask.sum(dim=1).clamp(min=1)
            
            margin = 0.5
            loss_per_sample = F.relu(margin + score_neg - score_pos)
            sync_loss_train = loss_per_sample.mean()
            
        # 6. TOTAL LOSS
        flow_loss = self.flow_loss_fn(v_flow, v_target, mask=valid_mask)        
        total_loss = flow_loss + self.W_PRIOR * prior_loss + self.W_SYNC * sync_loss_train + self.W_LENGTH * length_loss
        
        result = {
            'total': total_loss, 'flow': flow_loss, 
            'prior': prior_loss, 'sync': sync_loss_train, 
            'length': length_loss 
        }
        if return_attn_weights: result['attn_weights'] = attn_weights
        return result
    
    def _inference_forward(self, batch, text_features, text_mask, num_steps=50):
        device = text_features.device
        B = text_features.shape[0]
        
        with torch.no_grad():
            pred_len_norm = self.length_predictor(text_features, text_mask)
            pred_len = pred_len_norm * self.LENGTH_SCALE 
            target_lengths = pred_len.round().long().clamp(min=10, max=400)
            
        T = target_lengths.max().item()
        valid_mask = torch.arange(T, device=device)[None, :] < target_lengths[:, None]
        pose_padding_mask = ~valid_mask
        
        latent = torch.randn(B, T, self.latent_dim, device=device)
        dt = 1.0 / num_steps
        
        text_valid_mask = (~text_mask).unsqueeze(-1).float()
        text_pooled = (text_features * text_valid_mask).sum(1) / text_valid_mask.sum(1).clamp(min=1)
        
        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((B,), t_val, device=device)
            
            latent_grad = latent.detach().requires_grad_(True)
            
            v_flow = self.flow_block(latent_grad, t, text_features, 
                                   text_attn_mask=text_mask, 
                                   pose_attn_mask=pose_padding_mask)
            
            v_prior = None
            if self.use_ssm_prior and self.ssm_prior is not None:
                with torch.no_grad():
                    v_prior = self.ssm_prior(latent_grad.detach(), t, text_features, mask=valid_mask)
            
            lambda_t = self.get_lambda_t(t)
            v_pred = v_flow + lambda_t * v_prior if v_prior is not None else v_flow
            
            if self.use_sync_guidance and self.sync_head is not None and 0.2 < t_val < 0.8:
                sync_score_pred = self.sync_head(latent_grad, text_pooled, valid_mask)
                total_score = (sync_score_pred * valid_mask.float()).sum() / valid_mask.sum().clamp(min=1)
                
                sync_grad = torch.autograd.grad(total_score, latent_grad)[0]
                
                with torch.no_grad():
                    v_mag = v_pred.norm(dim=-1, keepdim=True).mean()
                    clamp_val = min(0.1, max(0.01, v_mag.item() * 0.1))
                
                sync_grad = torch.clamp(sync_grad, -clamp_val, clamp_val)
                
                guidance_force = self.gamma_guidance * sync_grad.detach()
                v_pred = v_pred + guidance_force * valid_mask.unsqueeze(-1).float()

            with torch.no_grad():
                latent = latent + dt * v_pred
                latent = latent * valid_mask.unsqueeze(-1).float()
                latent = latent.clamp(-10.0, 10.0)
        
        return latent