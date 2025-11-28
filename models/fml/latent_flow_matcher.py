import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# ==============================================================================
# 1. IMPORT HANDLING & DUMMY CLASSES (Prevent Crash)
# ==============================================================================
try:
    # Ưu tiên import local
    from .flow_matching import FlowMatchingBlock, FlowMatchingScheduler, FlowMatchingLoss
    from .mamba_prior import SimpleSSMPrior
    from .sync_guidance import SyncGuidanceHead
except ImportError:
    # Fallback paths (nếu chạy từ root)
    try:
        from models.fml.flow_matching import FlowMatchingBlock, FlowMatchingScheduler, FlowMatchingLoss
    except ImportError:
        print("⚠️ Warning: Could not import FlowMatching modules. Defining dummies.")
        class FlowMatchingBlock(nn.Module):
            def __init__(self, *args, **kwargs): super().__init__()
            def forward(self, x, *args, **kwargs): return x

        class FlowMatchingScheduler:
            def sample_timesteps(self, b, device): return torch.rand(b, device=device)
            def add_noise(self, x, t): return x, torch.zeros_like(x), torch.zeros_like(x)

        class FlowMatchingLoss(nn.Module):
            def forward(self, *args, **kwargs): return torch.tensor(0.0)

    # Dummy cho các module Optional (Mamba, SyncGuidance)
    class SimpleSSMPrior(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x, *args, **kwargs): return None 

    # --- FIX QUAN TRỌNG Ở ĐÂY ---
    class SyncGuidanceHead(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x, *args, **kwargs):
            # x shape: [Batch, Time, Dim]
            # Trả về [Batch, Time, 1] để khớp dimension khi tính toán
            return torch.zeros(x.shape[0], x.shape[1], 1, device=x.device)

# ==============================================================================
# 2. HELPER MODULES
# ==============================================================================
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
            keep_mask = mask.float().unsqueeze(-1) if mask.dtype != torch.float else mask.unsqueeze(-1)
            pooled_text = (text_features * keep_mask).sum(dim=1) / keep_mask.sum(dim=1).clamp(min=1)
        else:
            pooled_text = text_features.mean(dim=1)
        return self.net(pooled_text).squeeze(-1)

# ==============================================================================
# 3. MAIN MODEL CLASS
# ==============================================================================
class LatentFlowMatcher(nn.Module):
    def __init__(self, latent_dim=256, text_encoder_name='bert-base-multilingual-cased', 
                 hidden_dim=512, num_flow_layers=6, num_prior_layers=4, num_heads=8, 
                 dropout=0.1, use_ssm_prior=True, use_sync_guidance=True, 
                 lambda_prior=0.1, gamma_guidance=0.01, lambda_anneal=True,
                 W_PRIOR=0.0, W_SYNC=0.1, W_LENGTH=0.01):
        super().__init__()
        self.latent_dim = latent_dim
        self.use_ssm_prior = use_ssm_prior
        self.use_sync_guidance = use_sync_guidance
        self.lambda_prior = lambda_prior
        self.gamma_guidance = gamma_guidance
        self.lambda_anneal = lambda_anneal
        
        # Loss Weights
        self.W_PRIOR = W_PRIOR
        self.W_SYNC = W_SYNC
        self.W_LENGTH = W_LENGTH
        
        print(f"Loading Text Encoder: {text_encoder_name}...")
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
            
        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        self.length_predictor = LengthPredictor(input_dim=hidden_dim)
        self.length_loss_fn = nn.SmoothL1Loss()
        
        # Init scheduler & Main Flow Block
        self.scheduler = FlowMatchingScheduler()
        self.flow_block = FlowMatchingBlock(
            data_dim=latent_dim, condition_dim=hidden_dim, hidden_dim=hidden_dim,
            num_layers=num_flow_layers, num_heads=num_heads, dropout=dropout
        )
        
        # Optional Modules (Safe Init)
        self.ssm_prior = None
        if self.use_ssm_prior:
            try:
                self.ssm_prior = SimpleSSMPrior(latent_dim, hidden_dim, num_prior_layers, num_heads, dropout)
            except (NameError, TypeError):
                print("⚠️ Cannot init SSM Prior. Disabling.")
                self.use_ssm_prior = False

        self.sync_head = None
        if self.use_sync_guidance:
            try:
                self.sync_head = SyncGuidanceHead(latent_dim, hidden_dim // 2, dropout)
            except (NameError, TypeError):
                print("⚠️ Cannot init SyncGuidance. Disabling.")
                self.use_sync_guidance = False
        
        self.flow_loss_fn = FlowMatchingLoss()
        self.prior_reg_fn = nn.MSELoss()
    
    def encode_text(self, text_tokens, attention_mask):
        self.text_encoder.eval()
        with torch.no_grad():
            outputs = self.text_encoder(text_tokens, attention_mask=attention_mask)
        text_features = self.text_proj(outputs.last_hidden_state)
        text_valid_mask = attention_mask.bool() 
        return text_features, text_valid_mask
    
    def get_lambda_t(self, t):
        if self.lambda_anneal:
            return self.lambda_prior * (1 - t.float().view(-1, 1, 1))
        else:
            return self.lambda_prior
    
    def forward(self, batch, gt_latent=None, pose_gt=None, mode='train', num_inference_steps=50, latent_scale=1.0, return_attn_weights=False):
        # Common text encoding
        text_features, text_valid_mask = self.encode_text(batch['text_tokens'], batch['attention_mask'])
        
        if mode == 'train':
            return self._train_forward(batch, text_features, text_valid_mask, gt_latent, pose_gt, return_attn_weights)
        else:
            return self._inference_forward(batch, text_features, text_valid_mask, num_inference_steps, latent_scale)
    
    def _train_forward(self, batch, text_features, text_valid_mask, gt_latent, pose_gt=None, return_attn_weights=False):
        device = text_features.device
        B, T, D = gt_latent.shape
        
        # 1. Length Prediction Loss
        pred_length = self.length_predictor(text_features, text_valid_mask)
        target_length = batch['seq_lengths'].float().to(device)
        length_loss = self.length_loss_fn(pred_length, target_length)
        
        # 2. Flow Matching Setup
        t = self.scheduler.sample_timesteps(B, device)
        seq_mask = torch.arange(T, device=device)[None, :] < batch['seq_lengths'][:, None]
        
        # Add Noise: x0 (noise), x1 (data) -> xt
        latent_t, v_gt, latent_0 = self.scheduler.add_noise(gt_latent, t)
        pose_attn_mask = ~seq_mask 
        
        # 3. Predict Velocity (Vector Field)
        flow_output = self.flow_block(
            latent_t, t, text_features,
            text_attn_mask=text_valid_mask, 
            pose_attn_mask=pose_attn_mask,
            return_attn=return_attn_weights
        )
        
        if return_attn_weights: v_flow, attn_weights = flow_output
        else: v_flow, attn_weights = flow_output, None
        
        # 4. Prior Integration (Optional)
        v_prior = None
        if self.use_ssm_prior and self.ssm_prior is not None:
            prior_out = self.ssm_prior(latent_t.detach(), t, text_features, mask=seq_mask)
            if prior_out is not None:
                v_prior = prior_out
        
        lambda_t = self.get_lambda_t(t)
        v_pred_train = v_flow + lambda_t * v_prior if v_prior is not None else v_flow
        
        # 5. Losses Calculation
        flow_loss = self.flow_loss_fn(v_pred_train, v_gt, mask=seq_mask)
        prior_loss = torch.tensor(0.0, device=device)
        sync_loss = torch.tensor(0.0, device=device)
        
        # Sync Guidance Loss (Contrastive)
        if self.use_sync_guidance and self.sync_head is not None and self.W_SYNC > 0:
            sync_pos = self.sync_head(latent_t, seq_mask)
            
            # Kiểm tra nếu sync_pos trả về 0 do dummy class thì skip tính toán
            if sync_pos.sum() != 0 or sync_pos.shape[-1] > 1:
                with torch.no_grad():
                    idx_perm = torch.randperm(B, device=device)
                    latent_neg = latent_t[idx_perm]
                    mask_neg = seq_mask[idx_perm]
                sync_neg = self.sync_head(latent_neg, mask_neg)
                
                def masked_mean(val, m): 
                    # Val [B, T], Mask [B, T] -> OK
                    return (val * m.float()).sum() / m.sum().clamp(min=1)

                # sum(dim=-1) biến [B, T, 1] thành [B, T] -> OK
                pos_score = masked_mean(sync_pos.sum(dim=-1), seq_mask)
                neg_score = masked_mean(sync_neg.sum(dim=-1), mask_neg)
                
                sync_loss = F.relu(0.2 - pos_score + neg_score)

        total_loss = flow_loss + self.W_LENGTH * length_loss + self.W_SYNC * sync_loss + self.W_PRIOR * prior_loss

        result = {'total': total_loss, 'flow': flow_loss, 'length': length_loss, 'sync': sync_loss}
        if return_attn_weights: result['attn_weights'] = attn_weights
        return result
    
    def _inference_forward(self, batch, text_features, text_valid_mask, num_steps=50, latent_scale=1.0):
        device = text_features.device
        B = text_features.shape[0]
        
        if batch is not None and 'seq_lengths' in batch:
             target_lengths = batch['seq_lengths'].long()
        else:
             with torch.no_grad():
                pred_len = self.length_predictor(text_features, text_valid_mask)
                target_lengths = pred_len.round().long().clamp(min=20, max=400)
        
        T = target_lengths.max().item()
        seq_mask = torch.arange(T, device=device)[None, :] < target_lengths[:, None]
        
        latent = torch.randn(B, T, self.latent_dim, device=device) 
        
        dt = 1.0 / num_steps
        pose_attn_mask = ~seq_mask
        
        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((B,), t_val, device=device)
            
            latent_in = latent.detach()
            if self.use_sync_guidance:
                latent_in.requires_grad_(True)
            
            v_flow = self.flow_block(latent_in, t, text_features, 
                                     text_attn_mask=text_valid_mask, 
                                     pose_attn_mask=pose_attn_mask)
            
            v_prior = None
            if self.use_ssm_prior and self.ssm_prior is not None:
                with torch.no_grad():
                    prior_out = self.ssm_prior(latent_in, t, text_features, mask=seq_mask)
                    if prior_out is not None: v_prior = prior_out
            
            lambda_t = self.get_lambda_t(t)
            v_final = v_flow + lambda_t * v_prior if v_prior is not None else v_flow
            
            # Sync Guidance
            if self.use_sync_guidance and self.sync_head is not None and 0.1 < t_val < 0.9:
                sync_out = self.sync_head(latent_in, seq_mask)
                # Chỉ tính guidance nếu output khác 0
                if sync_out.sum() != 0:
                    with torch.enable_grad():
                        guidance_score = sync_out.mean()
                        sync_grad = torch.autograd.grad(guidance_score, latent_in)[0]
                    v_final = v_final + self.gamma_guidance * sync_grad.detach()
            
            with torch.no_grad():
                latent = latent + v_final.detach() * dt
                latent = latent * seq_mask.unsqueeze(-1).float()
                latent = torch.clamp(latent, -5.0, 5.0)

        return latent * latent_scale