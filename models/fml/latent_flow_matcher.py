import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel

# ==============================================================================
# ROBUST IMPORT
# ==============================================================================
try:
    from .flow_matching import FlowMatchingBlock, FlowMatchingScheduler, FlowMatchingLoss
    from .mamba_prior import SimpleSSMPrior
    from .sync_guidance import SyncGuidanceHead
except ImportError:
    try:
        from models.fml.flow_matching import FlowMatchingBlock, FlowMatchingScheduler, FlowMatchingLoss
    except ImportError:
        print("⚠️ Warning: Using dummy FlowMatching classes")
        class FlowMatchingBlock(nn.Module):
            def __init__(self, *args, **kwargs): super().__init__()
            def forward(self, x, *args, **kwargs): return x
        
        class FlowMatchingScheduler:
            def sample_timesteps(self, b, device): return torch.rand(b, device=device)
            def add_noise(self, x, t): return x, torch.zeros_like(x), torch.zeros_like(x)
        
        class FlowMatchingLoss(nn.Module):
            def forward(self, *args, **kwargs): return torch.tensor(0.0)
    
    class SimpleSSMPrior(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x, *args, **kwargs): return None
    
    class SyncGuidanceHead(nn.Module):
        def __init__(self, *args, **kwargs): super().__init__()
        def forward(self, x, *args, **kwargs): return None

# ==============================================================================
# LENGTH PREDICTOR
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
    
    def forward(self, text_features, padding_mask=None):
        """
        Args:
            text_features: [B, L, D]
            padding_mask: [B, L] where True=Padding, False=Valid
        """
        if padding_mask is not None:
            valid_mask = (~padding_mask).float().unsqueeze(-1)  # True → Valid
            pooled = (text_features * valid_mask).sum(1) / valid_mask.sum(1).clamp(min=1)
        else:
            pooled = text_features.mean(1)
        return self.net(pooled).squeeze(-1)

# ==============================================================================
# MAIN MODEL (FIXED VERSION)
# ==============================================================================
class LatentFlowMatcher(nn.Module):
    def __init__(self, latent_dim=256, text_encoder_name='bert-base-multilingual-cased',
                 hidden_dim=512, num_flow_layers=6, num_prior_layers=4, num_heads=8,
                 dropout=0.1, use_ssm_prior=True, use_sync_guidance=True,
                 lambda_prior=0.1, gamma_guidance=0.01, lambda_anneal=True,
                 W_PRIOR=0.05, W_SYNC=0.1, W_LENGTH=0.1,
                 sync_margin=0.2, clamp_scale=8.0):  # ✅ NEW: Configurable params
        super().__init__()
        
        self.latent_dim = latent_dim
        self.use_ssm_prior = use_ssm_prior
        self.use_sync_guidance = use_sync_guidance
        self.lambda_prior = lambda_prior
        self.gamma_guidance = gamma_guidance
        self.lambda_anneal = lambda_anneal
        
        # Loss weights
        self.W_PRIOR = W_PRIOR
        self.W_SYNC = W_SYNC
        self.W_LENGTH = W_LENGTH
        
        # ✅ FIX 1: Configurable hyperparameters
        self.sync_margin = sync_margin  # Margin for contrastive loss
        self.clamp_scale = clamp_scale  # Latent clamping range
        
        # Text Encoder (frozen)
        print(f"Loading Text Encoder: {text_encoder_name}...")
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        for param in self.text_encoder.parameters():
            param.requires_grad = False
        
        text_dim = self.text_encoder.config.hidden_size
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        # Length Predictor
        self.length_predictor = LengthPredictor(hidden_dim)
        self.length_loss_fn = nn.SmoothL1Loss()
        
        # Flow Matching
        self.scheduler = FlowMatchingScheduler()
        self.flow_block = FlowMatchingBlock(
            data_dim=latent_dim,
            condition_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Optional modules
        self.ssm_prior = None
        if self.use_ssm_prior:
            try:
                self.ssm_prior = SimpleSSMPrior(latent_dim, hidden_dim, num_prior_layers, num_heads, dropout)
            except (NameError, TypeError):
                print("⚠️ SSM Prior not available")
                self.use_ssm_prior = False
        
        self.sync_head = None
        if self.use_sync_guidance:
            try:
                self.sync_head = SyncGuidanceHead(latent_dim, hidden_dim // 2, dropout)
            except (NameError, TypeError):
                print("⚠️ Sync Guidance not available")
                self.use_sync_guidance = False
        
        self.flow_loss_fn = FlowMatchingLoss()
    
    def encode_text(self, text_tokens, attention_mask):
        """
        Args:
            attention_mask: BERT format (1=Valid, 0=Padding)
        Returns:
            text_features: [B, L, H]
            padding_mask: [B, L] (True=Padding, False=Valid) - PyTorch convention
        """
        self.text_encoder.eval()
        with torch.no_grad():
            outputs = self.text_encoder(text_tokens, attention_mask=attention_mask)
        
        text_features = self.text_proj(outputs.last_hidden_state)
        # Convert BERT mask (1=Valid) → PyTorch mask (True=Padding)
        padding_mask = ~attention_mask.bool()
        
        return text_features, padding_mask
    
    def get_lambda_t(self, t):
        """Annealing schedule for prior weight"""
        if self.lambda_anneal:
            return self.lambda_prior * (1 - t.float().view(-1, 1, 1))
        return self.lambda_prior
    
    def forward(self, batch, gt_latent=None, pose_gt=None, mode='train',
                num_inference_steps=50, latent_scale=1.0, return_attn_weights=False):
        
        text_features, text_padding_mask = self.encode_text(
            batch['text_tokens'], batch['attention_mask']
        )
        
        if mode == 'train':
            return self._train_forward(
                batch, text_features, text_padding_mask, gt_latent, pose_gt, return_attn_weights
            )
        else:
            return self._inference_forward(
                batch, text_features, text_padding_mask, num_inference_steps, latent_scale
            )
    
    def _train_forward(self, batch, text_features, text_padding_mask, gt_latent, pose_gt, return_attn_weights):
        device = text_features.device
        B, T, D = gt_latent.shape
        
        # 1. Length Prediction
        pred_length = self.length_predictor(text_features, text_padding_mask)
        target_length = batch['seq_lengths'].float().to(device)
        length_loss = self.length_loss_fn(pred_length, target_length)
        
        # 2. Flow Matching Setup
        t = self.scheduler.sample_timesteps(B, device)
        
        # Pose mask: True=Valid, False=Padding
        seq_valid_mask = torch.arange(T, device=device)[None, :] < batch['seq_lengths'][:, None]
        pose_padding_mask = ~seq_valid_mask  # Convert to PyTorch convention
        
        # Add noise (discard x0 to save memory)
        latent_t, v_gt, _ = self.scheduler.add_noise(gt_latent, t)
        
        # 3. Predict velocity
        flow_output = self.flow_block(
            latent_t, t, text_features,
            text_attn_mask=text_padding_mask,
            pose_attn_mask=pose_padding_mask,
            return_attn=return_attn_weights
        )
        
        v_flow, attn_weights = (flow_output if return_attn_weights else (flow_output, None))
        
        # 4. Prior integration
        v_prior = None
        if self.use_ssm_prior and self.ssm_prior is not None:
            prior_out = self.ssm_prior(
                latent_t.detach(), t, text_features, mask=seq_valid_mask
            )
            if prior_out is not None:
                v_prior = prior_out
        
        lambda_t = self.get_lambda_t(t)
        v_pred = v_flow + lambda_t * v_prior if v_prior is not None else v_flow
        
        # 5. Flow loss (main objective)
        flow_loss = self.flow_loss_fn(v_pred, v_gt, mask=seq_valid_mask)
        
        # ✅ FIX 2: IMPROVED Prior Regularization Loss
        # Purpose: Encourage flow to align with prior (regularization)
        prior_loss = torch.tensor(0.0, device=device)
        if v_prior is not None and self.W_PRIOR > 0:
            # MSE between flow prediction and prior prediction
            # Detach prior to avoid backprop through SSM (SSM trains separately if needed)
            prior_diff = F.mse_loss(v_flow, v_prior.detach(), reduction='none')
            prior_diff = prior_diff * seq_valid_mask.unsqueeze(-1).float()
            prior_loss = prior_diff.sum() / (seq_valid_mask.sum() * D).clamp(min=1)
        
        # ✅ FIX 3: IMPROVED Sync Guidance Loss with Time-Shift Negatives
        sync_loss = torch.tensor(0.0, device=device)
        if self.use_sync_guidance and self.sync_head is not None and self.W_SYNC > 0:
            # Positive: current latent
            sync_pos = self.sync_head(latent_t, seq_valid_mask)
            
            if sync_pos is not None:
                # ✅ SAFER Negative: Time-shifted latent (better than permutation)
                with torch.no_grad():
                    # Random shift between 1 and T//4
                    max_shift = max(1, T // 4)
                    shifts = torch.randint(1, max_shift + 1, (B,), device=device)
                    
                    # Apply time-shift to each sample
                    latent_neg = torch.stack([
                        torch.roll(latent_t[i], shifts=shifts[i].item(), dims=0)
                        for i in range(B)
                    ])
                    
                    # Mask stays the same (temporal structure matters)
                    mask_neg = seq_valid_mask
                
                sync_neg = self.sync_head(latent_neg, mask_neg)
                
                if sync_neg is not None:
                    # Contrastive loss: encourage positive > negative + margin
                    def masked_mean(val, m):
                        return (val * m.float()).sum() / m.sum().clamp(min=1)
                    
                    pos_score = masked_mean(sync_pos.sum(dim=-1), seq_valid_mask)
                    neg_score = masked_mean(sync_neg.sum(dim=-1), mask_neg)
                    
                    # Hinge loss with configurable margin
                    sync_loss = F.relu(self.sync_margin - pos_score + neg_score)
        
        # 8. Total loss
        total_loss = (
            flow_loss +
            self.W_PRIOR * prior_loss +
            self.W_SYNC * sync_loss +
            self.W_LENGTH * length_loss
        )
        
        result = {
            'total': total_loss,
            'flow': flow_loss,
            'prior': prior_loss,
            'sync': sync_loss,
            'length': length_loss
        }
        
        if return_attn_weights:
            result['attn_weights'] = attn_weights
        
        return result
    
    def _inference_forward(self, batch, text_features, text_padding_mask, num_steps, latent_scale):
        device = text_features.device
        B = text_features.shape[0]
        
        # Predict sequence length
        if batch is not None and 'seq_lengths' in batch:
            target_lengths = batch['seq_lengths'].long()
        else:
            with torch.no_grad():
                pred_len = self.length_predictor(text_features, text_padding_mask)
                target_lengths = pred_len.round().long().clamp(min=20, max=400)
        
        T = target_lengths.max().item()
        
        # Create masks
        seq_valid_mask = torch.arange(T, device=device)[None, :] < target_lengths[:, None]
        pose_padding_mask = ~seq_valid_mask
        
        # Initialize latent
        latent = torch.randn(B, T, self.latent_dim, device=device)
        dt = 1.0 / num_steps
        
        # ✅ FIX 4: Dynamic clamping based on latent_scale
        clamp_min = -self.clamp_scale * latent_scale
        clamp_max = self.clamp_scale * latent_scale
        
        # ODE solve
        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((B,), t_val, device=device)
            
            latent_in = latent.detach()
            
            # Enable gradient only when needed for guidance
            use_guidance = (
                self.use_sync_guidance and 
                self.sync_head is not None and 
                0.1 < t_val < 0.9
            )
            
            if use_guidance:
                latent_in.requires_grad_(True)
            
            # Flow prediction
            v_flow = self.flow_block(
                latent_in, t, text_features,
                text_attn_mask=text_padding_mask,
                pose_attn_mask=pose_padding_mask
            )
            
            # Prior prediction
            v_prior = None
            if self.use_ssm_prior and self.ssm_prior is not None:
                with torch.no_grad():
                    prior_out = self.ssm_prior(latent_in, t, text_features, mask=seq_valid_mask)
                    if prior_out is not None:
                        v_prior = prior_out
            
            # Combine flow and prior
            lambda_t = self.get_lambda_t(t)
            v_final = v_flow + lambda_t * v_prior if v_prior is not None else v_flow
            
            # Sync guidance with gradient
            if use_guidance:
                sync_out = self.sync_head(latent_in, seq_valid_mask)
                if sync_out is not None:
                    with torch.enable_grad():
                        guidance_score = sync_out.mean()
                        sync_grad = torch.autograd.grad(guidance_score, latent_in)[0]
                    v_final = v_final + self.gamma_guidance * sync_grad.detach()
            
            # Update latent
            with torch.no_grad():
                latent = latent + v_final.detach() * dt
                # Mask padding positions
                latent = latent * seq_valid_mask.unsqueeze(-1).float()
                # ✅ FIX 4: Dynamic clamping
                latent = torch.clamp(latent, clamp_min, clamp_max)
        
        return latent * latent_scale