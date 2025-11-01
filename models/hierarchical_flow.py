# models/hierarchical_flow.py
"""
Hierarchical Flow Matching for Manual Signs
3 models: Coarse (Body) → Medium (Arms) → Fine (Hands)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertModel
from .flow_matching import FlowMatchingBlock, FlowMatchingScheduler, FlowMatchingLoss

class HierarchicalFlowMatcher(nn.Module):
    """
    Complete manual sign generation system
    
    3 Levels:
        Level 1: Coarse (Body) - 66D (33 keypoints × 2)
        Level 2: Medium (Arms) - 16D (8 keypoints × 2)
        Level 3: Fine (Hands) - 68D (34 keypoints × 2)
    
    Total: 150D (75 keypoints × 2)
    """
    def __init__(
        self,
        text_encoder_name='bert-base-german-cased',
        hidden_dim=512,
        num_flow_layers=6,
        num_heads=8,
        dropout=0.1,
        freeze_text_encoder=False
    ):
        super().__init__()
        
        # Text encoder (shared across all levels)
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        text_dim = self.text_encoder.config.hidden_size  # 768
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("Text encoder frozen")
        
        # Flow scheduler
        self.scheduler = FlowMatchingScheduler()
        
        # Level 1: Coarse Flow (Body trajectory)
        self.coarse_flow = FlowMatchingBlock(
            data_dim=66,
            condition_dim=text_dim,
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Level 2: Medium Flow (Arm movements)
        self.medium_flow = FlowMatchingBlock(
            data_dim=16,
            condition_dim=text_dim,  # Only text (no coarse dependency for now)
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Level 3: Fine Flow (Hand articulation)
        self.fine_flow = FlowMatchingBlock(
            data_dim=68,
            condition_dim=text_dim,  # Only text
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers + 2,  # More layers for detail
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Loss functions
        self.loss_fn = FlowMatchingLoss()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"HierarchicalFlowMatcher: {total_params/1e6:.1f}M trainable parameters")
    
    def encode_text(self, text_tokens, attention_mask):
        """
        Encode German text
        
        Args:
            text_tokens: [B, L] token IDs
            attention_mask: [B, L] mask
        
        Returns:
            [B, L, 768] text features
        """
        outputs = self.text_encoder(
            input_ids=text_tokens,
            attention_mask=attention_mask
        )
        return outputs.last_hidden_state
    
    def forward(self, batch, mode='train', num_inference_steps=20):
        """
        Forward pass
        
        Args:
            batch: dict with data
            mode: 'train' or 'inference'
            num_inference_steps: ODE integration steps (inference only)
        
        Returns:
            If train: dict with losses
            If inference: dict with generated poses
        """
        # Encode text
        text_features = self.encode_text(
            batch['text_tokens'],
            batch['attention_mask']
        )  # [B, L, 768]
        
        if mode == 'train':
            return self._train_forward(batch, text_features)
        else:
            return self._inference_forward(batch, text_features, num_inference_steps)
    
    def _train_forward(self, batch, text_features):
        """Training forward pass"""
        device = text_features.device
        B = text_features.shape[0]
        
        # Sample timesteps
        t = self.scheduler.sample_timesteps(B, device)
        
        # Create padding mask
        T_max = batch['manual_coarse'].shape[1]
        seq_lengths = batch['seq_lengths']
        mask = torch.arange(T_max, device=device)[None, :] >= seq_lengths[:, None]
        # mask: True = padding, False = valid
        
        losses = {}
        
        # --- Level 1: Coarse (Body) ---
        x_0_coarse = batch['manual_coarse']  # [B, T, 66]
        x_t_coarse, v_gt_coarse, _ = self.scheduler.add_noise(x_0_coarse, t)
        
        v_pred_coarse = self.coarse_flow(
            x_t_coarse,
            t,
            text_features,
            mask=mask
        )
        
        losses['coarse'] = self.loss_fn(v_pred_coarse, v_gt_coarse, ~mask)
        
        # --- Level 2: Medium (Arms) ---
        x_0_medium = batch['manual_medium']  # [B, T, 16]
        x_t_medium, v_gt_medium, _ = self.scheduler.add_noise(x_0_medium, t)
        
        v_pred_medium = self.medium_flow(
            x_t_medium,
            t,
            text_features,
            mask=mask
        )
        
        losses['medium'] = self.loss_fn(v_pred_medium, v_gt_medium, ~mask)
        
        # --- Level 3: Fine (Hands) ---
        x_0_fine = batch['manual_fine']  # [B, T, 68]
        x_t_fine, v_gt_fine, _ = self.scheduler.add_noise(x_0_fine, t)
        
        v_pred_fine = self.fine_flow(
            x_t_fine,
            t,
            text_features,
            mask=mask
        )
        
        losses['fine'] = self.loss_fn(v_pred_fine, v_gt_fine, ~mask)
        
        # Total loss (weighted)
        losses['total'] = (
            1.0 * losses['coarse'] +
            1.0 * losses['medium'] +
            1.5 * losses['fine']  # Hands are more important!
        )
        
        return losses
    
    @torch.no_grad()
    def _inference_forward(self, batch, text_features, num_steps=20):
        """
        Inference: Generate poses from noise via ODE integration
        
        Args:
            num_steps: Number of Euler integration steps
        
        Returns:
            Generated pose sequences
        """
        device = text_features.device
        B, L, _ = text_features.shape
        T = batch.get('target_length', 50)  # Target sequence length
        
        # Start from noise
        x_coarse = torch.randn(B, T, 66, device=device)
        x_medium = torch.randn(B, T, 16, device=device)
        x_fine = torch.randn(B, T, 68, device=device)
        
        # Integration
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((B,), t_val, device=device)
            
            # Generate coarse
            v_coarse = self.coarse_flow(x_coarse, t, text_features)
            x_coarse = x_coarse + dt * v_coarse
            
            # Generate medium
            v_medium = self.medium_flow(x_medium, t, text_features)
            x_medium = x_medium + dt * v_medium
            
            # Generate fine
            v_fine = self.fine_flow(x_fine, t, text_features)
            x_fine = x_fine + dt * v_fine
        
        # Combine all levels
        full_pose = torch.cat([x_coarse, x_medium, x_fine], dim=-1)
        # [B, T, 150]
        
        return {
            'pose_sequence': full_pose,
            'coarse': x_coarse,
            'medium': x_medium,
            'fine': x_fine
        }