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
        text_encoder_name='bert-base-multilingual-cased',
        #text_encoder_name = '/Users/kieuvo/Learn/Research/SL/Implement/SignFML/slp-vtk/local_bert_model',
        hidden_dim=1024,  # ✅ giữ 1024 để có model mạnh hơn
        num_flow_layers=6,
        num_heads=8,
        dropout=0.1,
        freeze_text_encoder=False
    ):
        super().__init__()
        
        # --- TEXT ENCODER ---
        self.text_encoder = BertModel.from_pretrained(text_encoder_name)
        text_dim = self.text_encoder.config.hidden_size  # 768 (BERT output)
        
        # ✅ Linear projection: 768 → 1024 (đảm bảo khớp với hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        
        if freeze_text_encoder:
            for param in self.text_encoder.parameters():
                param.requires_grad = False
            print("Text encoder frozen")
        
        # --- FLOW SCHEDULER ---
        self.scheduler = FlowMatchingScheduler()
        
        # --- FLOW LEVELS ---
        self.coarse_flow = FlowMatchingBlock(
            data_dim=66,
            condition_dim=hidden_dim,  # ✅ dùng hidden_dim sau khi đã proj
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.medium_flow = FlowMatchingBlock(
            data_dim=16,
            condition_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.fine_flow = FlowMatchingBlock(
            data_dim=68,
            condition_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers + 2,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # --- LOSS ---
        self.loss_fn = FlowMatchingLoss()
        
        # --- PARAM COUNT ---
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"HierarchicalFlowMatcher: {total_params/1e6:.1f}M trainable parameters")
    
    def encode_text(self, text_tokens, attention_mask):
        """
        Encode German text
        
        Args:
            text_tokens: [B, L] token IDs
            attention_mask: [B, L] mask
        
        Returns:
            [B, L, 1024] text features (projected)
        """
        outputs = self.text_encoder(
            input_ids=text_tokens,
            attention_mask=attention_mask
        )
        text_features = outputs.last_hidden_state  # [B, L, 768]
        text_features = self.text_proj(text_features)  # ✅ map 768 → 1024
        return text_features
    
    def forward(self, batch, mode='train', num_inference_steps=20):
        """
        Forward pass
        """
        text_features = self.encode_text(
            batch['text_tokens'],
            batch['attention_mask']
        )  # [B, L, 1024]
        
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
        
        # Mask padding
        T_max = batch['manual_coarse'].shape[1]
        seq_lengths = batch['seq_lengths']
        mask = torch.arange(T_max, device=device)[None, :] >= seq_lengths[:, None]
        
        losses = {}
        
        # --- Level 1: Coarse (Body) ---
        x_0_coarse = batch['manual_coarse']
        x_t_coarse, v_gt_coarse, _ = self.scheduler.add_noise(x_0_coarse, t)
        
        v_pred_coarse = self.coarse_flow(
            x_t_coarse, t, text_features, mask=mask
        )
        losses['coarse'] = self.loss_fn(v_pred_coarse, v_gt_coarse, ~mask)
        
        # --- Level 2: Medium (Arms) ---
        x_0_medium = batch['manual_medium']
        x_t_medium, v_gt_medium, _ = self.scheduler.add_noise(x_0_medium, t)
        
        v_pred_medium = self.medium_flow(
            x_t_medium, t, text_features, mask=mask
        )
        losses['medium'] = self.loss_fn(v_pred_medium, v_gt_medium, ~mask)
        
        # --- Level 3: Fine (Hands) ---
        x_0_fine = batch['manual_fine']
        x_t_fine, v_gt_fine, _ = self.scheduler.add_noise(x_0_fine, t)
        
        v_pred_fine = self.fine_flow(
            x_t_fine, t, text_features, mask=mask
        )
        losses['fine'] = self.loss_fn(v_pred_fine, v_gt_fine, ~mask)
        
        # --- Total loss ---
        losses['total'] = (
            1.0 * losses['coarse'] +
            1.0 * losses['medium'] +
            1.5 * losses['fine']
        )
        return losses
    
    @torch.no_grad()
    def _inference_forward(self, batch, text_features, num_steps=20):
        """Inference: Generate poses from noise via ODE integration"""
        device = text_features.device
        B, L, _ = text_features.shape
        T = batch.get('target_length', 50)
        
        x_coarse = torch.randn(B, T, 66, device=device)
        x_medium = torch.randn(B, T, 16, device=device)
        x_fine = torch.randn(B, T, 68, device=device)
        
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((B,), t_val, device=device)
            
            v_coarse = self.coarse_flow(x_coarse, t, text_features)
            x_coarse = x_coarse + dt * v_coarse
            
            v_medium = self.medium_flow(x_medium, t, text_features)
            x_medium = x_medium + dt * v_medium
            
            v_fine = self.fine_flow(x_fine, t, text_features)
            x_fine = x_fine + dt * v_fine
        
        full_pose = torch.cat([x_coarse, x_medium, x_fine], dim=-1)
        
        return {
            'pose_sequence': full_pose,
            'coarse': x_coarse,
            'medium': x_medium,
            'fine': x_fine
        }
