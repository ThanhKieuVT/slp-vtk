# models/nmm_generator.py
"""
Non-Manual Marker Generators
4 independent flow models for facial expressions, head pose, gaze, mouth
"""
import torch
import torch.nn as nn
from transformers import BertModel
from .flow_matching import FlowMatchingBlock, FlowMatchingScheduler, FlowMatchingLoss

class NMMFlowGenerator(nn.Module):
    """
    Complete NMM generation system
    
    4 Components:
        1. Facial AUs (17D) - Facial Action Units
        2. Head Pose (3D) - Pitch, yaw, roll
        3. Eye Gaze (4D) - Left/right eye direction
        4. Mouth Shape (40D) - Mouth landmarks
    
    Total: 64D
    """
    def __init__(
        self,
        text_encoder_name='bert-base-german-cased',
        hidden_dim=384,  # Smaller than manual
        num_flow_layers=6,
        num_heads=6,
        dropout=0.1,
        freeze_text_encoder=False,
        share_text_encoder=None  # Share with manual branch
    ):
        super().__init__()
        
        # Text encoder (shared or separate)
        if share_text_encoder is not None:
            self.text_encoder = share_text_encoder
            print("Sharing text encoder with manual branch")
        else:
            self.text_encoder = BertModel.from_pretrained(text_encoder_name)
            if freeze_text_encoder:
                for param in self.text_encoder.parameters():
                    param.requires_grad = False
        
        text_dim = 768
        
        # Flow scheduler
        self.scheduler = FlowMatchingScheduler()
        
        # Component 1: Facial AUs
        self.facial_au_flow = FlowMatchingBlock(
            data_dim=17,
            condition_dim=text_dim,
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Component 2: Head Pose
        self.head_pose_flow = FlowMatchingBlock(
            data_dim=3,
            condition_dim=text_dim,
            hidden_dim=256,  # Even smaller
            num_layers=4,
            num_heads=4,
            dropout=dropout
        )
        
        # Component 3: Eye Gaze
        self.eye_gaze_flow = FlowMatchingBlock(
            data_dim=4,
            condition_dim=text_dim,
            hidden_dim=256,
            num_layers=4,
            num_heads=4,
            dropout=dropout
        )
        
        # Component 4: Mouth Shape
        self.mouth_shape_flow = FlowMatchingBlock(
            data_dim=40,
            condition_dim=text_dim,
            hidden_dim=hidden_dim,
            num_layers=num_flow_layers,
            num_heads=num_heads,
            dropout=dropout
        )
        
        # Loss function
        self.loss_fn = FlowMatchingLoss()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(f"NMMFlowGenerator: {total_params/1e6:.1f}M trainable parameters")
    
    def encode_text(self, text_tokens, attention_mask):
        """Encode text (same as manual branch)"""
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
        
        Returns:
            If train: dict with losses
            If inference: dict with generated NMMs
        """
        # Encode text
        text_features = self.encode_text(
            batch['text_tokens'],
            batch['attention_mask']
        )
        
        if mode == 'train':
            return self._train_forward(batch, text_features)
        else:
            return self._inference_forward(batch, text_features, num_inference_steps)
    
    def _train_forward(self, batch, text_features):
        """Training forward"""
        device = text_features.device
        B = text_features.shape[0]
        
        # Sample timesteps
        t = self.scheduler.sample_timesteps(B, device)
        
        # Mask
        T_max = batch['nmm_facial_aus'].shape[1]
        seq_lengths = batch['seq_lengths']
        mask = torch.arange(T_max, device=device)[None, :] >= seq_lengths[:, None]
        
        losses = {}
        
        # --- Component 1: Facial AUs ---
        x_0_au = batch['nmm_facial_aus']  # [B, T, 17]
        x_t_au, v_gt_au, _ = self.scheduler.add_noise(x_0_au, t)
        
        v_pred_au = self.facial_au_flow(x_t_au, t, text_features, mask=mask)
        losses['facial_au'] = self.loss_fn(v_pred_au, v_gt_au, ~mask)
        
        # --- Component 2: Head Pose ---
        x_0_head = batch['nmm_head_pose']  # [B, T, 3]
        x_t_head, v_gt_head, _ = self.scheduler.add_noise(x_0_head, t)
        
        v_pred_head = self.head_pose_flow(x_t_head, t, text_features, mask=mask)
        losses['head_pose'] = self.loss_fn(v_pred_head, v_gt_head, ~mask)
        
        # --- Component 3: Eye Gaze ---
        x_0_gaze = batch['nmm_eye_gaze']  # [B, T, 4]
        x_t_gaze, v_gt_gaze, _ = self.scheduler.add_noise(x_0_gaze, t)
        
        v_pred_gaze = self.eye_gaze_flow(x_t_gaze, t, text_features, mask=mask)
        losses['eye_gaze'] = self.loss_fn(v_pred_gaze, v_gt_gaze, ~mask)
        
        # --- Component 4: Mouth Shape ---
        x_0_mouth = batch['nmm_mouth_shape']  # [B, T, 40]
        x_t_mouth, v_gt_mouth, _ = self.scheduler.add_noise(x_0_mouth, t)
        
        v_pred_mouth = self.mouth_shape_flow(x_t_mouth, t, text_features, mask=mask)
        losses['mouth_shape'] = self.loss_fn(v_pred_mouth, v_gt_mouth, ~mask)
        
        # Total loss
        losses['total'] = (
            2.0 * losses['facial_au'] +    # AUs important
            1.0 * losses['head_pose'] +
            0.5 * losses['eye_gaze'] +
            1.5 * losses['mouth_shape']    # Mouth important for German
        )
        
        return losses
    
    @torch.no_grad()
    def _inference_forward(self, batch, text_features, num_steps=20):
        """Inference: Generate NMMs"""
        device = text_features.device
        B, L, _ = text_features.shape
        T = batch.get('target_length', 50)
        
        # Start from noise
        x_au = torch.randn(B, T, 17, device=device)
        x_head = torch.randn(B, T, 3, device=device)
        x_gaze = torch.randn(B, T, 4, device=device)
        x_mouth = torch.randn(B, T, 40, device=device)
        
        # Integration
        dt = 1.0 / num_steps
        
        for step in range(num_steps):
            t_val = step / num_steps
            t = torch.full((B,), t_val, device=device)
            
            # Generate all components
            v_au = self.facial_au_flow(x_au, t, text_features)
            x_au = x_au + dt * v_au
            
            v_head = self.head_pose_flow(x_head, t, text_features)
            x_head = x_head + dt * v_head
            
            v_gaze = self.eye_gaze_flow(x_gaze, t, text_features)
            x_gaze = x_gaze + dt * v_gaze
            
            v_mouth = self.mouth_shape_flow(x_mouth, t, text_features)
            x_mouth = x_mouth + dt * v_mouth
        
        return {
            'facial_aus': x_au,
            'head_pose': x_head,
            'eye_gaze': x_gaze,
            'mouth_shape': x_mouth
        }