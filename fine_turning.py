# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from peft import get_peft_model, LoraConfig, TaskType
from diffusers import AutoencoderKL, DDPMScheduler
from transformers import T5EncoderModel, T5Tokenizer
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import json

# ==============================
# CONFIGURATION
# ==============================
CONFIG = {
    'batch_size': 2,
    'num_epochs': 10,
    'lr': 5e-5,
    'lora_r': 8,
    'lora_alpha': 16,
    'anchor_dim': 768,
    'text_dim': 768,
    'motion_dim': 512,
    'num_frames': 16,
    'image_size': 256,
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'mixed_precision': True,
    'gradient_accumulation_steps': 4,
}

device = CONFIG['device']
print(f"🚀 Using device: {device}")

# ==============================
# 1. LATTE VIDEO TRANSFORMER
# ==============================
class LatteTransformer3D(nn.Module):
    """Simplified Latte-style 3D Transformer for video generation"""
    def __init__(self, dim=768, depth=12, heads=12, dim_head=64):
        super().__init__()
        self.dim = dim
        self.layers = nn.ModuleList([
            TransformerBlock(dim, heads, dim_head) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        
    def forward(self, x, context=None):
        """
        x: (B, T, H, W, C) - video latents
        context: (B, seq_len, dim) - conditioning
        """
        B, T, H, W, C = x.shape
        x = x.reshape(B, T*H*W, C)
        
        for layer in self.layers:
            x = layer(x, context)
        
        x = self.norm(x)
        x = x.reshape(B, T, H, W, C)
        return x

class TransformerBlock(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.self_attn = MultiHeadAttention(dim, heads, dim_head)
        
        self.norm2 = nn.LayerNorm(dim)
        self.cross_attn = MultiHeadAttention(dim, heads, dim_head)
        
        self.norm3 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, dim * 4)
        
    def forward(self, x, context=None):
        # Self-attention
        x = x + self.self_attn(self.norm1(x))
        
        # Cross-attention with conditioning
        if context is not None:
            x = x + self.cross_attn(self.norm2(x), context)
        
        # FFN
        x = x + self.ffn(self.norm3(x))
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, heads, dim_head):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head
        
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)
        self.to_out = nn.Linear(inner_dim, dim)
        
    def forward(self, x, context=None):
        B, N, C = x.shape
        h = self.heads
        
        q = self.to_q(x)
        k = self.to_k(context if context is not None else x)
        v = self.to_v(context if context is not None else x)
        
        q, k, v = map(lambda t: t.reshape(B, -1, h, C//h).transpose(1, 2), (q, k, v))
        
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(B, N, -1)
        return self.to_out(out)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim)
        )
    
    def forward(self, x):
        return self.net(x)

# ==============================
# 2. ANCHOR IMAGE ENCODER (ResNet)
# ==============================
class AnchorImageEncoder(nn.Module):
    def __init__(self, out_dim=768):
        super().__init__()
        base = models.resnet34(weights=models.ResNet34_Weights.IMAGENET1K_V1)
        self.features = nn.Sequential(*list(base.children())[:-1])
        self.projection = nn.Sequential(
            nn.Linear(512, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, x):
        # x: (B, 3, H, W)
        feat = self.features(x)  # (B, 512, 1, 1)
        feat = feat.flatten(1)   # (B, 512)
        return self.projection(feat).unsqueeze(1)  # (B, 1, 768)

# ==============================
# 3. MOTION EMBEDDING PROCESSOR
# ==============================
class MotionEmbeddingProcessor(nn.Module):
    def __init__(self, motion_dim=512, out_dim=768):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(motion_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim)
        )
        
    def forward(self, x):
        # x: (B, motion_dim) or (B, seq_len, motion_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, motion_dim)
        return self.projection(x)  # (B, seq_len, out_dim)

# ==============================
# 4. TEXT EMBEDDING PROCESSOR
# ==============================
class TextEmbeddingProcessor(nn.Module):
    def __init__(self, text_dim=768, out_dim=768):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(text_dim, out_dim),
            nn.LayerNorm(out_dim),
            nn.GELU()
        )
        
    def forward(self, x):
        # x: (B, text_dim) or (B, seq_len, text_dim)
        if x.dim() == 2:
            x = x.unsqueeze(1)  # (B, 1, text_dim)
        return self.projection(x)

# ==============================
# 5. CONDITIONING FUSION MODULE
# ==============================
class ConditioningFusion(nn.Module):
    def __init__(self, dim=768):
        super().__init__()
        self.text_gate = nn.Linear(dim, dim)
        self.motion_gate = nn.Linear(dim, dim)
        self.anchor_gate = nn.Linear(dim, dim)
        self.fusion = nn.Sequential(
            nn.Linear(dim * 3, dim),
            nn.LayerNorm(dim),
            nn.GELU()
        )
        
    def forward(self, text_emb, motion_emb, anchor_emb):
        # All: (B, seq_len, dim)
        text_w = torch.sigmoid(self.text_gate(text_emb))
        motion_w = torch.sigmoid(self.motion_gate(motion_emb))
        anchor_w = torch.sigmoid(self.anchor_gate(anchor_emb))
        
        weighted_text = text_emb * text_w
        weighted_motion = motion_emb * motion_w
        weighted_anchor = anchor_emb * anchor_w
        
        # Concatenate and fuse
        combined = torch.cat([weighted_text, weighted_motion, weighted_anchor], dim=-1)
        fused = self.fusion(combined)
        return fused

# ==============================
# 6. MAIN VIDEO GENERATION MODEL
# ==============================
class LatteVideoModel(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Core components
        self.transformer = LatteTransformer3D(dim=768, depth=12, heads=12)
        self.anchor_encoder = AnchorImageEncoder(out_dim=768)
        self.motion_processor = MotionEmbeddingProcessor(
            motion_dim=config['motion_dim'], 
            out_dim=768
        )
        self.text_processor = TextEmbeddingProcessor(
            text_dim=config['text_dim'],
            out_dim=768
        )
        self.fusion = ConditioningFusion(dim=768)
        
        # Video latent projection
        self.input_proj = nn.Conv3d(4, 768, kernel_size=1)
        self.output_proj = nn.Conv3d(768, 4, kernel_size=1)
        
    def forward(self, noisy_latents, timesteps, text_emb, motion_emb, anchor_img):
        """
        noisy_latents: (B, 4, T, H, W) - noisy video latents
        timesteps: (B,) - diffusion timesteps
        text_emb: (B, text_dim) - text embeddings
        motion_emb: (B, motion_dim) - motion embeddings
        anchor_img: (B, 3, H, W) - anchor first frame
        """
        B, C, T, H, W = noisy_latents.shape
        
        # Process all conditionings
        anchor_emb = self.anchor_encoder(anchor_img)  # (B, 1, 768)
        motion_emb = self.motion_processor(motion_emb)  # (B, 1, 768)
        text_emb = self.text_processor(text_emb)  # (B, 1, 768)
        
        # Fuse conditionings
        context = self.fusion(text_emb, motion_emb, anchor_emb)  # (B, 1, 768)
        
        # Add timestep embedding to context
        t_emb = self.get_timestep_embedding(timesteps, 768)  # (B, 768)
        context = context + t_emb.unsqueeze(1)
        
        # Project input
        x = self.input_proj(noisy_latents)  # (B, 768, T, H, W)
        x = x.permute(0, 2, 3, 4, 1)  # (B, T, H, W, 768)
        
        # Transformer with cross-attention
        x = self.transformer(x, context)  # (B, T, H, W, 768)
        
        # Project output
        x = x.permute(0, 4, 1, 2, 3)  # (B, 768, T, H, W)
        noise_pred = self.output_proj(x)  # (B, 4, T, H, W)
        
        return noise_pred
    
    def get_timestep_embedding(self, timesteps, dim):
        half_dim = dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=timesteps.device) * -emb)
        emb = timesteps[:, None] * emb[None, :]
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=-1)
        return emb

# ==============================
# 7. DATASET
# ==============================
class SignLanguageVideoDataset(Dataset):
    def __init__(self, video_root, text_emb_path, motion_emb_path, config):
        self.video_root = video_root
        self.config = config
        
        # Load embeddings
        self.text_emb = torch.tensor(np.load(text_emb_path)).float()
        self.motion_emb = torch.load(motion_emb_path).float()
        
        # Get video list
        self.video_list = sorted([d for d in os.listdir(video_root) 
                                 if os.path.isdir(os.path.join(video_root, d))])
        
        # Image transform
        self.transform = transforms.Compose([
            transforms.Resize((config['image_size'], config['image_size'])),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
    def __len__(self):
        return min(len(self.video_list), len(self.text_emb), len(self.motion_emb))
    
    def __getitem__(self, idx):
        vid_name = self.video_list[idx]
        vid_path = os.path.join(self.video_root, vid_name)
        
        # Get all frames
        frames = sorted([f for f in os.listdir(vid_path) if f.endswith(('.png', '.jpg'))])
        
        # Load anchor (first frame)
        anchor_img = Image.open(os.path.join(vid_path, frames[0])).convert('RGB')
        anchor_tensor = self.transform(anchor_img)
        
        # Sample frames for video
        num_frames = min(len(frames), self.config['num_frames'])
        indices = np.linspace(0, len(frames)-1, num_frames).astype(int)
        
        video_frames = []
        for i in indices:
            img = Image.open(os.path.join(vid_path, frames[i])).convert('RGB')
            video_frames.append(self.transform(img))
        
        video_tensor = torch.stack(video_frames)  # (T, 3, H, W)
        
        # Get embeddings
        text_emb = self.text_emb[idx]
        motion_emb = self.motion_emb[idx]
        
        return {
            'video': video_tensor,
            'anchor': anchor_tensor,
            'text_emb': text_emb,
            'motion_emb': motion_emb
        }

# ==============================
# 8. TRAINING LOOP
# ==============================
def train(model, dataloader, optimizer, scheduler, scaler, config):
    model.train()
    total_loss = 0
    
    for step, batch in enumerate(tqdm(dataloader, desc="Training")):
        video = batch['video'].to(device)  # (B, T, 3, H, W)
        anchor = batch['anchor'].to(device)  # (B, 3, H, W)
        text_emb = batch['text_emb'].to(device)
        motion_emb = batch['motion_emb'].to(device)
        
        B, T, C, H, W = video.shape
        
        # Simulate VAE encoding (normally would use real VAE)
        # For simplicity, we'll use a simple projection
        latents = video.permute(0, 2, 1, 3, 4)  # (B, 3, T, H, W)
        latents = F.interpolate(latents.flatten(0, 1), size=(32, 32))
        latents = latents.view(B, C, T, 32, 32)
        # Pad to 4 channels
        latents = F.pad(latents, (0, 0, 0, 0, 0, 0, 0, 1))  # (B, 4, T, 32, 32)
        
        # Sample noise and timesteps
        noise = torch.randn_like(latents)
        timesteps = torch.randint(0, 1000, (B,), device=device).long()
        
        # Add noise (simplified diffusion)
        alpha = (1000 - timesteps) / 1000
        alpha = alpha.view(B, 1, 1, 1, 1)
        noisy_latents = alpha * latents + (1 - alpha) * noise
        
        # Forward pass
        with torch.cuda.amp.autocast(enabled=config['mixed_precision']):
            noise_pred = model(noisy_latents, timesteps, text_emb, motion_emb, anchor)
            loss = F.mse_loss(noise_pred, noise)
        
        # Backward pass
        scaler.scale(loss).backward()
        
        if (step + 1) % config['gradient_accumulation_steps'] == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)

# ==============================
# 9. MAIN TRAINING SCRIPT
# ==============================
def main():
    # Create model
    model = LatteVideoModel(CONFIG).to(device)
    
    # Add LoRA to cross-attention layers
    lora_config = LoraConfig(
        r=CONFIG['lora_r'],
        lora_alpha=CONFIG['lora_alpha'],
        target_modules=["to_q", "to_k", "to_v"],
        lora_dropout=0.05,
        bias="none",
    )
    model.transformer = get_peft_model(model.transformer, lora_config)
    
    print(f"📊 Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"📊 Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create dataset (update paths to your data)
    dataset = SignLanguageVideoDataset(
        video_root="features/fullFrame-210x260px/train",
        text_emb_path="data/emdeding_text/phoenix_train_embeddings.npy",
        motion_emb_path="data/RWTH/PHOENIX-2014-T-release-v3/PHOENIX-2014-T/mcf_embeddings_single_file_v1/train_mcf_embeddings.pt",
        config=CONFIG
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=CONFIG['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    # Optimizer and scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'], weight_decay=0.01)
    scheduler = DDPMScheduler(num_train_timesteps=1000)
    scaler = torch.cuda.amp.GradScaler(enabled=CONFIG['mixed_precision'])
    
    # Training loop
    for epoch in range(CONFIG['num_epochs']):
        avg_loss = train(model, dataloader, optimizer, scheduler, scaler, CONFIG)
        print(f"✅ Epoch {epoch+1}/{CONFIG['num_epochs']} | Loss: {avg_loss:.6f}")
        
        # Save checkpoint
        if (epoch + 1) % 2 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f"checkpoint_epoch_{epoch+1}.pt")
    
    # Save final model
    model.transformer.save_pretrained("latte_lora_final")
    torch.save(model.state_dict(), "latte_video_model_final.pt")
    print("🎉 Training completed!")

if __name__ == "__main__":
    main()