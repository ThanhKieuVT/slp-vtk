# Tên file: train_ldm_teacher.py
import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# --- Import các model và data của chị ---
# (Giả sử chị copy/paste các file này vào đúng chỗ)
from utils.data_loader import SignLanguageDataset, collate_fn # Tái sử dụng
from models.autoencoder import Stage1Autoencoder # Tái sử dụng Stage 1
from models.ldm_denoiser import LDM_TransformerDenoiser # Model MỚI
from models.losses import VelocityLoss # Loss MỚI

# --- Import các thư viện SOTA ---
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import DDPMScheduler

# --- Cấu hình các "Bí kíp" SOTA ---
CFG_PROBABILITY = 0.1       # 10% bỏ trống text
GRAD_ACCUMULATION_STEPS = 4 # Tăng batch size hiệu dụng
W_VELOCITY_LOSS = 0.05      # Trọng số cho Velocity Loss

def train_epoch(
    ldm_model, autoencoder, text_encoder, tokenizer,
    dataloader, optimizer, scheduler, scaler,
    noise_scheduler, velocity_loss_fn, device, epoch
):
    ldm_model.train()
    text_encoder.eval()  # Luôn đóng băng text encoder
    autoencoder.eval()   # Luôn đóng băng autoencoder
    
    total_loss_epoch = 0.0
    total_base_loss_epoch = 0.0
    total_vel_loss_epoch = 0.0
    
    # --- Lấy token "NULL" (empty text) cho CFG ---
    null_text_input = tokenizer(
        "", padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        null_text_embeddings = text_encoder(
            null_text_input.input_ids,
            attention_mask=null_text_input.attention_mask
        )[0]
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    for i, batch in enumerate(pbar):
        
        poses = batch['poses'].to(device)
        pose_mask = batch['pose_mask'].to(device) # [B, T]
        text_batch = batch['text_list'] # Lấy list text
        
        # 1. Encode GT (với frozen A)
        with torch.no_grad():
            gt_z0 = autoencoder.encode(poses, mask=pose_mask) # [B, T, D]
        
        batch_size = gt_z0.shape[0]

        # 2. Encode Text (với frozen CLIP)
        text_inputs = tokenizer(
            text_batch, padding="max_length", 
            max_length=tokenizer.model_max_length,
            truncation=True, return_tensors="pt"
        ).to(device)
        with torch.no_grad():
            text_embeddings = text_encoder(
                text_inputs.input_ids,
                attention_mask=text_inputs.attention_mask
            )[0] # [B, L, D_text]

        # 3. Kỹ thuật: Classifier-Free Guidance (CFG)
        mask_cfg = torch.rand(batch_size, device=device) < CFG_PROBABILITY
        text_embeddings[mask_cfg] = null_text_embeddings
        
        # 4. Bước Diffusion: Thêm nhiễu
        noise_gt = torch.randn_like(gt_z0)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                (batch_size,), device=device).long()
        z_t = noise_scheduler.add_noise(gt_z0, noise_gt, timesteps)
        
        # 5. Huấn luyện (AMP + Gradient Accumulation)
        # Dùng autocast (float16)
        with autocast(device_type='cuda', dtype=torch.float16):
            
            # Mask cho Transformer (True = ignore)
            pose_padding_mask = ~pose_mask
            text_padding_mask = ~text_inputs.attention_mask.bool()

            # Dự đoán nhiễu
            noise_pred = ldm_model(
                z_t,                     # [B, T, D_latent]
                timesteps,               # [B]
                text_embeddings,         # [B, L, D_text]
                text_mask=text_padding_mask,
                pose_mask=pose_padding_mask
            )
            
            # --- TÍNH LOSS ---
            
            # 5a. Loss cơ bản (L1) trên nhiễu
            loss_base = F.l1_loss(noise_pred, noise_gt, reduction='none')
            loss_base = loss_base * pose_mask.unsqueeze(-1).float()
            loss_base = loss_base.sum() / pose_mask.sum().clamp(min=1)
            
            # 5b. Kỹ thuật: Velocity Loss (để mượt)
            # Lấy z0 dự đoán (pred_z0) từ scheduler
            pred_z0 = noise_scheduler.get_original_sample(z_t, timesteps, noise_pred)
            loss_vel = velocity_loss_fn(pred_z0, gt_z0, mask=pose_mask)
            
            # 5c. Tổng loss
            total_loss = loss_base + W_VELOCITY_LOSS * loss_vel
            
            # Chuẩn hóa loss cho Gradient Accumulation
            loss_to_backward = total_loss / GRAD_ACCUMULATION_STEPS
            
        # 6. Backward (AMP)
        scaler.scale(loss_to_backward).backward()

        # 7. Kỹ thuật: Gradient Accumulation
        if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ldm_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss_epoch += total_loss.item()
        total_base_loss_epoch += loss_base.item()
        total_vel_loss_epoch += loss_vel.item()
        
        pbar.set_postfix({
            'loss': total_loss.item(),
            'base': loss_base.item(),
            'vel': loss_vel.item(),
            'lr': optimizer.param_groups[0]['lr']
        })

    avg_loss = total_loss_epoch / len(dataloader)
    avg_base = total_base_loss_epoch / len(dataloader)
    avg_vel = total_vel_loss_epoch / len(dataloader)
    
    # Cập nhật LR Scheduler mỗi epoch
    scheduler.step()
    
    return {'total': avg_loss, 'base': avg_base, 'vel': avg_vel}


# (Hàm validate tương tự, nhưng không cần backward)
# ...

def main():
    parser = argparse.ArgumentParser()
    # (Thêm các parser args của chị: --data_dir, --output_dir, v.v...)
    parser.add...
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # --- 1. Tải các model đã đóng băng (Frozen) ---
    print("Loading Stage 1 Autoencoder...")
    autoencoder = Stage1Autoencoder(...) # (Load model của chị)
    autoencoder.load_state_dict(torch.load(args.autoencoder_checkpoint))
    autoencoder.to(device).eval().requires_grad_(False)
    
    print("Loading CLIP Text Encoder...")
    clip_name = "openai/clip-vit-base-patch32"
    tokenizer = CLIPTokenizer.from_pretrained(clip_name)
    text_encoder = CLIPTextModel.from_pretrained(clip_name).to(device)
    text_encoder.eval().requires_grad_(False)
    text_embed_dim = text_encoder.config.hidden_size
    
    # --- 2. Tải Dataloader (tái sử dụng) ---
    train_dataset = SignLanguageDataset(data_dir=args.data_dir, split='train', ...)
    train_loader = DataLoader(train_dataset, ...)
    # (Tương tự cho val_loader)

    # --- 3. Model chính (LDM Denoiser) ---
    print("Initializing LDM Denoiser model...")
    ldm_model = LDM_TransformerDenoiser(
        latent_dim=args.latent_dim,
        text_embed_dim=text_embed_dim,
        hidden_dim=args.hidden_dim,
        num_layers=6, # (Hyperparameter)
        num_heads=8
    ).to(device)
    
    # --- 4. Schedulers và Loss ---
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='squaredcos_cap_v2' # (Schedule SOTA)
    )
    velocity_loss_fn = VelocityLoss(loss_type='l1').to(device)
    
    # --- 5. Optimizer và Hỗ trợ Training ---
    optimizer = torch.optim.AdamW(ldm_model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # (Thêm logic resume checkpoint ở đây nếu cần)
    
    print("Starting LDM Teacher training...")
    for epoch in range(args.num_epochs):
        train_losses = train_epoch(
            ldm_model, autoencoder, text_encoder, tokenizer,
            train_loader, optimizer, scheduler, scaler,
            noise_scheduler, velocity_loss_fn, device, epoch
        )
        
        print(f"\nEpoch {epoch+1}/{args.num_epochs} Summary:")
        print(f"  Train Loss: {train_losses['total']:.6f}")
        print(f"    Base Loss (L1): {train_losses['base']:.6f}")
        print(f"    Vel. Loss (L1): {train_losses['vel']:.6f}")
        
        # (Chạy validation ở đây)
        
        # (Lưu checkpoint 'best_model.pt' và 'latest.pt')

if __name__ == '__main__':
    main()