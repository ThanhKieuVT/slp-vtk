# Tên file: train_ldm_mamba_teacher.py
# === PHIÊN BẢN FINAL MAMBA (FIXED: AUTO-SCALING + MSE LOSS) ===

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

from utils.data_loader import SignLanguageDataset, collate_fn
from models.autoencoder import UnifiedPoseAutoencoder
# === BACKBONE MAMBA ===
from models.ldm_denoiser_mamba import LDM_Mamba_Denoiser 
# ======================
from transformers import BertModel, BertTokenizer
from diffusers import DDPMScheduler

CFG_PROBABILITY = 0.1
GRAD_ACCUMULATION_STEPS = 4
VALIDATION_SEED = 42 

# Hàm tự động tính Scale Factor (BẮT BUỘC PHẢI CÓ CHO LDM)
def estimate_scale_factor(autoencoder, dataloader, device, num_batches=10):
    print(f"⏳ [MAMBA SETUP] Đang tính toán Latent Scale Factor...")
    autoencoder.eval()
    all_latents = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            poses = batch['poses'].to(device)
            pose_mask = batch['pose_mask'].to(device)
            z = autoencoder.encode(poses, mask=pose_mask)
            all_latents.append(z.cpu())
    
    all_latents = torch.cat(all_latents, dim=0)
    std = all_latents.std()
    scale_factor = 1.0 / std.item()
    
    print(f"✅ Latent Std gốc: {std.item():.4f}")
    print(f"✅ Scale Factor: {scale_factor:.6f}")
    return scale_factor

# -------------------------- TRAIN EPOCH --------------------------
def train_epoch(ldm_model, autoencoder, text_encoder, dataloader,
                optimizer, scheduler, scaler, noise_scheduler,
                null_text_embeddings, device, epoch, scale_factor):
    ldm_model.train()
    text_encoder.eval()
    autoencoder.eval()

    total_loss_epoch = 0.0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training (Mamba)")
    
    for i, batch in enumerate(pbar):
        poses = batch['poses'].to(device)
        pose_mask = batch['pose_mask'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        # 1. Encode & SCALE (Quan trọng nhất)
        with torch.no_grad():
            gt_z0 = autoencoder.encode(poses, mask=pose_mask)
            gt_z0 = gt_z0 * scale_factor

        batch_size = gt_z0.shape[0]

        # 2. Text Embedding & CFG
        with torch.no_grad():
            text_kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
            text_embeddings = text_encoder(text_tokens, **text_kwargs).last_hidden_state

        mask_cfg = torch.rand(batch_size, device=device) < CFG_PROBABILITY
        null_embed_expanded = null_text_embeddings.expand(batch_size, -1, -1).clone()
        text_embeddings = torch.where(mask_cfg.view(-1,1,1), null_embed_expanded, text_embeddings)

        # 3. Add Noise
        noise_gt = torch.randn_like(gt_z0)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        z_t = noise_scheduler.add_noise(gt_z0, noise_gt, timesteps)

        # 4. Forward Mamba
        with autocast(dtype=torch.float16):
            pose_padding_mask = (~pose_mask).to(torch.bool)
            text_padding_mask = (attention_mask == 0).to(torch.bool)

            noise_pred = ldm_model(z_t, timesteps, text_embeddings,
                                   text_mask=text_padding_mask, pose_mask=pose_padding_mask)

            # 5. MSE Loss (Chuẩn cho LDM)
            loss_mse = F.mse_loss(noise_pred, noise_gt, reduction='none')
            loss = (loss_mse * pose_mask.unsqueeze(-1).float()).sum() / pose_mask.sum().clamp(min=1)
            
            loss_to_backward = loss / GRAD_ACCUMULATION_STEPS

        scaler.scale(loss_to_backward).backward()

        if (i+1) % GRAD_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ldm_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss_epoch += loss.item()
        pbar.set_postfix({'mse_loss': f'{loss.item():.4f}'})

    avg_loss = total_loss_epoch / len(dataloader)
    scheduler.step()
    return avg_loss


# -------------------------- VALIDATION --------------------------
def validate(ldm_model, autoencoder, text_encoder, dataloader,
             noise_scheduler, null_text_embeddings, device, scale_factor):
    ldm_model.eval()
    total_loss_epoch = 0.0
    
    torch.manual_seed(VALIDATION_SEED)
    pbar = tqdm(dataloader, desc="Validating")

    with torch.no_grad():
        for batch in pbar:
            poses = batch['poses'].to(device)
            pose_mask = batch['pose_mask'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            # Encode & Scale
            gt_z0 = autoencoder.encode(poses, mask=pose_mask)
            gt_z0 = gt_z0 * scale_factor

            batch_size = gt_z0.shape[0]

            text_kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
            text_embeddings = text_encoder(text_tokens, **text_kwargs).last_hidden_state

            noise_gt = torch.randn_like(gt_z0)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
            z_t = noise_scheduler.add_noise(gt_z0, noise_gt, timesteps)

            with autocast(dtype=torch.float16):
                pose_padding_mask = (~pose_mask).to(torch.bool)
                text_padding_mask = (attention_mask == 0).to(torch.bool)

                noise_pred = ldm_model(z_t, timesteps, text_embeddings,
                                       text_mask=text_padding_mask, pose_mask=pose_padding_mask)

                loss_mse = F.mse_loss(noise_pred, noise_gt, reduction='none')
                loss = (loss_mse * pose_mask.unsqueeze(-1).float()).sum() / pose_mask.sum().clamp(min=1)

            total_loss_epoch += loss.item()
            pbar.set_postfix({'val_mse': f'{loss.item():.4f}'})

    avg_loss = total_loss_epoch / len(dataloader)
    return avg_loss


# -------------------------- MAIN --------------------------
def main():
    parser = argparse.ArgumentParser(description="Train LDM Mamba Teacher Model")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--ae_hidden_dim', type=int, default=512)
    parser.add_argument('--text_embed_dim', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--max_seq_len', type=int, default=120)
    parser.add_argument('--max_text_len', type=int, default=64)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. Setup Autoencoder (Frozen)
    autoencoder = UnifiedPoseAutoencoder(pose_dim=214, latent_dim=args.latent_dim, hidden_dim=args.ae_hidden_dim)
    ae_checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
    if 'model_state_dict' in ae_checkpoint:
        autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
    else:
        autoencoder.load_state_dict(ae_checkpoint)
    autoencoder.to(device).eval().requires_grad_(False)

    # 2. Setup BERT (Frozen)
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    text_encoder = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
    text_encoder.eval().requires_grad_(False)

    # 3. Datasets
    train_dataset = SignLanguageDataset(args.data_dir, split='train', max_seq_len=args.max_seq_len, max_text_len=args.max_text_len, normalize=True)
    val_dataset = SignLanguageDataset(args.data_dir, split='dev', max_seq_len=args.max_seq_len, max_text_len=args.max_text_len, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # === FIX: AUTO CALCULATE SCALE FACTOR ===
    latent_scale_factor = estimate_scale_factor(autoencoder, train_loader, device)

    # 4. LDM Model (MAMBA)
    # Lưu ý: Đảm bảo class LDM_Mamba_Denoiser nhận đúng tham số này
    ldm_model = LDM_Mamba_Denoiser(latent_dim=args.latent_dim, text_embed_dim=args.text_embed_dim,
                                  hidden_dim=args.text_embed_dim, num_layers=args.num_layers, num_heads=12).to(device)

    # 5. Scheduler & Optimizer
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    noise_scheduler.config.prediction_type = "epsilon"

    optimizer = torch.optim.AdamW(ldm_model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    scaler = GradScaler()

    # NULL embeddings
    null_text_input = tokenizer("", padding="max_length", max_length=args.max_text_len, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        null_text_embeddings = text_encoder(null_text_input.input_ids, attention_mask=null_text_input.attention_mask, token_type_ids=null_text_input.token_type_ids).last_hidden_state

    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        ldm_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        latent_scale_factor = checkpoint.get('latent_scale_factor', latent_scale_factor)
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed Scale Factor: {latent_scale_factor}")

    # Loop
    for epoch in range(start_epoch, args.num_epochs):
        train_loss = train_epoch(ldm_model, autoencoder, text_encoder, train_loader,
                                 optimizer, scheduler, scaler, noise_scheduler,
                                 null_text_embeddings, device, epoch, latent_scale_factor)
        val_loss = validate(ldm_model, autoencoder, text_encoder, val_loader,
                            noise_scheduler, null_text_embeddings, device, latent_scale_factor)

        print(f"\n=== EPOCH {epoch+1}/{args.num_epochs} (Mamba) ===")
        print(f"Train MSE: {train_loss:.6f}")
        print(f"Val   MSE: {val_loss:.6f} (Best: {best_val_loss:.6f})")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': ldm_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'latent_scale_factor': latent_scale_factor,
            'args': args
        }
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pt'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"✅ New best model saved! MSE: {best_val_loss:.6f}")

    print(f"\n🎉 TRAINING COMPLETED! Best Val MSE: {best_val_loss:.6f}")

if __name__ == '__main__':
    main()