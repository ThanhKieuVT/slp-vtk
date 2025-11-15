# Tên file: train_ldm_teacher.py
# === PHIÊN BẢN FINAL (NÂNG CẤP LÕI LÊN MAMBA) ===

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
# === SỬA DÒNG 1 ===
from models.ldm_denoiser_mamba import LDM_Mamba_Denoiser # Đã đổi sang Mamba
# ==================
from models.losses import VelocityLoss
from transformers import BertModel, BertTokenizer
from diffusers import DDPMScheduler

CFG_PROBABILITY = 0.1
GRAD_ACCUMULATION_STEPS = 4
W_VELOCITY_LOSS = 0.001
VALIDATION_SEED = 42  # Fix timesteps seed cho validation ổn định

# -------------------------- TRAIN EPOCH --------------------------
def train_epoch(ldm_model, autoencoder, text_encoder, dataloader,
                optimizer, scheduler, scaler, noise_scheduler,
                velocity_loss_fn, null_text_embeddings, device, epoch):
    ldm_model.train()
    text_encoder.eval()
    autoencoder.eval()

    total_loss_epoch = 0.0
    total_base_epoch = 0.0
    total_vel_epoch = 0.0
    total_recon_l1_epoch = 0.0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Training")
    
    for i, batch in enumerate(pbar):
        poses = batch['poses'].to(device)
        pose_mask = batch['pose_mask'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        # Encode poses
        with torch.no_grad():
            gt_z0 = autoencoder.encode(poses, mask=pose_mask)

        batch_size = gt_z0.shape[0]

        # Text embeddings
        with torch.no_grad():
            text_kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
            text_embeddings = text_encoder(text_tokens, **text_kwargs).last_hidden_state

        # CFG
        mask_cfg = torch.rand(batch_size, device=device) < CFG_PROBABILITY
        null_embed_expanded = null_text_embeddings.expand(batch_size, -1, -1).clone()
        text_embeddings = torch.where(mask_cfg.view(-1,1,1), null_embed_expanded, text_embeddings)

        # Noise
        noise_gt = torch.randn_like(gt_z0)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, (batch_size,), device=device).long()
        z_t = noise_scheduler.add_noise(gt_z0, noise_gt, timesteps)

        # Forward
        with autocast(dtype=torch.float16):
            pose_padding_mask = (~pose_mask).to(torch.bool)
            text_padding_mask = (attention_mask == 0).to(torch.bool)

            noise_pred = ldm_model(z_t, timesteps, text_embeddings,
                                   text_mask=text_padding_mask, pose_mask=pose_padding_mask)

            # Base loss
            loss_base = F.l1_loss(noise_pred, noise_gt, reduction='none')
            loss_base = (loss_base * pose_mask.unsqueeze(-1).float()).sum() / pose_mask.sum().clamp(min=1)

            # pred_z0
            alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
            alpha_t = torch.sqrt(alphas_cumprod[timesteps]).view(-1,1,1)
            sigma_t = torch.sqrt(1.0 - alphas_cumprod[timesteps]).view(-1,1,1)
            pred_z0 = (z_t - sigma_t * noise_pred) / alpha_t

            # Velocity loss
            loss_vel = velocity_loss_fn(pred_z0, gt_z0, mask=pose_mask)

            # Recon L1
            loss_recon_l1 = F.l1_loss(pred_z0, gt_z0, reduction='none')
            loss_recon_l1 = (loss_recon_l1 * pose_mask.unsqueeze(-1).float()).sum() / pose_mask.sum().clamp(min=1)

            # Total
            total_loss = loss_base + W_VELOCITY_LOSS * loss_vel
            loss_to_backward = total_loss / GRAD_ACCUMULATION_STEPS

        scaler.scale(loss_to_backward).backward()

        if (i+1) % GRAD_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ldm_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        # Accumulate
        total_loss_epoch += total_loss.item()
        total_base_epoch += loss_base.item()
        total_vel_epoch += loss_vel.item()
        total_recon_l1_epoch += loss_recon_l1.item()

        pbar.set_postfix({'loss': f'{total_loss.item():.4f}',
                          'recon_l1': f'{loss_recon_l1.item():.4f}'})

    avg_loss = total_loss_epoch / len(dataloader)
    avg_base = total_base_epoch / len(dataloader)
    avg_vel = total_vel_epoch / len(dataloader)
    avg_recon_l1 = total_recon_l1_epoch / len(dataloader)

    scheduler.step()
    return {'total': avg_loss, 'base': avg_base, 'vel': avg_vel, 'recon_l1': avg_recon_l1}


# -------------------------- VALIDATION --------------------------
def validate(ldm_model, autoencoder, text_encoder, dataloader,
             noise_scheduler, velocity_loss_fn, null_text_embeddings, device):
    ldm_model.eval()

    total_loss_epoch = 0.0
    total_base_epoch = 0.0
    total_vel_epoch = 0.0
    total_recon_l1_epoch = 0.0

    torch.manual_seed(VALIDATION_SEED)  # fix seed
    pbar = tqdm(dataloader, desc="Validating")

    with torch.no_grad():
        for batch in pbar:
            poses = batch['poses'].to(device)
            pose_mask = batch['pose_mask'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch['token_type_ids'].to(device)

            gt_z0 = autoencoder.encode(poses, mask=pose_mask)
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

                # Base loss
                loss_base = F.l1_loss(noise_pred, noise_gt, reduction='none')
                loss_base = (loss_base * pose_mask.unsqueeze(-1).float()).sum() / pose_mask.sum().clamp(min=1)

                # pred_z0
                alphas_cumprod = noise_scheduler.alphas_cumprod.to(timesteps.device)
                alpha_t = torch.sqrt(alphas_cumprod[timesteps]).view(-1,1,1)
                sigma_t = torch.sqrt(1.0 - alphas_cumprod[timesteps]).view(-1,1,1)
                pred_z0 = (z_t - sigma_t * noise_pred) / alpha_t

                # Velocity
                loss_vel = velocity_loss_fn(pred_z0, gt_z0, mask=pose_mask)

                # Recon L1
                loss_recon_l1 = F.l1_loss(pred_z0, gt_z0, reduction='none')
                loss_recon_l1 = (loss_recon_l1 * pose_mask.unsqueeze(-1).float()).sum() / pose_mask.sum().clamp(min=1)

                total_loss = loss_base + W_VELOCITY_LOSS * loss_vel

            total_loss_epoch += total_loss.item()
            total_base_epoch += loss_base.item()
            total_vel_epoch += loss_vel.item()
            total_recon_l1_epoch += loss_recon_l1.item()

            pbar.set_postfix({'loss': f'{total_loss.item():.4f}', 'recon_l1': f'{loss_recon_l1.item():.4f}'})

    avg_loss = total_loss_epoch / len(dataloader)
    avg_base = total_base_epoch / len(dataloader)
    avg_vel = total_vel_epoch / len(dataloader)
    avg_recon_l1 = total_recon_l1_epoch / len(dataloader)

    return {'total': avg_loss, 'base': avg_base, 'vel': avg_vel, 'recon_l1': avg_recon_l1}


# -------------------------- MAIN --------------------------
def main():
    parser = argparse.ArgumentParser(description="Train LDM Teacher Model (Stage 2a) - BERT")
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

    # Autoencoder
    autoencoder = UnifiedPoseAutoencoder(pose_dim=214, latent_dim=args.latent_dim, hidden_dim=args.ae_hidden_dim)
    ae_checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
    if 'model_state_dict' in ae_checkpoint:
        autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
    else:
        autoencoder.load_state_dict(ae_checkpoint)
    autoencoder.to(device).eval().requires_grad_(False)

    # BERT
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    text_encoder = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
    text_encoder.eval().requires_grad_(False)

    # Datasets
    train_dataset = SignLanguageDataset(args.data_dir, split='train', max_seq_len=args.max_seq_len, max_text_len=args.max_text_len, normalize=True)
    val_dataset = SignLanguageDataset(args.data_dir, split='dev', max_seq_len=args.max_seq_len, max_text_len=args.max_text_len, normalize=True)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # LDM
    # === SỬA DÒNG 2 ===
    ldm_model = LDM_Mamba_Denoiser(latent_dim=args.latent_dim, text_embed_dim=args.text_embed_dim,
                                  hidden_dim=args.text_embed_dim, num_layers=args.num_layers, num_heads=12).to(device)
    # ==================

    # Noise scheduler & losses
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    noise_scheduler.config.prediction_type = "epsilon"
    velocity_loss_fn = VelocityLoss(loss_type='l1').to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(ldm_model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    scaler = GradScaler()

    # NULL embeddings
    null_text_input = tokenizer("", padding="max_length", max_length=args.max_text_len, truncation=True, return_tensors="pt").to(device)
    with torch.no_grad():
        null_text_embeddings = text_encoder(null_text_input.input_ids, attention_mask=null_text_input.attention_mask, token_type_ids=null_text_input.token_type_ids).last_hidden_state

    # Resume checkpoint
    start_epoch = 0
    best_val_recon_l1 = float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        ldm_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_recon_l1 = checkpoint.get('best_val_recon_l1', checkpoint.get('val_recon_l1', float('inf')))

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        train_losses = train_epoch(ldm_model, autoencoder, text_encoder, train_loader,
                                   optimizer, scheduler, scaler, noise_scheduler,
                                   velocity_loss_fn, null_text_embeddings, device, epoch)
        val_losses = validate(ldm_model, autoencoder, text_encoder, val_loader,
                              noise_scheduler, velocity_loss_fn, null_text_embeddings, device)

        val_recon_l1 = val_losses['recon_l1']

        print(f"\n=== EPOCH {epoch+1}/{args.num_epochs} ===")
        print(f"Train Loss: {train_losses['total']:.6f} (Base: {train_losses['base']:.6f}, Vel: {train_losses['vel']:.6f}, Recon L1: {train_losses['recon_l1']:.6f})")
        print(f"Val   Loss: {val_losses['total']:.6f} (Base: {val_losses['base']:.6f}, Vel: {val_losses['vel']:.6f}, Recon L1: {val_losses['recon_l1']:.6f}, Best: {best_val_recon_l1:.6f})")
        print(f"LR: {optimizer.param_groups[0]['lr']:.2e}")

        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': ldm_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'train_loss': train_losses,
            'val_loss': val_losses,
            'val_recon_l1': val_recon_l1,
            'best_val_recon_l1': best_val_recon_l1,
            'args': args
        }
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pt'))

        # Save best model
        if val_recon_l1 < best_val_recon_l1:
            best_val_recon_l1 = val_recon_l1
            checkpoint['best_val_recon_l1'] = best_val_recon_l1
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"✅ New best model saved! Recon L1: {best_val_recon_l1:.6f}")

    print(f"\n🎉 TRAINING COMPLETED! Best Val Recon L1: {best_val_recon_l1:.6f}")
    

if __name__ == '__main__':
    main()