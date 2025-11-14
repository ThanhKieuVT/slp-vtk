# Tên file: train_ldm_teacher.py
# === PHIÊN BẢN CHUẨN: BERT (768-dim) ===

import os
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.cuda.amp import GradScaler, autocast
from tqdm import tqdm

# --- Import các model và data của chị ---
from utils.data_loader import SignLanguageDataset, collate_fn
from models.autoencoder import UnifiedPoseAutoencoder # (Đã sửa tên)
from models.ldm_denoiser import LDM_TransformerDenoiser
from models.losses import VelocityLoss

# --- Import các thư viện SOTA ---
from transformers import BertModel, BertTokenizer # <<< DÙNG BERT
from diffusers import DDPMScheduler

# --- Cấu hình các "Bí kíp" SOTA ---
CFG_PROBABILITY = 0.1
GRAD_ACCUMULATION_STEPS = 4
W_VELOCITY_LOSS = 0.05

def train_epoch(
    ldm_model, autoencoder, text_encoder, tokenizer,
    dataloader, optimizer, scheduler, scaler,
    noise_scheduler, velocity_loss_fn, device, epoch
):
    ldm_model.train()
    text_encoder.eval()
    autoencoder.eval()
    total_loss_epoch, total_base_loss_epoch, total_vel_loss_epoch = 0.0, 0.0, 0.0
    
    null_text_input = tokenizer(
        "", padding="max_length", 
        max_length=tokenizer.model_max_length, 
        truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        null_text_embeddings = text_encoder(
            null_text_input.input_ids,
            attention_mask=null_text_input.attention_mask
        ).last_hidden_state
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} Training")
    for i, batch in enumerate(pbar):
        poses = batch['poses'].to(device)
        pose_mask = batch['pose_mask'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        with torch.no_grad():
            gt_z0 = autoencoder.encode(poses, mask=pose_mask)
        batch_size = gt_z0.shape[0]

        with torch.no_grad():
            text_embeddings = text_encoder(
                text_tokens,
                attention_mask=attention_mask
            ).last_hidden_state

        mask_cfg = torch.rand(batch_size, device=device) < CFG_PROBABILITY
        text_embeddings[mask_cfg] = null_text_embeddings
        
        noise_gt = torch.randn_like(gt_z0)
        timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                (batch_size,), device=device).long()
        z_t = noise_scheduler.add_noise(gt_z0, noise_gt, timesteps)
        
        with autocast(device_type='cuda', dtype=torch.float16):
            pose_padding_mask = ~pose_mask
            text_padding_mask = ~attention_mask.bool()

            noise_pred = ldm_model(
                z_t, timesteps, text_embeddings,
                text_mask=text_padding_mask,
                pose_mask=pose_padding_mask
            )
            
            loss_base = F.l1_loss(noise_pred, noise_gt, reduction='none')
            loss_base = loss_base * pose_mask.unsqueeze(-1).float()
            loss_base = loss_base.sum() / pose_mask.sum().clamp(min=1)
            
            pred_z0 = noise_scheduler.get_original_sample(z_t, timesteps, noise_pred)
            loss_vel = velocity_loss_fn(pred_z0, gt_z0, mask=pose_mask)
            
            total_loss = loss_base + W_VELOCITY_LOSS * loss_vel
            loss_to_backward = total_loss / GRAD_ACCUMULATION_STEPS
            
        scaler.scale(loss_to_backward).backward()

        if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(ldm_model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        total_loss_epoch += total_loss.item()
        total_base_loss_epoch += loss_base.item()
        total_vel_loss_epoch += loss_vel.item()
        pbar.set_postfix({'loss': total_loss.item()})

    avg_loss = total_loss_epoch / len(dataloader)
    avg_base = total_base_loss_epoch / len(dataloader)
    avg_vel = total_vel_loss_epoch / len(dataloader)
    scheduler.step()
    return {'total': avg_loss, 'base': avg_base, 'vel': avg_vel}

def validate(
    ldm_model, autoencoder, text_encoder, tokenizer,
    dataloader, noise_scheduler, velocity_loss_fn, device
):
    ldm_model.eval()
    total_loss_epoch, total_base_loss_epoch, total_vel_loss_epoch = 0, 0, 0
    pbar = tqdm(dataloader, desc="Validating")
    with torch.no_grad():
        for i, batch in enumerate(pbar):
            poses = batch['poses'].to(device)
            pose_mask = batch['pose_mask'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            gt_z0 = autoencoder.encode(poses, mask=pose_mask)
            batch_size = gt_z0.shape[0]

            text_embeddings = text_encoder(
                text_tokens,
                attention_mask=attention_mask
            ).last_hidden_state

            noise_gt = torch.randn_like(gt_z0)
            timesteps = torch.randint(0, noise_scheduler.config.num_train_timesteps, 
                                    (batch_size,), device=device).long()
            z_t = noise_scheduler.add_noise(gt_z0, noise_gt, timesteps)
            
            with autocast(device_type='cuda', dtype=torch.float16):
                pose_padding_mask = ~pose_mask
                text_padding_mask = ~attention_mask.bool()

                noise_pred = ldm_model(
                    z_t, timesteps, text_embeddings,
                    text_mask=text_padding_mask,
                    pose_mask=pose_padding_mask
                )
                
                loss_base = F.l1_loss(noise_pred, noise_gt, reduction='none')
                loss_base = loss_base * pose_mask.unsqueeze(-1).float()
                loss_base = loss_base.sum() / pose_mask.sum().clamp(min=1)
                
                pred_z0 = noise_scheduler.get_original_sample(z_t, timesteps, noise_pred)
                loss_vel = velocity_loss_fn(pred_z0, gt_z0, mask=pose_mask)
                total_loss = loss_base + W_VELOCITY_LOSS * loss_vel
            
            total_loss_epoch += total_loss.item()
            total_base_loss_epoch += loss_base.item()
            total_vel_loss_epoch += loss_vel.item()

    avg_loss = total_loss_epoch / len(dataloader)
    avg_base = total_base_loss_epoch / len(dataloader)
    avg_vel = total_vel_loss_epoch / len(dataloader)
    return {'total': avg_loss, 'base': avg_base, 'vel': avg_vel}

def main():
    parser = argparse.ArgumentParser(description="Train LDM Teacher Model (Stage 2a) - BERT ver.")
    
    # --- Paths ---
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True)
    parser.add_argument('--resume_from', type=str, default=None)

    # --- Training Params ---
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    
    # --- Model Params ---
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--ae_hidden_dim', type=int, default=512, help='Hidden dim của Autoencoder (khớp Stage 1)')
    parser.add_argument('--text_embed_dim', type=int, default=768, help='Text embed dim (768 cho BERT-base)')
    
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--max_seq_len', type=int, default=120)
    parser.add_argument('--max_text_len', type=int, default=64)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- 1. Tải các model đã đóng băng (Frozen) ---
    print("Loading Stage 1 Autoencoder...")
    try:
        autoencoder = UnifiedPoseAutoencoder( # <<< ĐÃ SỬA TÊN
            pose_dim=214,
            latent_dim=args.latent_dim,
            hidden_dim=args.ae_hidden_dim
        )
        ae_checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
        if 'model_state_dict' in ae_checkpoint:
            autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
        else:
            autoencoder.load_state_dict(ae_checkpoint)
        autoencoder.to(device).eval().requires_grad_(False)
    except Exception as e:
        print(f"Lỗi khi load UnifiedPoseAutoencoder: {e}")
        return

    # === Tải BERT và Tokenizer ===
    print("Loading BERT Text Encoder...")
    bert_name = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    text_encoder = BertModel.from_pretrained(bert_name).to(device)
    text_encoder.eval().requires_grad_(False)
    
    # --- 2. Tải Dataloader ---
    print("Loading datasets...")
    train_dataset = SignLanguageDataset(
        data_dir=args.data_dir, split='train', 
        max_seq_len=args.max_seq_len, max_text_len=args.max_text_len
    )
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )
    
    val_dataset = SignLanguageDataset(
        data_dir=args.data_dir, split='dev', 
        max_seq_len=args.max_seq_len, max_text_len=args.max_text_len,
        stats_path=os.path.join(args.data_dir, "normalization_stats.npz")
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # --- 3. Model chính (LDM Denoiser) ---
    print("Initializing LDM Denoiser model...")
    ldm_model = LDM_TransformerDenoiser(
        latent_dim=args.latent_dim,
        text_embed_dim=args.text_embed_dim, # 768
        hidden_dim=args.text_embed_dim,    # 768
        num_layers=args.num_layers,
        num_heads=12 # (BERT-base dùng 12 heads)
    ).to(device)
    
    # --- 4. Schedulers và Loss ---
    noise_scheduler = DDPMScheduler(
        num_train_timesteps=1000,
        beta_schedule='squaredcos_cap_v2'
    )
    velocity_loss_fn = VelocityLoss(loss_type='l1').to(device)
    
    # --- 5. Optimizer và Hỗ trợ Training ---
    optimizer = torch.optim.AdamW(ldm_model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    scaler = GradScaler()
    
    # --- 6. LOGIC RESUME ---
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        ldm_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}, Best Val Loss: {best_val_loss:.6f}")

    # --- VÒNG LẶP HUẤN LUYỆN CHÍNH ---
    print(f"Starting LDM Teacher training from epoch {start_epoch}...")
    for epoch in range(start_epoch, args.num_epochs):
        
        train_losses = train_epoch(
            ldm_model, autoencoder, text_encoder, tokenizer,
            train_loader, optimizer, scheduler, scaler,
            noise_scheduler, velocity_loss_fn, device, epoch
        )
        
        val_losses = validate(
            ldm_model, autoencoder, text_encoder, tokenizer,
            val_loader, noise_scheduler, velocity_loss_fn, device
        )
        val_loss = val_losses['total']

        print(f"\n--- Epoch {epoch+1}/{args.num_epochs} Summary ---")
        print(f"  Train Loss: {train_losses['total']:.6f} (Base: {train_losses['base']:.6f}, Vel: {train_losses['vel']:.6f})")
        print(f"  Valid Loss: {val_losses['total']:.6f} (Base: {val_losses['base']:.6f}, Vel: {val_losses['vel']:.6f})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': ldm_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'args': args
        }
        
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pt'))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✅ Saved new best model (Val Loss: {best_val_loss:.6f})")

    print("\n✅ Training completed!")
    print(f"Best validation loss: {best_val_loss:.6f}")

if __name__ == '__main__':
    main()