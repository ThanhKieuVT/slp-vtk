# Tên file: train_pfm_student.py
# === PHIÊN BẢN CHUẨN: FIX LỖI SHAPE `t` ===

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
from models.ldm_denoiser_mamba import LDM_Mamba_Denoiser
from models.pfm_student import PFM_Student
from models.sampler_ddim import DDIMSampler

from transformers import BertModel, BertTokenizer
from diffusers import DDPMScheduler

GRAD_ACCUMULATION_STEPS = 4
TEACHER_SAMPLING_STEPS = 50
TEACHER_CFG_SCALE = 7.0
VALIDATION_SEED = 42


def train_epoch(teacher_model, student_model, teacher_sampler,
                autoencoder, text_encoder, dataloader,
                optimizer, scheduler, scaler, device, epoch):
    student_model.train()
    teacher_model.eval()
    autoencoder.eval()
    text_encoder.eval()

    total_loss = 0.0
    pbar = tqdm(dataloader, desc=f"Epoch {epoch+1} Distilling")

    for i, batch in enumerate(pbar):
        poses = batch['poses'].to(device)
        pose_mask = batch['pose_mask'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        with torch.no_grad():
            gt_z0 = autoencoder.encode(poses, mask=pose_mask)
            text_kwargs = {'attention_mask': attention_mask, 'token_type_ids': token_type_ids}
            text_embeddings = text_encoder(text_tokens, **text_kwargs).last_hidden_state

        batch_size = gt_z0.shape[0]

        z_1 = torch.randn_like(gt_z0)

        pose_padding_mask = (~pose_mask).to(torch.bool)
        text_padding_mask = (attention_mask == 0).to(torch.bool)

        with torch.no_grad():
            pred_z0_teacher = teacher_sampler.sample(
                z_1, text_embeddings,
                text_mask_bool=text_padding_mask,
                pose_mask_bool=pose_padding_mask,
                num_inference_steps=TEACHER_SAMPLING_STEPS,
                cfg_scale=TEACHER_CFG_SCALE,
                device=device
            )

        assert not torch.isnan(pred_z0_teacher).any(), "Teacher output NaN!"
        assert pred_z0_teacher.shape == z_1.shape, "Teacher output shape mismatch!"

        # === SỬA LỖI SHAPE (1/3) ===
        t = torch.rand(batch_size, device=device) # Shape [B]
        t_view = t.view(-1, 1, 1) # Shape [B, 1, 1]

        # === SỬA LỖI SHAPE (2/3) ===
        z_t = (1 - t_view) * pred_z0_teacher + t_view * z_1
        v_target = z_1 - pred_z0_teacher

        with autocast(dtype=torch.float16):
            # === SỬA LỖI SHAPE (3/3) ===
            v_pred_student = student_model(
                z_t, t, text_embeddings, # Truyền `t` (shape [B])
                text_mask=text_padding_mask,
                pose_mask=pose_padding_mask
            )
            loss = F.l1_loss(v_pred_student, v_target, reduction='none')
            loss = (loss * pose_mask.unsqueeze(-1).float()).sum() / pose_mask.sum().clamp(min=1)
            loss_to_backward = loss / GRAD_ACCUMULATION_STEPS

        scaler.scale(loss_to_backward).backward()

        if (i + 1) % GRAD_ACCUMULATION_STEPS == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(student_model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    scheduler.step()
    return {'total': total_loss / len(dataloader)}


@torch.no_grad()
def validate(teacher_model, student_model, teacher_sampler,
             autoencoder, text_encoder, dataloader, device):
    student_model.eval()
    teacher_model.eval()
    autoencoder.eval()
    text_encoder.eval()

    total_loss = 0.0
    torch.manual_seed(VALIDATION_SEED)
    pbar = tqdm(dataloader, desc="Validating Student")

    for batch in pbar:
        poses = batch['poses'].to(device)
        pose_mask = batch['pose_mask'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        token_type_ids = batch['token_type_ids'].to(device)

        gt_z0 = autoencoder.encode(poses, mask=pose_mask)
        text_embeddings = text_encoder(text_tokens, attention_mask=attention_mask, token_type_ids=token_type_ids).last_hidden_state
        batch_size = gt_z0.shape[0]

        z_1 = torch.randn_like(gt_z0)
        pose_padding_mask = (~pose_mask).to(torch.bool)
        text_padding_mask = (attention_mask == 0).to(torch.bool)

        pred_z0_teacher = teacher_sampler.sample(
            z_1, text_embeddings,
            text_mask_bool=text_padding_mask,
            pose_mask_bool=pose_padding_mask,
            num_inference_steps=TEACHER_SAMPLING_STEPS,
            cfg_scale=TEACHER_CFG_SCALE,
            device=device
        )

        assert not torch.isnan(pred_z0_teacher).any(), "Teacher NaN (Val)"

        # === SỬA LỖI SHAPE (1/3) - VALIDATE ===
        t = torch.rand(batch_size, device=device) # Shape [B]
        t_view = t.view(-1, 1, 1) # Shape [B, 1, 1]

        # === SỬA LỖI SHAPE (2/3) - VALIDATE ===
        z_t = (1 - t_view) * pred_z0_teacher + t_view * z_1
        v_target = z_1 - pred_z0_teacher

        with autocast(dtype=torch.float16):
            # === SỬA LỖI SHAPE (3/3) - VALIDATE ===
            v_pred_student = student_model(
                z_t, t, text_embeddings, # Truyền `t` (shape [B])
                text_mask=text_padding_mask,
                pose_mask=pose_padding_mask
            )
            loss = F.l1_loss(v_pred_student, v_target, reduction='none')
            loss = (loss * pose_mask.unsqueeze(-1).float()).sum() / pose_mask.sum().clamp(min=1)

        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{loss.item():.4f}'})

    return {'total': total_loss / len(dataloader)}


def main():
    parser = argparse.ArgumentParser(description="Train PFM Student (Diff2Flow)")
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True)
    parser.add_argument('--teacher_checkpoint', type=str, required=True)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--max_seq_len', type=int, default=120)
    parser.add_argument('--max_text_len', type=int, default=64)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--ae_hidden_dim', type=int, default=512)
    parser.add_argument('--text_embed_dim', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=6)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🚀 Using device: {device}")

    # Load frozen modules
    autoencoder = UnifiedPoseAutoencoder(pose_dim=214, latent_dim=args.latent_dim, hidden_dim=args.ae_hidden_dim).to(device)
    ae_checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
    autoencoder.load_state_dict(ae_checkpoint.get('model_state_dict', ae_checkpoint))
    autoencoder.eval().requires_grad_(False)

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    text_encoder = BertModel.from_pretrained("bert-base-multilingual-cased").to(device)
    text_encoder.eval().requires_grad_(False)

    teacher_model = LDM_Mamba_Denoiser(
        latent_dim=args.latent_dim, text_embed_dim=args.text_embed_dim,
        hidden_dim=args.text_embed_dim, num_layers=args.num_layers, num_heads=12
    ).to(device)
    teacher_checkpoint = torch.load(args.teacher_checkpoint, map_location=device)
    teacher_model.load_state_dict(teacher_checkpoint['model_state_dict'])
    teacher_model.eval().requires_grad_(False)

    # Teacher sampler
    noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule='squaredcos_cap_v2')
    noise_scheduler.config.prediction_type = "epsilon"
    null_input = tokenizer("", padding="max_length", max_length=args.max_text_len, truncation=True, return_tensors="pt").to(device)
    null_embeddings = text_encoder(null_input.input_ids, attention_mask=null_input.attention_mask, token_type_ids=null_input.token_type_ids).last_hidden_state
    null_mask = (null_input.attention_mask == 0).to(torch.bool)
    teacher_sampler = DDIMSampler(teacher_model, noise_scheduler, null_embeddings, null_mask)

    # Dataloaders
    train_loader = DataLoader(SignLanguageDataset(args.data_dir, split='train', max_seq_len=args.max_seq_len, max_text_len=args.max_text_len, normalize=True),
                              batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(SignLanguageDataset(args.data_dir, split='dev', max_seq_len=args.max_seq_len, max_text_len=args.max_text_len, normalize=True),
                            batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)

    # Student
    student_model = PFM_Student(latent_dim=args.latent_dim, text_embed_dim=args.text_embed_dim,
                                hidden_dim=args.text_embed_dim, num_layers=args.num_layers, num_heads=12).to(device)
    print(f"✅ Student params: {sum(p.numel() for p in student_model.parameters() if p.requires_grad):,}")

    optimizer = torch.optim.AdamW(student_model.parameters(), lr=args.learning_rate)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)
    scaler = GradScaler()

    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        checkpoint = torch.load(args.resume_from, map_location=device)
        student_model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', best_val_loss)
        print(f"📂 Resumed from {args.resume_from} (epoch {start_epoch})")

    # Training loop
    for epoch in range(start_epoch, args.num_epochs):
        train_losses = train_epoch(teacher_model, student_model, teacher_sampler, autoencoder, text_encoder, train_loader,
                                   optimizer, scheduler, scaler, device, epoch)
        val_losses = validate(teacher_model, student_model, teacher_sampler, autoencoder, text_encoder, val_loader, device)
        val_loss = val_losses['total']

        print(f"\nEpoch {epoch+1}: Train Loss = {train_losses['total']:.6f}, Val Loss = {val_loss:.6f} (Best {best_val_loss:.6f})")

        checkpoint = {
            'epoch': epoch,
            'model_state_dict': student_model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'args': args
        }
        torch.save(checkpoint, os.path.join(args.output_dir, 'student_latest.pt'))

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, 'student_best_model.pt'))
            print(f"✅ New best student saved! Val Loss: {best_val_loss:.6f}")

    print(f"\n🎉 Student distillation completed! Best val loss: {best_val_loss:.6f}")


if __name__ == '__main__':
    main()