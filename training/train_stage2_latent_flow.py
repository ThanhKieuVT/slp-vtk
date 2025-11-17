"""
Training Script: Stage 2 Latent Flow Matching (Transformer Backbone)
Fixed by Gemini: Matches exact CLI arguments provided by User
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# Giả định các module này nằm trong PYTHONPATH như lệnh chị chạy
from dataset import SignLanguageDataset, collate_fn
from models.fml.autoencoder import UnifiedPoseAutoencoder
from models.fml.latent_flow_matcher import LatentFlowMatcher

# === 1. HÀM TÍNH SCALE FACTOR (QUAN TRỌNG ĐỂ LOSS KHÔNG NỔ) ===
def estimate_scale_factor(encoder, dataloader, device, num_batches=20):
    print(f"⏳ Đang tính toán Latent Scale Factor ({num_batches} batches)...")
    encoder.eval()
    all_latents = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            poses = batch['poses'].to(device)
            pose_mask = batch['pose_mask'].to(device)
            # Encode
            z = encoder(poses, mask=pose_mask)
            all_latents.append(z.cpu())
    
    all_latents = torch.cat(all_latents, dim=0) # [N, T, D]
    std = all_latents.std()
    # Tránh chia cho 0
    scale_factor = 1.0 / (std.item() + 1e-6)
    
    print(f"✅ Latent Std gốc: {std.item():.4f}")
    print(f"✅ Scale Factor tự động tính: {scale_factor:.6f}")
    return scale_factor

# === 2. HÀM TRAIN ===
def train_epoch(flow_matcher, encoder, dataloader, optimizer, device, epoch, scale_factor):
    flow_matcher.train()
    encoder.eval()
    
    total_loss = 0.0
    losses_log = {'flow': 0.0, 'prior': 0.0, 'sync': 0.0}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        poses = batch['poses'].to(device)
        pose_mask = batch['pose_mask'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        seq_lengths = batch['seq_lengths'].to(device)
        
        batch_dict = {
            'text_tokens': text_tokens,
            'attention_mask': attention_mask,
            'seq_lengths': seq_lengths,
            'target_length': seq_lengths
        }
        
        # Step 1: Encode & Scale
        with torch.no_grad():
            gt_latent = encoder(poses, mask=pose_mask)
            gt_latent = gt_latent * scale_factor  # Scale về std=1

        # Step 2: Flow Matching
        # Lưu ý: mode='train' để tính loss
        losses = flow_matcher(
            batch_dict,
            gt_latent=gt_latent,
            pose_gt=poses, 
            mode='train'
        )
        
        # Step 3: Backward
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(flow_matcher.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Logging
        total_loss += losses['total'].item()
        losses_log['flow'] += losses['flow'].item()
        if 'prior' in losses: losses_log['prior'] += losses['prior'].item()
        if 'sync' in losses: losses_log['sync'] += losses['sync'].item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'flow': f"{losses['flow'].item():.4f}"
        })
    
    return {k: v / num_batches for k, v in losses_log.items()}, total_loss / num_batches

# === 3. HÀM VALIDATE ===
def validate(flow_matcher, encoder, dataloader, device, scale_factor):
    flow_matcher.eval()
    encoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc="Validating")
    for batch in pbar:
        poses = batch['poses'].to(device)
        pose_mask = batch['pose_mask'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        seq_lengths = batch['seq_lengths'].to(device)
        
        batch_dict = {'text_tokens': text_tokens, 'attention_mask': attention_mask, 'seq_lengths': seq_lengths}
        
        with torch.no_grad():
            gt_latent = encoder(poses, mask=pose_mask)
            gt_latent = gt_latent * scale_factor 

        # Enable grad tạm thời nếu loss function yêu cầu (Sync loss đôi khi cần)
        # Nhưng không gọi backward()
        with torch.set_grad_enabled(True): 
            losses = flow_matcher(
                batch_dict,
                gt_latent=gt_latent,
                pose_gt=poses,
                mode='train'
            )
            
        total_loss += losses['total'].item()
        num_batches += 1
            
    return total_loss / num_batches if num_batches > 0 else 0.0

# === 4. MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True)
    
    # Training Params
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    
    # Model Params
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=120)
    
    # === CÁC THAM SỐ CHỊ ĐÃ THÊM TRONG LỆNH CHẠY (QUAN TRỌNG) ===
    parser.add_argument('--ae_hidden_dim', type=int, default=512, 
                        help="Hidden dim của Autoencoder (để load checkpoint đúng)")
    parser.add_argument('--text_embed_dim', type=int, default=768, 
                        help="Kích thước embedding của BERT/Text Encoder")
    # ============================================================

    # Tham số cho Flow Matcher (có thể chị không truyền nhưng code cần default)
    parser.add_argument('--hidden_dim', type=int, default=512, help="Hidden dim của Flow Model")
    parser.add_argument('--use_ssm_prior', action='store_true', help="Dùng Mamba (Mặc định False)")
    parser.add_argument('--use_sync_guidance', action='store_true', help="Dùng Sync Loss (Mặc định False)")
    
    # Các tham số phụ khác
    parser.add_argument('--lambda_prior', type=float, default=0.0)
    parser.add_argument('--gamma_guidance', type=float, default=0.0)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--resume_from', type=str, default=None)

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Bắt đầu training Stage 2: Flow Matching (Transformer Backbone)")
    print(f"📂 Output dir: {args.output_dir}")

    # --- 1. Load Autoencoder (Frozen) ---
    print(f"🔄 Loading autoencoder from {args.autoencoder_checkpoint}...")
    # Sử dụng args.ae_hidden_dim mà lệnh chạy đã truyền vào
    autoencoder = UnifiedPoseAutoencoder(pose_dim=214, latent_dim=args.latent_dim, hidden_dim=args.ae_hidden_dim)
    
    checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
    else: state_dict = checkpoint
    autoencoder.load_state_dict(state_dict)
    
    encoder = autoencoder.encoder.to(device).eval().requires_grad_(False)
    print("✅ Autoencoder loaded & frozen.")

    # --- 2. Dataset ---
    print("📚 Loading datasets...")
    train_dataset = SignLanguageDataset(args.data_dir, split='train', max_seq_len=args.max_seq_len)
    val_dataset = SignLanguageDataset(args.data_dir, split='dev', max_seq_len=args.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    # --- 3. Calculate Scale Factor (BẮT BUỘC) ---
    latent_scale_factor = estimate_scale_factor(encoder, train_loader, device)

    # --- 4. Init Flow Matcher Model ---
    print("🔧 Initializing LatentFlowMatcher...")
    # Lưu ý: Nếu lệnh chạy không có --hidden_dim, nó sẽ lấy default 512
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim, # Dùng hidden_dim của model train (khác ae_hidden_dim)
        use_ssm_prior=args.use_ssm_prior, # Mặc định False (Transformer)
        use_sync_guidance=args.use_sync_guidance
    ).to(device)
    
    optimizer = torch.optim.AdamW(flow_matcher.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # --- 5. Resume Logic ---
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"♻️ Resuming from {args.resume_from}...")
        ckpt = torch.load(args.resume_from, map_location=device)
        flow_matcher.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        latent_scale_factor = ckpt.get('latent_scale_factor', latent_scale_factor)
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"✅ Resumed. Epoch: {start_epoch}, Scale Factor: {latent_scale_factor}")

    # --- 6. Training Loop ---
    for epoch in range(start_epoch, args.num_epochs):
        train_metrics, avg_train_loss = train_epoch(flow_matcher, encoder, train_loader, optimizer, device, epoch, latent_scale_factor)
        val_loss = validate(flow_matcher, encoder, val_loader, device, latent_scale_factor)
        scheduler.step()
        
        print(f"\n=== Epoch {epoch+1}/{args.num_epochs} ===")
        print(f"📉 Train Loss: {avg_train_loss:.6f} (Flow: {train_metrics['flow']:.4f})")
        print(f"📉 Val Loss:   {val_loss:.6f} (Best: {best_val_loss:.6f})")
        
        # Save Checkpoints
        save_dict = {
            'epoch': epoch,
            'model_state_dict': flow_matcher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'latent_scale_factor': latent_scale_factor, # Lưu lại để dùng khi Inference
            'best_val_loss': best_val_loss,
            'args': args
        }
        torch.save(save_dict, os.path.join(args.output_dir, 'latest.pt'))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict['best_val_loss'] = best_val_loss
            torch.save(save_dict, os.path.join(args.output_dir, 'best_model.pt'))
            print("🏆 New Best Model Saved!")

if __name__ == '__main__':
    main()