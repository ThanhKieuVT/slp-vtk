"""
Training Script: Stage 2 Latent Flow Matching (Direct Training)
Fixed by Gemini: 
- Added 'sys.path' fix for module imports
- Added '--num_layers' argument to fix CLI error
- Auto-calculates latent scale factor
- FIXED RESUME LOGIC (Load best_val_loss & Scheduler)
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

# --- FIX PATH ---
sys.path.append(os.getcwd()) 
# ----------------

try:
    from dataset import SignLanguageDataset, collate_fn
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit(1)

# === 1. HÀM TÍNH SCALE FACTOR ===
def estimate_scale_factor(encoder, dataloader, device, num_batches=20):
    print(f"⏳ Đang tính toán Latent Scale Factor ({num_batches} batches)...")
    encoder.eval()
    all_latents = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            poses = batch['poses'].to(device)
            pose_mask = batch['pose_mask'].to(device)
            z = encoder(poses, mask=pose_mask)
            all_latents.append(z.cpu())
    
    all_latents = torch.cat(all_latents, dim=0)
    std = all_latents.std()
    scale_factor = 1.0 / (std.item() + 1e-6)
    
    print(f"✅ Latent Std gốc: {std.item():.4f}")
    print(f"✅ Scale Factor tự động tính: {scale_factor:.6f}")
    return scale_factor

# === 2. HÀM TRAIN ===
def train_epoch(flow_matcher, encoder, dataloader, optimizer, device, epoch, scale_factor):
    flow_matcher.train()
    encoder.eval()
    
    total_loss = 0.0
    losses_log = {'flow': 0.0}
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
        
        with torch.no_grad():
            gt_latent = encoder(poses, mask=pose_mask)
            gt_latent = gt_latent * scale_factor 

        losses = flow_matcher(batch_dict, gt_latent=gt_latent, pose_gt=poses, mode='train')
        
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(flow_matcher.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += losses['total'].item()
        losses_log['flow'] += losses['flow'].item()
        num_batches += 1
        
        pbar.set_postfix({'loss': f"{losses['total'].item():.4f}"})
    
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

        with torch.set_grad_enabled(True): 
            losses = flow_matcher(batch_dict, gt_latent=gt_latent, pose_gt=poses, mode='train')
            
        total_loss += losses['total'].item()
        num_batches += 1
            
    return total_loss / num_batches if num_batches > 0 else 0.0

# === 4. MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=200)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=2)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--max_seq_len', type=int, default=120)
    parser.add_argument('--ae_hidden_dim', type=int, default=512)
    parser.add_argument('--text_embed_dim', type=int, default=768)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--use_ssm_prior', action='store_true')
    parser.add_argument('--use_sync_guidance', action='store_true')
    
    # Extra args
    parser.add_argument('--lambda_prior', type=float, default=0.0)
    parser.add_argument('--gamma_guidance', type=float, default=0.0)
    parser.add_argument('--num_inference_steps', type=int, default=50)
    parser.add_argument('--resume_from', type=str, default=None)

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.backends.mps.is_available() and torch.backends.mps.is_built():
        print("🍎 Phát hiện chip Apple Silicon (MPS).")
        # device = torch.device('mps') # Uncomment nếu muốn dùng MPS
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Bắt đầu training Flow Matching trên: {device}")

    # 1. Load Autoencoder
    print("🔄 Loading Autoencoder...")
    autoencoder = UnifiedPoseAutoencoder(pose_dim=214, latent_dim=args.latent_dim, hidden_dim=args.ae_hidden_dim)
    checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
    if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
    else: state_dict = checkpoint
    autoencoder.load_state_dict(state_dict)
    encoder = autoencoder.encoder.to(device).eval().requires_grad_(False)

    # 2. Dataset
    print("📚 Loading Dataset...")
    train_dataset = SignLanguageDataset(args.data_dir, split='train', max_seq_len=args.max_seq_len)
    val_dataset = SignLanguageDataset(args.data_dir, split='dev', max_seq_len=args.max_seq_len)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn)
    
    # 3. Scale Factor
    latent_scale_factor = estimate_scale_factor(encoder, train_loader, device)

    # 4. Init Model
    print("🔧 Init Flow Matcher...")
    try:
        flow_matcher = LatentFlowMatcher(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            use_ssm_prior=args.use_ssm_prior,
            use_sync_guidance=args.use_sync_guidance,
            num_layers=args.num_layers
        ).to(device)
    except TypeError:
        print("⚠️ Model không nhận 'num_layers'. Dùng default...")
        flow_matcher = LatentFlowMatcher(
            latent_dim=args.latent_dim,
            hidden_dim=args.hidden_dim,
            use_ssm_prior=args.use_ssm_prior,
            use_sync_guidance=args.use_sync_guidance
        ).to(device)
    
    optimizer = torch.optim.AdamW(flow_matcher.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # === 5. RESUME LOGIC (SAFE FIX) ===
    start_epoch = 0
    best_val_loss = float('inf')

    # Tự động tìm latest.pt nếu không truyền resume_from
    if args.resume_from is None:
        potential_latest = os.path.join(args.output_dir, 'latest.pt')
        if os.path.exists(potential_latest):
            args.resume_from = potential_latest
            print(f"🔍 Tự động tìm thấy checkpoint: {args.resume_from}")

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"♻️ Resuming from {args.resume_from}...")
        ckpt = torch.load(args.resume_from, map_location=device)
        
        # 1. Load Model (Bắt buộc)
        flow_matcher.load_state_dict(ckpt['model_state_dict'])
        print("✅ Loaded Model weights.")

        # 2. Load Optimizer (An toàn: Có thì load, không thì thôi)
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("✅ Loaded Optimizer state.")
            except Exception as e:
                print(f"⚠️ Không thể load Optimizer (Lỗi version hoặc params). Sẽ khởi tạo mới. Lỗi: {e}")
        else:
            print("⚠️ Checkpoint cũ thiếu 'optimizer_state_dict'. Optimizer sẽ chạy lại từ đầu.")
        
        # 3. Load Scheduler (An toàn)
        if 'scheduler_state_dict' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            
        # 4. Load Epoch & Best Loss
        # Nếu checkpoint cũ không có key 'epoch', mặc định là 0
        if 'epoch' in ckpt:
            start_epoch = ckpt['epoch'] + 1
        else:
            print("⚠️ Checkpoint không có thông tin epoch. Bắt đầu từ epoch 0.")

        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        latent_scale_factor = ckpt.get('latent_scale_factor', latent_scale_factor)
        
        print(f"⏩ Resuming at Epoch: {start_epoch}")
        print(f"ℹ️ Current Best Val Loss: {best_val_loss:.4f}")
        print(f"ℹ️ Scale Factor: {latent_scale_factor:.6f}")

    # 6. Loop
    for epoch in range(start_epoch, args.num_epochs):
        train_metrics, avg_train_loss = train_epoch(flow_matcher, encoder, train_loader, optimizer, device, epoch, latent_scale_factor)
        val_loss = validate(flow_matcher, encoder, val_loader, device, latent_scale_factor)
        scheduler.step()
        
        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.6f} | Val: {val_loss:.6f}")
        
        # Save Latest (Kèm scheduler & best_loss)
        save_dict = {
            'epoch': epoch,
            'model_state_dict': flow_matcher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(), # Đã thêm dòng này
            'latent_scale_factor': latent_scale_factor,
            'best_val_loss': best_val_loss
        }
        torch.save(save_dict, os.path.join(args.output_dir, 'latest.pt'))
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Cập nhật best_val_loss vào save_dict trước khi lưu best_model
            save_dict['best_val_loss'] = best_val_loss 
            torch.save(save_dict, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"🏆 New Best Model Saved! (Loss: {best_val_loss:.4f})")

if __name__ == '__main__':
    main()