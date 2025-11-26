"""
Training Script: Stage 2 Latent Flow Matching (FINAL SOTA VERSION)
FIXED: Scheduler Logic, Loss Weights, Scale Factor Estimation, Robust Resume.
"""
import os
import sys
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts 

# Đảm bảo python tìm thấy các module
sys.path.append(os.getcwd()) 

try:
    from dataset import SignLanguageDataset, collate_fn
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher 
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("💡 Hãy đảm bảo bạn đang chạy từ thư mục gốc chứa 'models' và 'dataset.py'")
    sys.exit(1)

# === 1. HÀM TÍNH SCALE FACTOR (ĐÃ FIX: Per-dimension Std) ===
def estimate_scale_factor(encoder, dataloader, device, num_batches=30):
    print(f"⏳ Đang tính toán Latent Scale Factor ({num_batches} batches)...")
    encoder.eval()
    all_latents = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if i >= num_batches: break
            poses = batch['poses'].to(device)
            pose_mask = batch['pose_mask'].to(device)
            
            # Encode
            z = encoder(poses, mask=pose_mask) # [B, T, D]
            
            # Chỉ lấy phần valid (bỏ padding)
            z_flat = z[pose_mask] # [N_valid, D]
            all_latents.append(z_flat.cpu())
    
    if len(all_latents) > 0:
        all_latents = torch.cat(all_latents, dim=0) # [Total_N, D]
        
        # Tính std cho từng chiều feature (dim=0) sau đó lấy mean
        # Cách này tốt hơn là tính std gộp toàn bộ tensor
        std_per_dim = all_latents.std(dim=0)
        mean_std = std_per_dim.mean().item()
        
        # Công thức scale
        scale_factor = 1.0 / (mean_std + 1e-6)
        
        # Boost nhẹ tín hiệu (x1.2 hoặc x1.5) để tránh pose bị "nhát" (biên độ nhỏ)
        scale_factor *= 1.2 
    else:
        print("⚠️ Cảnh báo: Không lấy được latent mẫu. Dùng scale=1.0")
        mean_std = 1.0
        scale_factor = 1.0
    
    print(f"✅ Mean Latent Std: {mean_std:.6f}")
    print(f"✅ Scale Factor chốt: {scale_factor:.6f}")
    return scale_factor

# === 2. HÀM TRAIN (ĐÃ FIX: Bỏ Scheduler Step) ===
def train_epoch(flow_matcher, encoder, dataloader, optimizer, device, epoch, scale_factor, log_attn_freq=100):
    flow_matcher.train()
    encoder.eval()
    
    total_loss = 0.0
    losses_log = {'flow': 0.0, 'sync': 0.0, 'length': 0.0}
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch_idx, batch in enumerate(pbar):
        # Prepare Data
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
        
        # Get Ground Truth Latent
        with torch.no_grad():
            gt_latent = encoder(poses, mask=pose_mask)
            gt_latent = gt_latent * scale_factor # Scale lên

        return_attn = (batch_idx % log_attn_freq == 0)
        
        # Forward Pass
        losses = flow_matcher(
            batch_dict, 
            gt_latent=gt_latent, 
            pose_gt=poses, # Truyền pose gốc cho Sync Loss (Contrastive)
            mode='train', 
            return_attn_weights=return_attn
        )
        
        # Backward Pass
        optimizer.zero_grad()
        losses['total'].backward()
        
        # Gradient Clipping (Chống nổ gradient)
        torch.nn.utils.clip_grad_norm_(flow_matcher.parameters(), max_norm=1.0)
        
        optimizer.step()
        # ❌ ĐÃ BỎ: scheduler.step() (Sẽ gọi sau epoch)
        
        # Logging
        total_loss += losses['total'].item()
        losses_log['flow'] += losses['flow'].item()
        losses_log['sync'] += losses.get('sync', torch.tensor(0.0)).item()
        losses_log['length'] += losses.get('length', torch.tensor(0.0)).item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}", 
            'flow': f"{losses['flow'].item():.4f}",
            'sync': f"{losses.get('sync', torch.tensor(0.0)).item():.4f}",
            'len': f"{losses.get('length', torch.tensor(0.0)).item():.4f}",
            'lr': f"{optimizer.param_groups[0]['lr']:.2e}"
        })
    
    denom = num_batches if num_batches > 0 else 1
    return {k: v / denom for k, v in losses_log.items()}, total_loss / denom

# === 3. HÀM VALIDATE ===
def validate(flow_matcher, encoder, dataloader, device, scale_factor):
    flow_matcher.eval()
    encoder.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            poses = batch['poses'].to(device)
            pose_mask = batch['pose_mask'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            
            batch_dict = {'text_tokens': text_tokens, 'attention_mask': attention_mask, 'seq_lengths': seq_lengths}
            
            gt_latent = encoder(poses, mask=pose_mask) * scale_factor 

            # Vẫn dùng mode='train' để tính loss validation
            losses = flow_matcher(batch_dict, gt_latent=gt_latent, pose_gt=poses, mode='train')
            
            total_loss += losses['total'].item()
            num_batches += 1
            
    return total_loss / (num_batches if num_batches > 0 else 1)

# === 4. MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True, help="Thư mục chứa processed_data/data")
    parser.add_argument('--output_dir', type=str, required=True, help="Nơi lưu checkpoints")
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True)
    
    # Training Params
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=200) 
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    
    # Model Config
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512) # Hidden dim của Flow Matcher
    parser.add_argument('--ae_hidden_dim', type=int, default=512) # Hidden dim của Autoencoder
    parser.add_argument('--max_seq_len', type=int, default=120)
    parser.add_argument('--num_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    parser.add_argument('--dropout', type=float, default=0.1)
    
    # Feature Flags
    parser.add_argument('--use_ssm_prior', action='store_true', default=True) 
    parser.add_argument('--use_sync_guidance', action='store_true', default=True) 
    
    # Loss Weights (ĐÃ FIX MẶC ĐỊNH CHUẨN)
    parser.add_argument('--W_PRIOR', type=float, default=0.1, help="Weight for Prior Reg Loss")
    parser.add_argument('--W_SYNC', type=float, default=0.1, help="Weight for Sync Loss (Contrastive, nên nhỏ)")
    parser.add_argument('--W_LENGTH', type=float, default=1.0, help="Weight for Length Loss (Đã normalize nên để 1.0)")
    
    # Inference Params (cho model init)
    parser.add_argument('--lambda_prior', type=float, default=0.1)
    parser.add_argument('--gamma_guidance', type=float, default=0.1)
    
    # Misc
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--log_attn_freq', type=int, default=100) 
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Bắt đầu training SOTA Flow Matching trên: {device}")

    # 1. Load Autoencoder
    print("🔄 Loading Autoencoder...")
    # Lưu ý: dùng args.ae_hidden_dim cho AE
    autoencoder = UnifiedPoseAutoencoder(pose_dim=214, latent_dim=args.latent_dim, hidden_dim=args.ae_hidden_dim)
    try:
        checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
        if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
        else: state_dict = checkpoint
        autoencoder.load_state_dict(state_dict, strict=False) # strict=False cho an toàn
        print("✅ Autoencoder loaded.")
    except Exception as e:
        print(f"❌ Lỗi load Autoencoder: {e}")
        sys.exit(1)
        
    encoder = autoencoder.encoder.to(device).eval().requires_grad_(False)

    # 2. Dataset
    print("📚 Loading Dataset...")
    train_dataset = SignLanguageDataset(args.data_dir, split='train', max_seq_len=args.max_seq_len)
    val_dataset = SignLanguageDataset(args.data_dir, split='dev', max_seq_len=args.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, collate_fn=collate_fn, pin_memory=True)
    
    # 3. Scale Factor
    latent_scale_factor = estimate_scale_factor(encoder, train_loader, device)

    # 4. Init Model
    print("🔧 Init SOTA Flow Matcher...")
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_flow_layers=args.num_layers,
        num_prior_layers=args.num_layers,
        num_heads=args.num_heads,
        dropout=args.dropout,
        use_ssm_prior=args.use_ssm_prior,
        use_sync_guidance=args.use_sync_guidance,
        lambda_prior=args.lambda_prior,
        gamma_guidance=args.gamma_guidance,
        lambda_anneal=True,
        W_PRIOR=args.W_PRIOR,
        W_SYNC=args.W_SYNC,
        W_LENGTH=args.W_LENGTH
    ).to(device)
    
    # 5. Optimizer & Scheduler
    optimizer = torch.optim.AdamW(flow_matcher.parameters(), lr=args.learning_rate, weight_decay=0.01)
    
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=20, 
        T_mult=2,
        eta_min=1e-6
    )
    
    # 6. RESUME LOGIC (ROBUST VERSION)
    start_epoch = 0
    best_val_loss = float('inf')

    # Tự động tìm latest.pt
    if args.resume_from is None:
        potential_latest = os.path.join(args.output_dir, 'latest.pt')
        if os.path.exists(potential_latest):
            args.resume_from = potential_latest
            print(f"🔍 Tự động tìm thấy checkpoint: {args.resume_from}")

    if args.resume_from and os.path.exists(args.resume_from):
        print(f"♻️ Resuming from {args.resume_from}...")
        ckpt = torch.load(args.resume_from, map_location=device)
        
        # Load Model
        try:
            flow_matcher.load_state_dict(ckpt['model_state_dict'])
            print("✅ Model weights loaded.")
        except Exception as e:
            print(f"❌ Lỗi load model weights: {e}")
            sys.exit(1)

        # Load Optimizer (Try-catch)
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("✅ Optimizer state loaded.")
            except Exception as e:
                print(f"⚠️ Không load được Optimizer (Lỗi: {e}). Sẽ dùng Optimizer mới.")
        
        # Load Scheduler (Try-catch)
        if 'scheduler_state_dict' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                print("✅ Scheduler state loaded.")
            except Exception as e:
                 print(f"⚠️ Không load được Scheduler (Lỗi: {e}). Sẽ dùng Scheduler mới.")

        start_epoch = ckpt.get('epoch', -1) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        # Ưu tiên dùng scale factor trong checkpoint để đồng bộ
        latent_scale_factor = ckpt.get('latent_scale_factor', latent_scale_factor)
        
        print(f"⏩ Resuming at Epoch: {start_epoch}")
        print(f"ℹ️ Best Val Loss: {best_val_loss:.4f}")
        print(f"ℹ️ Scale Factor: {latent_scale_factor:.6f}")

    # 7. Loop
    for epoch in range(start_epoch, args.num_epochs):
        metrics, avg_train_loss = train_epoch(
            flow_matcher, encoder, train_loader, optimizer, 
            device, epoch, latent_scale_factor, args.log_attn_freq
        )
        
        val_loss = validate(flow_matcher, encoder, val_loader, device, latent_scale_factor)
        
        # ✅ Step Scheduler SAU KHI HẾT EPOCH (Đúng chuẩn)
        scheduler.step()
        
        print(f"Epoch {epoch+1} | Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
              f"Flow: {metrics['flow']:.4f} | Sync: {metrics['sync']:.4f} | Len: {metrics['length']:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
        
        # Save Latest
        save_dict = {
            'epoch': epoch,
            'model_state_dict': flow_matcher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'latent_scale_factor': latent_scale_factor,
            'best_val_loss': best_val_loss
        }
        torch.save(save_dict, os.path.join(args.output_dir, 'latest.pt'))
        
        # Save Best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict['best_val_loss'] = best_val_loss 
            torch.save(save_dict, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"🏆 Best Model Saved! Loss: {best_val_loss:.4f}")
            
        # Save Milestone (Mỗi 50 epochs)
        if (epoch + 1) % 50 == 0:
            milestone_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save(save_dict, milestone_path)
            print(f"💾 Milestone saved: {milestone_path}")

if __name__ == '__main__':
    main()