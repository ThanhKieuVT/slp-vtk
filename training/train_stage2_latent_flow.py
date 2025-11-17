"""
Training Script cho Stage 2: Latent Flow Matching với SSM Prior + Sync Guidance
Fixed by Gemini: Added Latent Scaling & Robust Validation
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np

from dataset import SignLanguageDataset, collate_fn
from models.fml.autoencoder import UnifiedPoseAutoencoder
from models.fml.latent_flow_matcher import LatentFlowMatcher
# from models.fml.consistency_distillation import ConsistencyDistillation # Chưa dùng thì comment

# === HÀM TÍNH SCALE FACTOR (BẮT BUỘC) ===
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
    scale_factor = 1.0 / (std.item() + 1e-6)
    
    print(f"✅ Latent Std gốc: {std.item():.4f}")
    print(f"✅ Scale Factor: {scale_factor:.6f}")
    return scale_factor

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
        
        # 1. Encode & Scale
        with torch.no_grad():
            gt_latent = encoder(poses, mask=pose_mask)
            gt_latent = gt_latent * scale_factor # <--- FIX QUAN TRỌNG

        # 2. Flow Matching Step
        losses = flow_matcher(
            batch_dict,
            gt_latent=gt_latent,
            pose_gt=poses, # Truyền pose gốc vào nếu Sync Loss cần tính trên Pose
            mode='train'
        )
        
        # 3. Backward
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(flow_matcher.parameters(), max_norm=1.0)
        optimizer.step()
        
        # Logging
        total_loss += losses['total'].item()
        losses_log['flow'] += losses['flow'].item()
        losses_log['prior'] += losses['prior'].item()
        losses_log['sync'] += losses['sync'].item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': f"{losses['total'].item():.4f}",
            'flow': f"{losses['flow'].item():.4f}"
        })
    
    return {k: v / num_batches for k, v in losses_log.items()}, total_loss / num_batches

def validate(flow_matcher, encoder, dataloader, device, scale_factor):
    flow_matcher.eval()
    encoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    # Lưu ý: Nếu Sync Guidance Loss cần tính gradient (ví dụ gradient penalty),
    # bạn phải dùng torch.enable_grad(). Nếu chỉ là MSE thuần tuý thì dùng no_grad().
    # Ở đây tôi giả định Sync Loss cần gradient nên dùng enable_grad() nhưng KHÔNG backward.
    
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
            gt_latent = gt_latent * scale_factor # Scale cả lúc validate

        # Context manager tuỳ thuộc vào việc Loss function có cần autograd không
        with torch.set_grad_enabled(True): # Bật grad để tính loss phức tạp
            losses = flow_matcher(
                batch_dict,
                gt_latent=gt_latent,
                pose_gt=poses,
                mode='train'
            )
            
        total_loss += losses['total'].item()
        num_batches += 1
            
    return total_loss / num_batches if num_batches > 0 else 0.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--max_seq_len', type=int, default=120)
    parser.add_argument('--use_ssm_prior', action='store_true')
    parser.add_argument('--use_sync_guidance', action='store_true')
    parser.add_argument('--resume_from', type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 1. Load Encoder (Frozen)
    print(f"Loading autoencoder...")
    autoencoder = UnifiedPoseAutoencoder(pose_dim=214, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim)
    checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
    
    # Xử lý key state_dict
    if 'model_state_dict' in checkpoint: state_dict = checkpoint['model_state_dict']
    else: state_dict = checkpoint
    autoencoder.load_state_dict(state_dict)
    
    encoder = autoencoder.encoder.to(device).eval().requires_grad_(False)
    # Decoder không cần load trong training loop nếu không visualize, tiết kiệm VRAM
    
    # 2. Dataset
    train_dataset = SignLanguageDataset(args.data_dir, split='train', max_seq_len=args.max_seq_len)
    val_dataset = SignLanguageDataset(args.data_dir, split='dev', max_seq_len=args.max_seq_len)
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_fn)
    
    # 3. CALCULATE SCALE FACTOR (QUAN TRỌNG)
    latent_scale_factor = estimate_scale_factor(encoder, train_loader, device)

    # 4. Flow Matcher Model
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        use_ssm_prior=args.use_ssm_prior,
        use_sync_guidance=args.use_sync_guidance
    ).to(device)
    
    optimizer = torch.optim.AdamW(flow_matcher.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs)
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    if args.resume_from:
        ckpt = torch.load(args.resume_from, map_location=device)
        flow_matcher.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        latent_scale_factor = ckpt.get('latent_scale_factor', latent_scale_factor) # Load scale cũ
        print(f"Resumed. Using Scale Factor: {latent_scale_factor}")

    # 5. Loop
    for epoch in range(start_epoch, args.num_epochs):
        train_metrics, avg_train_loss = train_epoch(flow_matcher, encoder, train_loader, optimizer, device, epoch, latent_scale_factor)
        val_loss = validate(flow_matcher, encoder, val_loader, device, latent_scale_factor)
        scheduler.step()
        
        print(f"Epoch {epoch+1} | Train Loss: {avg_train_loss:.6f} | Val Loss: {val_loss:.6f}")
        print(f"   Details: Flow={train_metrics['flow']:.4f}, Prior={train_metrics['prior']:.4f}, Sync={train_metrics['sync']:.4f}")
        
        # Save
        save_dict = {
            'epoch': epoch,
            'model_state_dict': flow_matcher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'latent_scale_factor': latent_scale_factor, # Nhớ lưu cái này để lúc Inference dùng
            'best_val_loss': best_val_loss
        }
        torch.save(save_dict, os.path.join(args.output_dir, 'latest.pt'))
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            save_dict['best_val_loss'] = best_val_loss
            torch.save(save_dict, os.path.join(args.output_dir, 'best_model.pt'))
            print("✅ Saved Best Model")

if __name__ == '__main__':
    main()