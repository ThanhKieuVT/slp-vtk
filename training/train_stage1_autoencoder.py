"""
Training Script cho Stage 1: Autoencoder
✅ FIXED: max_seq_len = 400, grouped normalization support
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


def train_epoch(model, dataloader, optimizer, device, epoch):
    """Train một epoch"""
    model.train()
    total_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        poses = batch['poses'].to(device)  # [B, T, 214]
        pose_mask = batch['pose_mask'].to(device)  # [B, T]
        
        # Forward
        reconstructed_pose, latent = model(poses, mask=pose_mask)
        
        # Loss: MSE trên valid positions
        loss = nn.functional.mse_loss(
            reconstructed_pose,
            poses,
            reduction='none'
        )  # [B, T, 214]
        
        # Mask out padding
        loss = loss * pose_mask.unsqueeze(-1).float()
        
        # ✅ Scale Loss đúng chuẩn MSE
        num_features = poses.shape[-1]  # 214
        total_valid_elements = pose_mask.sum().clamp(min=1) * num_features
        
        loss = loss.sum() / total_valid_elements
        
        # ✅ Velocity Loss for temporal smoothness (SOTA improvement)
        # Compute first-order difference (velocity)
        if poses.shape[1] > 1:  # Only if sequence has more than 1 frame
            vel_recon = reconstructed_pose[:, 1:] - reconstructed_pose[:, :-1]  # [B, T-1, 214]
            vel_gt = poses[:, 1:] - poses[:, :-1]  # [B, T-1, 214]
            
            loss_velocity = nn.functional.mse_loss(
                vel_recon,
                vel_gt,
                reduction='none'
            )  # [B, T-1, 214]
            
            # Mask out padding for velocity (shift mask by 1)
            vel_mask = pose_mask[:, 1:].unsqueeze(-1).float()  # [B, T-1, 1]
            loss_velocity = loss_velocity * vel_mask
            
            # Scale velocity loss
            num_valid_vel = vel_mask.sum().clamp(min=1) * num_features
            loss_velocity = loss_velocity.sum() / num_valid_vel
            
            # Combine losses (velocity weight = 0.1)
            loss = loss + 0.1 * loss_velocity
        
        # Backward
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient Clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
        
        pbar.set_postfix({'loss': loss.item()})
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def validate(model, dataloader, device):
    """Validate"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
            poses = batch['poses'].to(device)
            pose_mask = batch['pose_mask'].to(device)
            
            reconstructed_pose, _ = model(poses, mask=pose_mask)
            
            loss = nn.functional.mse_loss(
                reconstructed_pose,
                poses,
                reduction='none'
            )
            
            # Mask out padding
            loss = loss * pose_mask.unsqueeze(-1).float()
            
            # Scale Loss
            num_features = poses.shape[-1]
            total_valid_elements = pose_mask.sum().clamp(min=1) * num_features
            
            loss = loss.sum() / total_valid_elements
            
            total_loss += loss.item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Đường dẫn đến processed_data/data/')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Thư mục lưu checkpoints')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_epochs', type=int, default=100,
                        help='Số epochs')
    parser.add_argument('--learning_rate', type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Số workers cho DataLoader')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--max_seq_len', type=int, default=400,  # ✅ FIXED: 120 → 400
                        help='Max sequence length')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume từ checkpoint')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ✅ Datasets với max_seq_len = 400
    stats_file = os.path.join(args.data_dir, "normalization_stats.npz")
    
    train_dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        split='train',
        max_seq_len=args.max_seq_len  # ✅ Use argument
    )
    
    val_stats = stats_file if os.path.exists(stats_file) else None
    if val_stats is None:
        print("⚠️ Warning: normalization_stats.npz not found for validation/test.")
    
    val_dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        split='dev',
        max_seq_len=args.max_seq_len,  # ✅ Use argument
        stats_path=val_stats
    )
    test_dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        split='test',
        max_seq_len=args.max_seq_len,  # ✅ Use argument
        stats_path=val_stats
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Model
    model = UnifiedPoseAutoencoder(
        pose_dim=214,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        encoder_layers=6,
        decoder_coarse_layers=4,
        decoder_medium_layers=4,
        decoder_fine_layers=6,
        num_heads=8,
        dropout=0.1
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Loading checkpoint from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\n{'='*50}")
    print(f"Starting training...")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Max seq len: {args.max_seq_len}")  # ✅ Log max_seq_len
    print(f"{'='*50}\n")
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(model, val_loader, device)
        
        # Scheduler step
        scheduler.step()
        
        # Log
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {train_loss:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_loss,
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
            'max_seq_len': args.max_seq_len,  # ✅ Save max_seq_len
        }
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.output_dir, 'latest.pt'))
        
        # Save best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint['best_val_loss'] = best_val_loss
            torch.save(checkpoint, os.path.join(args.output_dir, 'best_model.pt'))
            print(f"  ✅ Saved best model (val_loss: {val_loss:.6f})")
        
        # Save periodic
        if (epoch + 1) % 10 == 0:
            torch.save(checkpoint, os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt'))
    
    print(f"\n✅ Training completed!")
    print(f"Best val loss: {best_val_loss:.6f}")
    
    # Final evaluation on test set
    print(f"\n{'='*50}")
    print("Final evaluation on test set...")
    print(f"{'='*50}")
    test_loss = validate(model, test_loader, device)
    print(f"Test Loss: {test_loss:.6f}")


if __name__ == '__main__':
    main()