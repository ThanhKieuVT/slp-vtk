"""
Training Script cho Stage 2: Latent Flow Matching với SSM Prior + Sync Guidance
"""
import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import SignLanguageDataset, collate_fn
from models.autoencoder import UnifiedPoseAutoencoder
from models.fml.latent_flow_matcher import LatentFlowMatcher
from models.fml.consistency_distillation import ConsistencyDistillation


def train_epoch(flow_matcher, encoder, decoder, dataloader, optimizer, device, epoch):
    """Train một epoch"""
    flow_matcher.train()
    encoder.eval()  # Freeze encoder
    decoder.eval()  # Freeze decoder
    
    total_loss = 0.0
    total_flow_loss = 0.0
    total_prior_loss = 0.0
    total_sync_loss = 0.0
    num_batches = 0
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
    for batch in pbar:
        poses = batch['poses'].to(device)  # [B, T, 214]
        pose_mask = batch['pose_mask'].to(device)  # [B, T]
        text_tokens = batch['text_tokens'].to(device)  # [B, L]
        attention_mask = batch['attention_mask'].to(device)  # [B, L]
        seq_lengths = batch['seq_lengths'].to(device)  # [B]
        
        # Prepare batch dict
        batch_dict = {
            'text_tokens': text_tokens,
            'attention_mask': attention_mask,
            'seq_lengths': seq_lengths,
            'target_length': seq_lengths
        }
        
        # Step 1: Encode GT pose → latent (với frozen encoder)
        with torch.no_grad():
            gt_latent = encoder(poses, mask=pose_mask)  # [B, T, 256]
        
        # Step 2: Flow Matching trên latent
        losses = flow_matcher(
            batch_dict,
            gt_latent=gt_latent,
            pose_gt=poses,
            mode='train'
        )
        
        # Backward
        optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(flow_matcher.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += losses['total'].item()
        total_flow_loss += losses['flow'].item()
        total_prior_loss += losses['prior'].item()
        total_sync_loss += losses['sync'].item()
        num_batches += 1
        
        pbar.set_postfix({
            'loss': losses['total'].item(),
            'flow': losses['flow'].item(),
            'prior': losses['prior'].item(),
            'sync': losses['sync'].item()
        })
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    avg_flow = total_flow_loss / num_batches if num_batches > 0 else 0.0
    avg_prior = total_prior_loss / num_batches if num_batches > 0 else 0.0
    avg_sync = total_sync_loss / num_batches if num_batches > 0 else 0.0
    
    return {
        'total': avg_loss,
        'flow': avg_flow,
        'prior': avg_prior,
        'sync': avg_sync
    }


def validate(flow_matcher, encoder, decoder, dataloader, device):
    """Validate"""
    flow_matcher.eval()
    encoder.eval()
    decoder.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Validating"):
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
            
            # Encode GT
            gt_latent = encoder(poses, mask=pose_mask)
            
            # Flow Matching
            losses = flow_matcher(
                batch_dict,
                gt_latent=gt_latent,
                pose_gt=poses,
                mode='train'
            )
            
            total_loss += losses['total'].item()
            num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Đường dẫn đến processed_data/data/')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Thư mục lưu checkpoints')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                        help='Đường dẫn đến checkpoint của Stage 1 (autoencoder)')
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
    parser.add_argument('--max_seq_len', type=int, default=120,
                        help='Max sequence length')
    parser.add_argument('--use_ssm_prior', action='store_true',
                        help='Sử dụng SSM prior')
    parser.add_argument('--use_sync_guidance', action='store_true',
                        help='Sử dụng sync guidance')
    parser.add_argument('--lambda_prior', type=float, default=0.1,
                        help='Weight cho SSM prior')
    parser.add_argument('--gamma_guidance', type=float, default=0.01,
                        help='Weight cho sync guidance')
    parser.add_argument('--num_inference_steps', type=int, default=50,
                        help='Số bước ODE khi inference')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume từ checkpoint')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Autoencoder (Stage 1) - chỉ cần decoder
    print(f"Loading autoencoder from {args.autoencoder_checkpoint}")
    autoencoder = UnifiedPoseAutoencoder(
        pose_dim=214,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    )
    checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.to(device)
    autoencoder.eval()
    
    # Freeze encoder và decoder
    for param in autoencoder.encoder.parameters():
        param.requires_grad = False
    for param in autoencoder.decoder.parameters():
        param.requires_grad = False
    
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder
    
    # Datasets
    train_dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        split='train',
        max_seq_len=args.max_seq_len
    )
    val_dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        split='dev',
        max_seq_len=args.max_seq_len,
        stats_path=os.path.join(args.data_dir, "normalization_stats.npz")
    )
    test_dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        split='test',
        max_seq_len=args.max_seq_len,
        stats_path=os.path.join(args.data_dir, "normalization_stats.npz")
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
    
    # Model: Latent Flow Matcher
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_flow_layers=6,
        num_prior_layers=4,
        num_heads=8,
        dropout=0.1,
        use_ssm_prior=args.use_ssm_prior,
        use_sync_guidance=args.use_sync_guidance,
        lambda_prior=args.lambda_prior,
        gamma_guidance=args.gamma_guidance,
        lambda_anneal=True
    ).to(device)
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        flow_matcher.parameters(),
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
        flow_matcher.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print(f"\n{'='*50}")
    print(f"Starting training Stage 2: Latent Flow Matching")
    print(f"  SSM Prior: {args.use_ssm_prior}")
    print(f"  Sync Guidance: {args.use_sync_guidance}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    print(f"{'='*50}\n")
    
    for epoch in range(start_epoch, args.num_epochs):
        # Train
        train_losses = train_epoch(flow_matcher, encoder, decoder, train_loader, optimizer, device, epoch)
        
        # Validate
        val_loss = validate(flow_matcher, encoder, decoder, val_loader, device)
        
        # Scheduler step
        scheduler.step()
        
        # Log
        print(f"\nEpoch {epoch+1}/{args.num_epochs}")
        print(f"  Train Loss: {train_losses['total']:.6f}")
        print(f"    Flow: {train_losses['flow']:.6f}")
        print(f"    Prior: {train_losses['prior']:.6f}")
        print(f"    Sync: {train_losses['sync']:.6f}")
        print(f"  Val Loss: {val_loss:.6f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': flow_matcher.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'train_loss': train_losses['total'],
            'val_loss': val_loss,
            'best_val_loss': best_val_loss,
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
    test_loss = validate(flow_matcher, encoder, decoder, test_loader, device)
    print(f"Test Loss: {test_loss:.6f}")
    print(f"\n💡 Tip: Chạy evaluate.py để có metrics chi tiết (DTW, Sync, BLEU, etc.)")


if __name__ == '__main__':
    main()

