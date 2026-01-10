#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✅ CRITICAL FIXES for NaN loss:
  1. Much lower learning rate (1e-6)
  2. Warmup scheduler
  3. Better gradient clipping
  4. Safer division operations
  5. Add gradient norm logging
  6. Disable sync_guidance initially
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())

from dataset import SignLanguageDataset, collate_fn
from models.fml.autoencoder import UnifiedPoseAutoencoder
from models.fml.latent_flow_matcher import LatentFlowMatcher


def get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps, min_lr_ratio=0.1):
    """Warmup + Cosine decay scheduler"""
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / float(max(1, warmup_steps))
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return max(min_lr_ratio, 0.5 * (1.0 + np.cos(np.pi * progress)))
    
    return LambdaLR(optimizer, lr_lambda)


def safe_float(x):
    """Convert to float safely"""
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    elif hasattr(x, "item"):
        return x.item()
    else:
        return float(x)


def extract_poses(batch):
    """Extract pose tensor from batch"""
    if isinstance(batch, dict):
        if "poses" in batch:
            return batch["poses"]
        if "pose" in batch:
            return batch["pose"]
        if "x" in batch:
            return batch["x"]
        return None
    else:
        return batch[0] if len(batch) > 0 else None


def estimate_scale_factor(encoder, dataloader, device, max_samples=1024):
    """
    ✅ IMPROVED: Use median + clipping for robustness
    """
    encoder.eval()
    latents = []
    seen = 0
    
    print("📊 Computing latent scale factor...")
    
    with torch.no_grad():
        for batch in dataloader:
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
            elif isinstance(batch, (list, tuple)):
                batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            
            poses = extract_poses(batch)
            if poses is None:
                continue
            
            # ✅ Check for bad data
            if torch.isnan(poses).any() or torch.isinf(poses).any():
                print("⚠️ Found NaN/Inf in poses, skipping batch")
                continue
            
            z = encoder.encode(poses)
            
            # ✅ Check encoded latents
            if torch.isnan(z).any() or torch.isinf(z).any():
                print("⚠️ Encoder produced NaN/Inf, skipping")
                continue
            
            latents.append(z.detach().cpu().flatten())
            
            seen += z.shape[0]
            if seen >= max_samples:
                break
    
    if not latents:
        print("⚠️ No valid latents collected, using scale=1.0")
        return 1.0
    
    latents = torch.cat(latents, dim=0)
    
    # Subsample if too large
    MAX_ELEMENTS = 1_000_000
    if latents.numel() > MAX_ELEMENTS:
        step = latents.numel() // MAX_ELEMENTS
        latents = latents[::step]
    
    # ✅ Use percentile-based std for robustness
    q25, q75 = torch.quantile(latents, torch.tensor([0.25, 0.75]))
    iqr = q75 - q25
    scale = iqr / 1.349  # IQR-based std estimate
    
    # Fallback to regular std if IQR too small
    if scale < 0.01:
        scale = float(latents.std())
    
    scale = max(scale, 0.1)  # Minimum scale
    
    print(f"   Latent stats: mean={latents.mean():.4f}, std={latents.std():.4f}")
    print(f"   IQR-based scale: {scale:.6f}")
    
    return scale


def prepare_batch(batch, device):
    """Prepare batch with validation"""
    if isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        poses = extract_poses(batch)
        
        # ✅ Validate poses
        if poses is not None:
            if torch.isnan(poses).any() or torch.isinf(poses).any():
                return None, None
        
        if "seq_lengths" not in batch and poses is not None:
            B, T, _ = poses.shape
            batch["seq_lengths"] = torch.full((B,), T, device=device, dtype=torch.long)
            
        return poses, batch
    else:
        if len(batch) < 3:
            return None, None
            
        poses = batch[0].to(device)
        
        # ✅ Validate
        if torch.isnan(poses).any() or torch.isinf(poses).any():
            return None, None
            
        if len(batch) >= 4:
            seq_lens = batch[3].to(device)
        else:
            B, T, _ = poses.shape
            seq_lens = torch.full((B,), T, device=device, dtype=torch.long)
        
        return poses, {
            'text_tokens': batch[1].to(device),
            'attention_mask': batch[2].to(device),
            'seq_lengths': seq_lens
        }


@torch.no_grad()
def validate(model, ae, val_loader, device, latent_scale):
    """Validation loop"""
    model.eval()
    metrics = {"total": 0.0, "flow": 0.0, "length": 0.0, "sync": 0.0, "prior": 0.0}
    n = 0
    
    for batch in val_loader:
        try:
            poses, batch_dict = prepare_batch(batch, device)
            if poses is None:
                continue
            
            gt_latent = ae.encode(poses) / latent_scale
            
            # ✅ Check encoded latent
            if torch.isnan(gt_latent).any() or torch.isinf(gt_latent).any():
                continue
            
            losses = model(
                batch_dict, 
                gt_latent, 
                pose_gt=poses.detach(),
                mode="train",
                prior_scale=1.0
            )
            
            for k in metrics:
                val = safe_float(losses.get(k, 0))
                if not np.isnan(val) and not np.isinf(val):
                    metrics[k] += val
            n += 1
            
        except Exception as e:
            print(f"⚠️ Validation error: {e}")
            continue
    
    if n == 0:
        return None
    
    return {k: v / n for k, v in metrics.items()}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ae_ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./ckpts_stage2")
    p.add_argument("--flow_ckpt", type=str, default=None)
    
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)  # ✅ Reduced batch size
    p.add_argument("--lr", type=float, default=1e-6)  # ✅ Much lower LR
    p.add_argument("--warmup_epochs", type=int, default=5)  # ✅ Warmup
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_grad_norm", type=float, default=0.5)  # ✅ Stricter clipping
    
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--ae_hidden_dim", type=int, default=512)  # ✅ Restored
    p.add_argument("--num_flow_layers", type=int, default=6)
    p.add_argument("--num_prior_layers", type=int, default=4)
    p.add_argument("--dropout", type=float, default=0.1)
    
    p.add_argument("--use_ssm_prior", type=int, default=1)
    p.add_argument("--use_sync_guidance", type=int, default=0)
    
    p.add_argument("--W_PRIOR", type=float, default=0.05)
    p.add_argument("--W_SYNC", type=float, default=0.0)
    p.add_argument("--W_LENGTH", type=float, default=1.0)
    
    p.add_argument("--prior_anneal_epochs", type=int, default=50)
    p.add_argument("--prior_warmup_epochs", type=int, default=5)  # ✅ Restored (ignored/deprecated)
    p.add_argument("--max_seq_len", type=int, default=400)
    p.add_argument("--patience", type=int, default=20)
    
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    device = torch.device(args.device)
    
    print("=" * 70)
    print("🚀 STAGE 2: LATENT FLOW MATCHING TRAINING (FIXED)")
    print("=" * 70)
    
    # Load datasets
    print("\n📂 Loading datasets...")
    train_ds = SignLanguageDataset(args.data_dir, split="train", max_seq_len=args.max_seq_len)
    val_ds = SignLanguageDataset(args.data_dir, split="val", max_seq_len=args.max_seq_len)
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=4
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=4
    )
    
    print(f"   Train: {len(train_ds)} samples")
    print(f"   Val: {len(val_ds)} samples")
    
    # Load autoencoder
    print(f"\n🔧 Loading autoencoder from: {args.ae_ckpt}")
    ae = UnifiedPoseAutoencoder(
        latent_dim=args.latent_dim,
        hidden_dim=args.ae_hidden_dim  # ✅ Use restored arg
    ).to(device)
    
    ckpt = torch.load(args.ae_ckpt, map_location=device)
    if "model_state_dict" in ckpt:
        ae.load_state_dict(ckpt["model_state_dict"])
    else:
        ae.load_state_dict(ckpt)
    
    ae.eval()
    for p in ae.parameters():
        p.requires_grad = False
    
    # Estimate latent scale
    latent_scale = estimate_scale_factor(ae, train_loader, device)
    print(f"   Latent scale factor: {latent_scale:.6f}")
    
    # Create flow matcher
    print("\n🌊 Creating Flow Matcher...")
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_flow_layers=args.num_flow_layers,
        num_prior_layers=args.num_prior_layers,
        dropout=args.dropout,
        use_ssm_prior=bool(args.use_ssm_prior),
        use_sync_guidance=bool(args.use_sync_guidance),
        W_PRIOR=args.W_PRIOR,
        W_SYNC=args.W_SYNC,
        W_LENGTH=args.W_LENGTH
    ).to(device)
    
    # Load checkpoint if resuming
    start_epoch = 0
    if args.flow_ckpt and os.path.exists(args.flow_ckpt):
        print(f"   Loading checkpoint: {args.flow_ckpt}")
        ckpt = torch.load(args.flow_ckpt, map_location=device)
        flow_matcher.load_state_dict(ckpt["model_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
    
    # ✅ Optimizer with weight decay
    optimizer = torch.optim.AdamW(
        flow_matcher.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=1e-4  # ✅ Add weight decay
    )
    
    # ✅ Warmup scheduler
    total_steps = len(train_loader) * args.epochs
    warmup_steps = len(train_loader) * args.warmup_epochs
    scheduler = get_warmup_cosine_scheduler(optimizer, warmup_steps, total_steps)
    
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    print("\n" + "=" * 70)
    print("🎯 TRAINING START")
    print("=" * 70)
    
    for epoch in range(start_epoch, args.epochs):
        flow_matcher.train()
        
        # ✅ Prior annealing
        progress = min(1.0, epoch / args.prior_anneal_epochs)
        current_prior_scale = progress
        
        metrics = {"total": 0.0, "flow": 0.0, "prior": 0.0, "sync": 0.0, "length": 0.0}
        n_batches = 0
        
        pbar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.epochs} [Prior={current_prior_scale:.1f}]"
        )
        
        for batch in pbar:
            try:
                poses, batch_dict = prepare_batch(batch, device)
                if poses is None:
                    continue
                
                # Encode to latent
                with torch.no_grad():
                    gt_latent = ae.encode(poses) / latent_scale
                
                # ✅ Validate latent
                if torch.isnan(gt_latent).any() or torch.isinf(gt_latent).any():
                    print(f"\n⚠️ NaN/Inf in latent encoding, skipping batch")
                    continue
                
                optimizer.zero_grad()
                
                # Forward
                losses = flow_matcher(
                    batch=batch_dict,
                    gt_latent=gt_latent,
                    pose_gt=poses.detach(),
                    mode="train",
                    prior_scale=current_prior_scale
                )
                total_loss = losses["total"]
                
                # ✅ Check for NaN
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"\n⚠️ NaN/Inf loss at epoch {epoch+1}, batch {n_batches}")
                    print(f"   Flow loss: {losses['flow'].item():.6f}")
                    print(f"   Prior loss: {losses['prior'].item():.6f}")
                    print(f"   Sync loss: {losses['sync'].item():.6f}")
                    print(f"   Length loss: {losses['length'].item():.6f}")
                    optimizer.zero_grad()
                    continue
                
                # Backward
                total_loss.backward()
                
                # ✅ Log gradient norms
                total_norm = 0.0
                for p in flow_matcher.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                
                # ✅ Gradient clipping
                torch.nn.utils.clip_grad_norm_(
                    flow_matcher.parameters(),
                    args.max_grad_norm
                )
                
                optimizer.step()
                scheduler.step()
                
                # Accumulate metrics
                for k in metrics:
                    val = safe_float(losses.get(k, 0))
                    if not np.isnan(val) and not np.isinf(val):
                        metrics[k] += val
                
                n_batches += 1
                
                # ✅ Log gradient norm every 10 batches
                if n_batches % 10 == 0:
                    writer.add_scalar("Train/grad_norm", total_norm, epoch * len(train_loader) + n_batches)
                
                pbar.set_postfix({
                    'Loss': f"{metrics['total']/n_batches:.4f}",
                    'Flow': f"{metrics['flow']/n_batches:.3f}",
                    'GradNorm': f"{total_norm:.2f}"
                })
                
            except Exception as e:
                print(f"\n❌ Error in batch: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if n_batches == 0:
            print(f"⚠️ No valid batches in epoch {epoch+1}")
            continue
        
        # Log metrics
        for k in metrics:
            writer.add_scalar(f"Train/{k}", metrics[k]/n_batches, epoch)
        
        # Print summary
        print(f"\n📊 Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"   Train Loss: {metrics['total']/n_batches:.4f}")
        print(f"   Flow: {metrics['flow']/n_batches:.4f}")
        print(f"   Prior: {metrics['prior']/n_batches:.4f}")
        print(f"   Sync: {metrics['sync']/n_batches:.4f}")
        print(f"   Length: {metrics['length']/n_batches:.4f}")
        
        # Validation
        val_res = validate(flow_matcher, ae, val_loader, device, latent_scale)
        
        if val_res:
            writer.add_scalar("Val/Total", val_res['total'], epoch)
            print(f"   Val Loss: {val_res['total']:.4f}")
            
            # Save best
            if val_res['total'] < best_val_loss:
                best_val_loss = val_res['total']
                patience_counter = 0
                
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": flow_matcher.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "latent_scale_factor": latent_scale,
                    "val_loss": best_val_loss,
                    "max_seq_len": args.max_seq_len,
                    "hidden_dim": args.hidden_dim
                }, os.path.join(args.save_dir, "best_model.pt"))
                
                print("   💾 Saved best model!")
            else:
                patience_counter += 1
                print(f"   ⏳ Patience: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    print("\nℹ️ Early stopping triggered!")
                    break
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": flow_matcher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "latent_scale_factor": latent_scale,
                "max_seq_len": args.max_seq_len
            }, os.path.join(args.save_dir, f"ckpt_epoch_{epoch+1}.pt"))
    
    writer.close()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    print(f"💾 Best model saved to: {args.save_dir}/best_model.pt")
    print(f"📉 Best val loss: {best_val_loss:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()