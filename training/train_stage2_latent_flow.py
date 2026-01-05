#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
✅ FIXED:
  - estimate_scale_factor uses .std() instead of .abs().quantile()
  - Better checkpoint validation
  - Proper optimizer initialization order
"""
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())

from dataset import SignLanguageDataset, collate_fn
from models.fml.autoencoder import UnifiedPoseAutoencoder
from models.fml.latent_flow_matcher import LatentFlowMatcher


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
    ✅ FIXED: Use .std() instead of .abs().quantile()
    
    Estimates the scale of latent space for better numerical stability
    """
    encoder.eval()
    latents = []
    seen = 0
    
    print("📊 Computing latent scale factor...")
    
    with torch.no_grad():
        for batch in dataloader:
            # Move batch to device
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor):
                        batch[k] = v.to(device)
            elif isinstance(batch, (list, tuple)):
                batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            
            poses = extract_poses(batch)
            if poses is None:
                continue
            
            z = encoder.encode(poses)
            latents.append(z.detach().cpu())
            
            seen += z.shape[0]
            if seen >= max_samples:
                break
    
    if not latents:
        print("⚠️ No latents collected, using scale=1.0")
        return 1.0
    
    latents = torch.cat(latents, dim=0).flatten()
    
    # Subsample if too large
    MAX_ELEMENTS = 1_000_000
    if latents.numel() > MAX_ELEMENTS:
        step = latents.numel() // MAX_ELEMENTS
        latents = latents[::step]
    
    # ✅ FIXED: Use std instead of abs().quantile()
    scale = float(latents.std())
    
    print(f"   Latent stats: mean={latents.mean():.4f}, std={scale:.4f}")
    print(f"   Using scale factor: {scale:.6f}")
    
    return max(scale, 1e-6)


def prepare_batch(batch, device):
    """Prepare batch for training"""
    if isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        poses = extract_poses(batch)
        
        if "seq_lengths" not in batch and poses is not None:
            B, T, _ = poses.shape
            batch["seq_lengths"] = torch.full((B,), T, device=device, dtype=torch.long)
            
        return poses, batch
    else:
        if len(batch) < 3:
            return None, None
            
        poses = batch[0].to(device)
        
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
            
            losses = model(
                batch_dict, 
                gt_latent, 
                pose_gt=poses.detach(),
                mode="train",
                prior_scale=1.0
            )
            
            for k in metrics:
                metrics[k] += safe_float(losses.get(k, 0))
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
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--patience", type=int, default=20)
    
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--ae_hidden_dim", type=int, default=512)
    p.add_argument("--max_seq_len", type=int, default=400)
    
    p.add_argument("--W_SYNC", type=float, default=0.1)
    p.add_argument("--W_LENGTH", type=float, default=0.01)
    p.add_argument("--W_PRIOR", type=float, default=0.1)
    p.add_argument("--prior_warmup_epochs", type=int, default=5)
    
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print("="*70)
    print("🚀 STAGE 2: LATENT FLOW MATCHER TRAINING")
    print("="*70)
    print(f"📁 Data: {args.data_dir}")
    print(f"🤖 Autoencoder: {args.ae_ckpt}")
    print(f"📦 Batch size: {args.batch_size}")
    print(f"📏 Max seq len: {args.max_seq_len}")
    print(f"💾 Save dir: {args.save_dir}")
    print("="*70)

    # Load datasets
    print("\n⏳ Loading datasets...")
    train_dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        split="train",
        max_seq_len=args.max_seq_len
    )
    
    # Find validation split
    val_dataset = None
    for split in ["dev", "test", "val"]:
        path = os.path.join(args.data_dir, split)
        if os.path.exists(path):
            val_dataset = SignLanguageDataset(
                data_dir=args.data_dir,
                split=split,
                max_seq_len=args.max_seq_len
            )
            print(f"   ✅ Using '{split}' as validation set")
            break
    
    if not val_dataset:
        print("   ⚠️ No validation split found, splitting from training...")
        full = len(train_dataset)
        val_sz = int(full * 0.1)
        train_dataset, val_dataset = torch.utils.data.random_split(
            train_dataset, [full - val_sz, val_sz]
        )
    
    print(f"   📊 Train: {len(train_dataset)} samples")
    print(f"   📊 Val: {len(val_dataset)} samples")
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn,
        drop_last=True,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )

    # Load Autoencoder (frozen)
    print("\n📦 Loading Autoencoder...")
    ae = UnifiedPoseAutoencoder(
        latent_dim=args.latent_dim,
        hidden_dim=args.ae_hidden_dim
    ).to(device)
    
    ae_ckpt = torch.load(args.ae_ckpt, map_location=device)
    ae.load_state_dict(ae_ckpt.get("model_state_dict", ae_ckpt), strict=False)
    ae.eval()
    
    for param in ae.parameters():
        param.requires_grad = False
    
    print("   ✅ Autoencoder loaded (frozen)")
    
    # Initialize Flow Matcher
    print("\n🔧 Initializing Flow Matcher...")
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        W_SYNC=args.W_SYNC,
        W_LENGTH=args.W_LENGTH,
        W_PRIOR=args.W_PRIOR
    ).to(device)
    
    total_params = sum(p.numel() for p in flow_matcher.parameters())
    trainable_params = sum(p.numel() for p in flow_matcher.parameters() if p.requires_grad)
    print(f"   📊 Total params: {total_params:,}")
    print(f"   📊 Trainable params: {trainable_params:,}")
    
    # ✅ Initialize optimizer BEFORE loading checkpoint
    optimizer = torch.optim.AdamW(
        flow_matcher.parameters(),
        lr=args.lr,
        weight_decay=1e-6
    )
    
    # Resume or estimate scale
    start_epoch = 0
    best_val_loss = float('inf')
    latent_scale = 1.0
    patience_counter = 0
    
    if args.flow_ckpt and os.path.exists(args.flow_ckpt):
        print(f"\n📄 Resuming from: {args.flow_ckpt}")
        ckpt = torch.load(args.flow_ckpt, map_location=device)
        
        flow_matcher.load_state_dict(ckpt['model_state_dict'], strict=False)
        
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("   ✅ Optimizer state loaded")
            except Exception as e:
                print(f"   ⚠️ Could not load optimizer state: {e}")
        
        start_epoch = ckpt.get("epoch", 0)
        latent_scale = float(ckpt.get("latent_scale_factor", 1.0))
        best_val_loss = float(ckpt.get("val_loss", float('inf')))
        
        # ✅ Validation
        if latent_scale == 1.0 and 'latent_scale_factor' not in ckpt:
            print("   ⚠️ WARNING: latent_scale_factor not in checkpoint!")
            print("   ⚠️ Re-estimating scale factor...")
            latent_scale = estimate_scale_factor(ae, train_loader, device)
        
        print(f"   ✅ Resumed from epoch {start_epoch}")
        print(f"   📏 Latent scale: {latent_scale:.6f}")
        print(f"   📉 Best val loss: {best_val_loss:.4f}")
    else:
        print("\n📊 Estimating latent scale factor...")
        latent_scale = estimate_scale_factor(ae, train_loader, device)
    
    print(f"\n📏 Final latent scale: {latent_scale:.6f}")
    
    # Scheduler
    scheduler = CosineAnnealingWarmRestarts(
        optimizer,
        T_0=50,
        T_mult=2,
        eta_min=1e-7
    )
    
    if start_epoch > 0:
        for _ in range(start_epoch):
            scheduler.step()
    
    # Training setup
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))
    
    PRIOR_WARMUP_EPOCHS = args.prior_warmup_epochs
    
    print("\n" + "="*70)
    print("🏋️ STARTING TRAINING")
    print("="*70)
    
    for epoch in range(start_epoch, args.epochs):
        flow_matcher.train()
        ae.eval()
        
        # Prior warmup
        current_prior_scale = 0.0 if epoch < PRIOR_WARMUP_EPOCHS else 1.0
        
        metrics = {
            "total": 0.0,
            "flow": 0.0,
            "length": 0.0,
            "sync": 0.0,
            "prior": 0.0
        }
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
                
                # Encode to latent and scale
                with torch.no_grad():
                    gt_latent = ae.encode(poses) / latent_scale
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    losses = flow_matcher(
                        batch=batch_dict,
                        gt_latent=gt_latent,
                        pose_gt=poses.detach(),
                        mode="train",
                        prior_scale=current_prior_scale
                    )
                    total_loss = losses["total"]
                
                # Check for NaN
                if torch.isnan(total_loss):
                    print(f"\n⚠️ NaN loss detected at epoch {epoch}, batch {n_batches}")
                    optimizer.zero_grad()
                    continue
                
                # Backward
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    flow_matcher.parameters(),
                    args.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                
                # Accumulate metrics
                for k in metrics:
                    metrics[k] += safe_float(losses.get(k, 0))
                n_batches += 1
                
                pbar.set_postfix({
                    'Loss': f"{metrics['total']/n_batches:.4f}",
                    'Flow': f"{metrics['flow']/n_batches:.3f}"
                })
                
            except Exception as e:
                print(f"\n❌ Error in batch: {e}")
                continue
        
        if n_batches == 0:
            print(f"⚠️ No valid batches in epoch {epoch+1}")
            continue
        
        # Log to tensorboard
        for k in metrics:
            writer.add_scalar(f"Train/{k}", metrics[k]/n_batches, epoch)
        
        # Validation
        print(f"\n📊 Epoch {epoch+1}/{args.epochs} Summary:")
        print(f"   Train Loss: {metrics['total']/n_batches:.4f}")
        print(f"   Flow: {metrics['flow']/n_batches:.4f}")
        print(f"   Prior: {metrics['prior']/n_batches:.4f}")
        print(f"   Sync: {metrics['sync']/n_batches:.4f}")
        print(f"   Length: {metrics['length']/n_batches:.4f}")
        
        val_res = validate(flow_matcher, ae, val_loader, device, latent_scale)
        
        if val_res:
            writer.add_scalar("Val/Total", val_res['total'], epoch)
            print(f"   Val Loss: {val_res['total']:.4f}")
            
            # Save best model
            if val_res['total'] < best_val_loss:
                best_val_loss = val_res['total']
                patience_counter = 0
                
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": flow_matcher.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
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
                    print("\n⏹️ Early stopping triggered!")
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
        
        scheduler.step()
    
    writer.close()
    
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETED!")
    print("="*70)
    print(f"💾 Best model saved to: {args.save_dir}/best_model.pt")
    print(f"📉 Best val loss: {best_val_loss:.4f}")
    print("="*70)


if __name__ == "__main__":
    main()