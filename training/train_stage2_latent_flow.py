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
            # Flatten immediately to avoid shape mismatch (varying T)
            latents.append(z.detach().cpu().flatten())
            
            seen += z.shape[0]
            if seen >= max_samples:
                break
    
    if not latents:
        print("⚠️ No latents collected, using scale=1.0")
        return 1.0
    
    # Concatenate 1D tensors
    latents = torch.cat(latents, dim=0)
    
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
        
        # ✅ Check for bad data in input
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
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-5) # Reduced LR
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # ... (args) ...

    # Training setup
    # scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda")) # Removed AMP
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))
    
    # ...

                optimizer.zero_grad()
                
                # with torch.cuda.amp.autocast(enabled=(device.type == "cuda")): # Removed AMP
                losses = flow_matcher(
                    batch=batch_dict,
                    gt_latent=gt_latent,
                    pose_gt=poses.detach(),
                    mode="train",
                    prior_scale=current_prior_scale
                )
                total_loss = losses["total"]
                
                # Check for NaN immediately
                if torch.isnan(total_loss) or torch.isinf(total_loss):
                    print(f"\n⚠️ NaN/Inf loss at epoch {epoch+1}, batch {n_batches}. Skipping...")
                    optimizer.zero_grad()
                    continue
                
                # Backward (Standard FP32)
                total_loss.backward()
                
                # scaler.scale(total_loss).backward() # Removed AMP
                
                # Unscale BEFORE clipping - Standard FP32 doesn't need unscale
                # scaler.unscale_(optimizer) # Removed AMP
                
                torch.nn.utils.clip_grad_norm_(
                    flow_matcher.parameters(),
                    args.max_grad_norm
                )
                
                # Step
                optimizer.step()
                # scaler.step(optimizer) # Removed AMP
                # scaler.update() # Removed AMP
                
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