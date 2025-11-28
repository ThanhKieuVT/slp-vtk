#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())

# Import Models & Data
# Fallback import logic
try:
    from dataset import SignLanguageDataset, collate_fn
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher
except ImportError:
    try:
        from dataset import SignLanguageDataset, collate_fn
        from models.autoencoder import UnifiedPoseAutoencoder
        from models.fml.latent_flow_matcher import LatentFlowMatcher
    except ImportError:
        # Last resort for structure mismatch
        from dataset import SignLanguageDataset, collate_fn
        from autoencoder import UnifiedPoseAutoencoder
        from latent_flow_matcher import LatentFlowMatcher

def safe_float(x):
    return float(x) if isinstance(x, (int, float, np.floating)) else x.item() if hasattr(x, "item") else float(x)

# --- FIX 1: extract_poses viết lại để không dùng OR với Tensor ---
def extract_poses(batch):
    """Extract poses from batch (handles dict or tuple)"""
    if isinstance(batch, dict):
        # Dùng if key in dict thay vì .get() or .get() để tránh lỗi Tensor ambiguous
        if "poses" in batch: return batch["poses"]
        if "pose" in batch: return batch["pose"]
        if "x" in batch: return batch["x"]
        return None
    else:
        # Tuple format
        return batch[0] if len(batch) > 0 else None

def estimate_scale_factor(encoder, dataloader, device, max_samples=1024):
    encoder.eval()
    latents = []
    seen = 0
    print("Computing latent scale factor...")
    with torch.no_grad():
        for batch in dataloader:
            poses = extract_poses(batch)
            if poses is None: 
                continue
                
            poses = poses.to(device)
            z = encoder.encode(poses)
            latents.append(z.detach().cpu())
            seen += z.shape[0]
            if seen >= max_samples:
                break
    
    if not latents: 
        return 1.0
    
    latents = torch.cat(latents, dim=0)
    # Robust scale estimation using quantile
    scale = float(latents.abs().quantile(0.95))
    return max(scale, 1e-6)

def prepare_batch(batch, device):
    """Convert batch to standard dict format"""
    if isinstance(batch, dict):
        poses = extract_poses(batch)
        
        if "seq_lengths" not in batch:
            # Fallback if seq_lengths missing: use full length
            if poses is not None:
                B, T, _ = poses.shape
                batch["seq_lengths"] = torch.full((B,), T, dtype=torch.long)
        
        # Move tensors to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        return poses, batch
    
    else:
        # Tuple format: (poses, text_tokens, attention_mask, seq_lengths)
        if len(batch) < 3:
            return None, None
            
        poses = batch[0].to(device)
        
        # Handle seq_lengths if present in tuple
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
    total_loss = 0.0
    flow_loss = 0.0
    length_loss = 0.0
    sync_loss = 0.0
    n = 0
    
    for batch in val_loader:
        try:
            poses, batch_dict = prepare_batch(batch, device)
            if poses is None:
                continue
            
            gt_latent = ae.encode(poses) / latent_scale
            losses = model(batch_dict, gt_latent, pose_gt=poses, mode="train")
            
            total_loss += safe_float(losses["total"])
            flow_loss += safe_float(losses["flow"])
            length_loss += safe_float(losses["length"])
            sync_loss += safe_float(losses["sync"])
            n += 1
            
        except Exception as e:
            # print(f"⚠️  Validation batch error: {e}")
            continue
    
    if n == 0:
        return None
    
    return {
        "total": total_loss / n,
        "flow": flow_loss / n,
        "length": length_loss / n,
        "sync": sync_loss / n
    }

def parse_args():
    p = argparse.ArgumentParser()
    
    # Data & Paths
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ae_ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./ckpts")
    p.add_argument("--flow_ckpt", type=str, default=None)
    
    # Training
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--val_split", type=float, default=0.1, help="Validation split ratio")
    p.add_argument("--patience", type=int, default=20, help="Early stopping patience")
    
    # Model
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--ae_hidden_dim", type=int, default=512)
    
    # NEW: Max Seq Len (This fixes the error)
    p.add_argument("--max_seq_len", type=int, default=120, help="Max sequence length for padding")
    
    # Loss Weights
    p.add_argument("--W_SYNC", type=float, default=0.1)
    p.add_argument("--W_LENGTH", type=float, default=0.01)
    
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"🚀 Training Configuration:")
    print(f"   • Data: {args.data_dir}")
    print(f"   • AE Checkpoint: {args.ae_ckpt}")
    print(f"   • Device: {device}")
    print(f"   • Batch Size: {args.batch_size}")
    print(f"   • Learning Rate: {args.lr}")
    print(f"   • Max Seq Len: {args.max_seq_len}")
    print(f"   • Weights: Sync={args.W_SYNC}, Length={args.W_LENGTH}")

    # 1. Load Dataset
    print("\n⏳ Loading dataset...")
    # Thử truyền max_seq_len vào dataset, nếu dataset ko hỗ trợ thì fallback
    try:
        full_dataset = SignLanguageDataset(data_dir=args.data_dir, max_seq_len=args.max_seq_len)
    except TypeError:
        print("⚠️  SignLanguageDataset does not accept 'max_seq_len' arg. Using default length.")
        full_dataset = SignLanguageDataset(data_dir=args.data_dir)
        # Nếu muốn dùng 160 mà dataset mặc định 120, bạn cần sửa file dataset.py nữa.
    
    # Split train/val
    val_size = int(len(full_dataset) * args.val_split)
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # FIX: Giảm num_workers xuống 2 theo khuyến nghị của Warning
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, collate_fn=collate_fn, pin_memory=True
    )
    
    print(f"✅ Dataset loaded: {train_size} train, {val_size} val samples")

    # 2. Initialize Models
    ae = UnifiedPoseAutoencoder(
        latent_dim=args.latent_dim, 
        hidden_dim=args.ae_hidden_dim
    ).to(device)
    
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        W_SYNC=args.W_SYNC,
        W_LENGTH=args.W_LENGTH
    ).to(device)

    # 3. Load Autoencoder
    if not os.path.exists(args.ae_ckpt):
        raise FileNotFoundError(f"❌ AE checkpoint not found: {args.ae_ckpt}")
    
    ck = torch.load(args.ae_ckpt, map_location=device)
    state = ck.get("model_state_dict") or ck.get("state_dict") or ck
    ae.load_state_dict(state, strict=False)
    ae.eval()
    print("✅ Loaded AE checkpoint")

    # 4. Compute Latent Scale
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.flow_ckpt and os.path.exists(args.flow_ckpt):
        print(f"🔄 Resuming from: {args.flow_ckpt}")
        ck = torch.load(args.flow_ckpt, map_location=device)
        flow_matcher.load_state_dict(ck['model_state_dict'], strict=False)
        start_epoch = ck.get("epoch", 0)
        latent_scale = float(ck.get("latent_scale_factor", 1.0))
        best_val_loss = float(ck.get("val_loss", float('inf')))
    else:
        latent_scale = estimate_scale_factor(ae, train_loader, device)
    
    print(f"📏 Latent Scale Factor: {latent_scale:.6f}")

    # 5. Setup Training
    optimizer = torch.optim.AdamW(
        flow_matcher.parameters(), 
        lr=args.lr, 
        weight_decay=1e-6
    )
    scheduler = CosineAnnealingWarmRestarts(
        optimizer, T_0=50, T_mult=2, eta_min=1e-7
    )
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    # TensorBoard
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))
    
    # Early Stopping
    patience_counter = 0

    # 6. Training Loop
    print("\n🏋️  Starting training...\n")
    
    for epoch in range(start_epoch, args.epochs):
        flow_matcher.train()
        ae.eval()
        
        epoch_loss = 0.0
        epoch_flow = 0.0
        epoch_length = 0.0
        epoch_sync = 0.0
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            try:
                # Prepare batch
                poses, batch_dict = prepare_batch(batch, device)
                if poses is None:
                    continue
                
                # Encode to latent
                with torch.no_grad():
                    gt_latent = ae.encode(poses) / latent_scale
                
                # Forward
                optimizer.zero_grad()
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    losses = flow_matcher(
                        batch=batch_dict,
                        gt_latent=gt_latent,
                        pose_gt=poses,
                        mode="train"
                    )
                    total_loss = losses["total"]
                
                if torch.isnan(total_loss):
                    print("⚠️  NaN loss, skipping batch")
                    continue
                
                # Backward
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    flow_matcher.parameters(), 
                    max_norm=args.max_grad_norm
                )
                scaler.step(optimizer)
                scaler.update()
                
                # Accumulate metrics
                epoch_loss += safe_float(total_loss)
                epoch_flow += safe_float(losses["flow"])
                epoch_length += safe_float(losses["length"])
                epoch_sync += safe_float(losses["sync"])
                n_batches += 1
                
                # Update progress bar
                pbar.set_postfix({
                    "Loss": f"{epoch_loss/n_batches:.4f}",
                    "Flow": f"{epoch_flow/n_batches:.3f}",
                    "Len": f"{epoch_length/n_batches:.3f}"
                })
                
            except Exception as e:
                # print(f"\n⚠️  Batch error: {e}")
                continue
        
        if n_batches == 0:
            print("❌ No valid batches in this epoch!")
            continue
        
        # Compute epoch averages
        avg_loss = epoch_loss / n_batches
        avg_flow = epoch_flow / n_batches
        avg_length = epoch_length / n_batches
        avg_sync = epoch_sync / n_batches
        
        # Log to TensorBoard
        writer.add_scalar("Train/Total_Loss", avg_loss, epoch)
        writer.add_scalar("Train/Flow_Loss", avg_flow, epoch)
        writer.add_scalar("Train/Length_Loss", avg_length, epoch)
        writer.add_scalar("Train/Sync_Loss", avg_sync, epoch)
        writer.add_scalar("LR", optimizer.param_groups[0]['lr'], epoch)
        
        # Validation
        val_metrics = validate(flow_matcher, ae, val_loader, device, latent_scale)
        
        if val_metrics is not None:
            writer.add_scalar("Val/Total_Loss", val_metrics['total'], epoch)
            writer.add_scalar("Val/Flow_Loss", val_metrics['flow'], epoch)
            
            print(f"\n📊 Epoch {epoch+1}/{args.epochs}")
            print(f"   Train Loss: {avg_loss:.4f} | Val Loss: {val_metrics['total']:.4f}")
            
            # Save best model
            if val_metrics['total'] < best_val_loss:
                best_val_loss = val_metrics['total']
                patience_counter = 0
                
                save_path = os.path.join(args.save_dir, "best_model.pt")
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": flow_matcher.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "latent_scale_factor": latent_scale,
                    "val_loss": best_val_loss
                }, save_path)
                print(f"   💾 New best model saved! (Val Loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"   ⏳ No improvement for {patience_counter}/{args.patience} epochs")
            
            # Early stopping
            if patience_counter >= args.patience:
                print(f"\n⏹️  Early stopping triggered after {epoch+1} epochs")
                break
        
        # Periodic checkpoint
        if (epoch + 1) % 10 == 0:
            periodic_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": flow_matcher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "latent_scale_factor": latent_scale,
                "val_loss": val_metrics['total'] if val_metrics else None
            }, periodic_path)
        
        scheduler.step(epoch + 1)
    
    writer.close()
    print("\n✅ Training completed!")

if __name__ == "__main__":
    main()