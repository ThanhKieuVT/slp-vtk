#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import argparse
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

sys.path.append(os.getcwd())

# Import Models & Data
try:
    from dataset import SignLanguageDataset, collate_fn
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher
except ImportError:
    from dataset import SignLanguageDataset, collate_fn
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher

def safe_float(x):
    return float(x) if isinstance(x, (int, float, np.floating)) else x.item() if hasattr(x, "item") else float(x)

def estimate_scale_factor(encoder, dataloader, device, max_samples=1024):
    encoder.eval()
    latents = []
    seen = 0
    print("Computing latent scale factor...")
    with torch.no_grad():
        for batch in dataloader:
            # --- FIX LOGIC TENSOR ---
            poses = None
            if isinstance(batch, dict):
                if "poses" in batch: poses = batch["poses"]
                elif "pose" in batch: poses = batch["pose"]
                elif "x" in batch: poses = batch["x"]
            else:
                poses = batch[0]
            
            if poses is None:
                continue # Skip bad batch
                
            poses = poses.to(device)
            z = encoder.encode(poses)
            latents.append(z.detach().cpu())
            seen += z.shape[0]
            if seen >= max_samples:
                break
    
    if not latents: return 1.0
    latents = torch.cat(latents, dim=0)
    scale = float(latents.std().mean())
    return max(scale, 1e-6)

def parse_args():
    p = argparse.ArgumentParser()
    
    # --- Data & Paths ---
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ae_ckpt", "--autoencoder_checkpoint", dest="ae_ckpt", type=str, required=True)
    p.add_argument("--save_dir", "--outputza_dir", dest="save_dir", type=str, default="./ckpts")
    p.add_argument("--flow_ckpt", "--resume_from", dest="flow_ckpt", type=str, default=None)
    
    # --- Training Params ---
    p.add_argument("--epochs", "--num_epochs", dest="epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", "--learning_rate", dest="lr", type=float, default=1e-4)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    
    # --- Model Params ---
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--ae_hidden_dim", type=int, default=512)
    p.add_argument("--max_seq_len", type=int, default=120)
    
    # --- Loss Weights ---
    p.add_argument("--W_SYNC", type=float, default=0.1)
    p.add_argument("--W_LENGTH", type=float, default=0.01)
    
    args = p.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"🚀 Training Config:")
    print(f"   • Data: {args.data_dir}")
    print(f"   • AE Checkpoint: {args.ae_ckpt}")
    print(f"   • Weights: Sync={args.W_SYNC}, Length={args.W_LENGTH}")

    # 1. Init Dataset
    print("⏳ Loading dataset...")
    try:
        train_dataset = SignLanguageDataset(data_dir=args.data_dir)
    except TypeError:
        train_dataset = SignLanguageDataset(data_dir=args.data_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4, 
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"✅ Loaded {len(train_dataset)} samples.")

    # 2. Init Models
    ae = UnifiedPoseAutoencoder(latent_dim=args.latent_dim, hidden_dim=args.ae_hidden_dim).to(device)
    
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim, 
        hidden_dim=args.hidden_dim,
        W_SYNC=args.W_SYNC,
        W_LENGTH=args.W_LENGTH
    ).to(device)

    # 3. Load AE Checkpoint
    if args.ae_ckpt and os.path.exists(args.ae_ckpt):
        ck = torch.load(args.ae_ckpt, map_location=device)
        state = ck.get("model_state_dict") or ck.get("state_dict") or ck
        ae.load_state_dict(state, strict=False)
        ae.eval()
        print("✅ Loaded AE checkpoint")
    else:
        raise FileNotFoundError(f"AE checkpoint not found: {args.ae_ckpt}")

    start_epoch = 0
    latent_scale = 1.0
    
    # 4. Resume Flow Checkpoint (Optional)
    if args.flow_ckpt and os.path.exists(args.flow_ckpt):
        print(f"🔄 Resuming from: {args.flow_ckpt}")
        ck = torch.load(args.flow_ckpt, map_location=device)
        flow_matcher.load_state_dict(ck['model_state_dict'], strict=False)
        start_epoch = ck.get("epoch", 0)
        latent_scale = float(ck.get("latent_scale_factor", 1.0))
    else:
        latent_scale = estimate_scale_factor(ae, train_loader, device)
        pass

    print(f"📏 Using Latent Scale Factor: {latent_scale:.6f}")

    optimizer = torch.optim.AdamW(flow_matcher.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    # 5. Training Loop
    for epoch in range(start_epoch, args.epochs):
        flow_matcher.train()
        ae.eval()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}", leave=False)
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in pbar:
            # --- FIX LOGIC TENSOR (Lặp lại fix ở đây) ---
            poses = None
            if isinstance(batch, dict):
                # Ưu tiên lấy poses
                if "poses" in batch: poses = batch["poses"]
                elif "pose" in batch: poses = batch["pose"]
                elif "x" in batch: poses = batch["x"]
                
                # Setup seq_lengths nếu chưa có
                if poses is not None and "seq_lengths" not in batch:
                     batch["seq_lengths"] = torch.full((poses.shape[0],), poses.shape[1], device=device, dtype=torch.long)
            else:
                poses = batch[0]
                batch = {
                    'text_tokens': batch[1].to(device),
                    'attention_mask': batch[2].to(device),
                    'seq_lengths': torch.full((poses.shape[0],), poses.shape[1], device=device, dtype=torch.long)
                }
            
            if poses is None:
                # Debug print nếu cần
                continue

            poses = poses.to(device)
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and k != 'poses':
                     batch[k] = v.to(device)

            # Get Ground Truth Latents
            with torch.no_grad():
                gt_latent = ae.encode(poses)
                gt_latent = gt_latent / latent_scale
            
            # Forward
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                losses = flow_matcher(
                    batch=batch,
                    gt_latent=gt_latent,
                    pose_gt=poses,
                    mode="train"
                )
                total_loss = losses["total"]

                if torch.isnan(total_loss):
                    print("⚠️ NaN Loss detected, skipping batch.")
                    continue

            # Backward
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(flow_matcher.parameters(), max_norm=args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += safe_float(total_loss)
            n_batches += 1
            pbar.set_postfix({
                "Loss": f"{epoch_loss/n_batches:.4f}", 
                "Flow": f"{safe_float(losses['flow']):.3f}",
                "Len": f"{safe_float(losses['length']):.3f}"
            })

        scheduler.step(epoch + 1)
        
        # Save Checkpoint
        save_path = os.path.join(args.save_dir, "best_model.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": flow_matcher.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "latent_scale_factor": latent_scale
        }, save_path)
        
        if (epoch + 1) % 10 == 0:
             print(f"💾 Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()