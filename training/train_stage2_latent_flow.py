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
# Hãy đảm bảo file dataset.py và models/autoencoder.py tồn tại
from dataset import SignLanguageDataset, collate_fn
from models.autoencoder import UnifiedPoseAutoencoder
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
            if isinstance(batch, dict):
                poses = batch.get("poses") or batch.get("pose") or batch.get("x")
            else:
                poses = batch[0]
            
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
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ae_ckpt", type=str, required=True)
    p.add_argument("--flow_ckpt", type=str, default=None)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--hidden_dim", type=int, default=512)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--save_dir", type=str, default="./ckpts")
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    args = p.parse_args()
    return args

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

   # 1. KHỞI TẠO DATALOADER (Đưa lên đầu) ========================
    print("⏳ Loading dataset...")
    train_dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        # Các tham số khác tùy dataset của bạn
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4, 
        collate_fn=collate_fn,
        pin_memory=True if torch.cuda.is_available() else False
    )
    print(f"✅ Loaded {len(train_dataset)} samples.")
    # ==============================================================

    # --- Initialize Models ---
    ae = UnifiedPoseAutoencoder(latent_dim=args.latent_dim).to(device)
    flow_matcher = LatentFlowMatcher(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)

    # --- Load Checkpoints ---
    if args.ae_ckpt and os.path.exists(args.ae_ckpt):
        ck = torch.load(args.ae_ckpt, map_location=device)
        state = ck.get("model_state_dict") or ck.get("state_dict") or ck
        ae.load_state_dict(state, strict=False)
        ae.eval()
        print("Loaded AE from", args.ae_ckpt)
    else:
        raise FileNotFoundError(f"AE checkpoint not found: {args.ae_ckpt}")

    start_epoch = 0
    latent_scale = 1.0
    
    if args.flow_ckpt and os.path.exists(args.flow_ckpt):
        ck = torch.load(args.flow_ckpt, map_location=device)
        flow_matcher.load_state_dict(ck['model_state_dict'], strict=False)
        start_epoch = ck.get("epoch", 0)
        latent_scale = float(ck.get("latent_scale_factor", 1.0))
        print(f"Resumed Flow from epoch {start_epoch}")
    else:
        # Nếu chưa có scale factor thì tính toán
        latent_scale = estimate_scale_factor(ae, train_loader, device) # Uncomment khi có loader
        pass

    print(f"Using Latent Scale Factor: {latent_scale:.6f}")

    # --- Optimizer & Scheduler ---
    optimizer = torch.optim.AdamW(flow_matcher.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    # --- Training Loop ---
    for epoch in range(start_epoch, args.epochs):
        flow_matcher.train()
        ae.eval()
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}", leave=False) # Uncomment khi có loader
        # Để test code chạy được thì loop giả:
        #pbar = [] 
        
        epoch_loss = 0.0
        n_batches = 0
        
        for batch in pbar:
            # 1. Handle Batch Logic
            if isinstance(batch, dict):
                poses = batch.get("poses") or batch.get("pose") or batch.get("x")
                if "seq_lengths" not in batch:
                     batch["seq_lengths"] = torch.full((poses.shape[0],), poses.shape[1], device=device, dtype=torch.long)
            else:
                poses = batch[0]
                batch = {
                    'text_tokens': batch[1],
                    'attention_mask': batch[2],
                    'seq_lengths': torch.full((poses.shape[0],), poses.shape[1], device=device, dtype=torch.long)
                }
            
            poses = poses.to(device)
            # Move all tensors to device
            for k, v in batch.items():
                if isinstance(v, torch.Tensor) and k != 'poses':
                     batch[k] = v.to(device)

            # 2. Get Ground Truth Latents
            with torch.no_grad():
                gt_latent = ae.encode(poses)
            
            # 3. Forward
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
                    print("NaN Loss detected, skipping.")
                    continue

            # 4. Backward
            scaler.scale(total_loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(flow_matcher.parameters(), max_norm=args.max_grad_norm)
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += safe_float(total_loss)
            n_batches += 1
            pbar.set_postfix({"loss": f"{epoch_loss/n_batches:.4f}"})

        scheduler.step(epoch + 1)
        
        # Save Checkpoint
        save_path = os.path.join(args.save_dir, f"flow_epoch_{epoch+1}.pt")
        torch.save({
            "epoch": epoch + 1,
            "model_state_dict": flow_matcher.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "latent_scale_factor": latent_scale
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

if __name__ == "__main__":
    main()