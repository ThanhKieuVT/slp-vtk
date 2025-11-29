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
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.getcwd())

# Import Models & Data
try:
    from dataset import SignLanguageDataset, collate_fn
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher
except ImportError:
    from dataset import SignLanguageDataset, collate_fn
    from models.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher

def safe_float(x):
    return float(x) if isinstance(x, (int, float, np.floating)) else x.item() if hasattr(x, "item") else float(x)

def extract_poses(batch):
    if isinstance(batch, dict):
        if "poses" in batch: return batch["poses"]
        if "pose" in batch: return batch["pose"]
        if "x" in batch: return batch["x"]
        return None
    else:
        return batch[0] if len(batch) > 0 else None

def estimate_scale_factor(encoder, dataloader, device, max_samples=1024):
    encoder.eval()
    latents = []
    seen = 0
    print("Computing latent scale factor...")
    with torch.no_grad():
        for batch in dataloader:
            # Sửa lỗi logic ở đây tương tự prepare_batch
            if isinstance(batch, dict):
                for k, v in batch.items():
                    if isinstance(v, torch.Tensor): batch[k] = v.to(device)
            elif isinstance(batch, (list, tuple)):
                batch = [x.to(device) if isinstance(x, torch.Tensor) else x for x in batch]
            
            poses = extract_poses(batch)
            if poses is None: continue
            
            # poses đã ở trên device
            z = encoder.encode(poses)
            latents.append(z.detach().cpu())
            seen += z.shape[0]
            if seen >= max_samples: break
    
    if not latents: return 1.0
    latents = torch.cat(latents, dim=0).flatten()
    MAX_ELEMENTS = 1_000_000
    if latents.numel() > MAX_ELEMENTS:
        step = latents.numel() // MAX_ELEMENTS
        latents = latents[::step]
    scale = float(latents.abs().quantile(0.95))
    return max(scale, 1e-6)

# === FIX QUAN TRỌNG: Đảo thứ tự to(device) ===
def prepare_batch(batch, device):
    if isinstance(batch, dict):
        # 1. Chuyển TOÀN BỘ dict lên GPU trước
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        # 2. Sau đó mới lấy poses (lúc này poses đã là GPU tensor)
        poses = extract_poses(batch)
        
        # 3. Tạo seq_lengths nếu thiếu (trên GPU)
        if "seq_lengths" not in batch and poses is not None:
            B, T, _ = poses.shape
            batch["seq_lengths"] = torch.full((B,), T, device=device, dtype=torch.long)
            
        return poses, batch
    else:
        # Trường hợp Tuple/List
        if len(batch) < 3: return None, None
        poses = batch[0].to(device)
        if len(batch) >= 4: seq_lens = batch[3].to(device)
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
    model.eval()
    metrics = {"total": 0.0, "flow": 0.0, "length": 0.0, "sync": 0.0, "prior": 0.0}
    n = 0
    
    for batch in val_loader:
        try:
            poses, batch_dict = prepare_batch(batch, device)
            if poses is None: continue
            
            gt_latent = ae.encode(poses) / latent_scale
            
            losses = model(batch_dict, gt_latent, pose_gt=poses.detach(), mode="train", prior_scale=1.0)
            
            for k in metrics:
                metrics[k] += safe_float(losses.get(k, 0))
            n += 1
        except Exception: continue
    
    if n == 0: return None
    return {k: v / n for k, v in metrics.items()}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ae_ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./ckpts")
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
    p.add_argument("--max_seq_len", type=int, default=120)
    p.add_argument("--W_SYNC", type=float, default=0.1)
    p.add_argument("--W_LENGTH", type=float, default=0.01)
    p.add_argument("--W_PRIOR", type=float, default=0.1)
    p.add_argument("--prior_warmup_epochs", type=int, default=5, help="Epochs to warmup prior")
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"🚀 Training Config: Data={args.data_dir} | AE={args.ae_ckpt} | Batch={args.batch_size}")

    print("⏳ Loading data...")
    train_dataset = SignLanguageDataset(data_dir=args.data_dir, split="train", max_seq_len=args.max_seq_len)
    
    val_dataset = None
    for split in ["dev", "test", "val"]:
        path = os.path.join(args.data_dir, split)
        if os.path.exists(path):
            val_dataset = SignLanguageDataset(data_dir=args.data_dir, split=split, max_seq_len=args.max_seq_len)
            break
    
    if not val_dataset:
        full = len(train_dataset)
        val_sz = int(full * 0.1)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [full - val_sz, val_sz])
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, collate_fn=collate_fn, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, collate_fn=collate_fn)

    ae = UnifiedPoseAutoencoder(latent_dim=args.latent_dim, hidden_dim=args.ae_hidden_dim).to(device)
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        W_SYNC=args.W_SYNC, W_LENGTH=args.W_LENGTH, W_PRIOR=args.W_PRIOR
    ).to(device)

    ck = torch.load(args.ae_ckpt, map_location=device)
    ae.load_state_dict(ck.get("model_state_dict", ck), strict=False)
    ae.eval()
    
    # 1. Khởi tạo Optimizer TRƯỚC
    optimizer = torch.optim.AdamW(flow_matcher.parameters(), lr=args.lr, weight_decay=1e-6)

    # 2. Sau đó mới Load Checkpoint (Resume)
    start_epoch = 0
    best_val_loss = float('inf')
    latent_scale = 1.0

    if args.flow_ckpt and os.path.exists(args.flow_ckpt):
        print(f"🔄 Resuming: {args.flow_ckpt}")
        ck = torch.load(args.flow_ckpt, map_location=device)
        
        flow_matcher.load_state_dict(ck['model_state_dict'], strict=False)
        
        if 'optimizer_state_dict' in ck:
            try:
                optimizer.load_state_dict(ck['optimizer_state_dict'])
                print("   ✅ Optimizer state loaded!")
            except:
                print("   ⚠️ Could not load optimizer state (architecture mismatch?), starting fresh.")

        start_epoch = ck.get("epoch", 0)
        latent_scale = float(ck.get("latent_scale_factor", 1.0))
        best_val_loss = float(ck.get("val_loss", float('inf')))
    else:
        latent_scale = estimate_scale_factor(ae, train_loader, device)
    
    print(f"📏 Latent Scale: {latent_scale:.6f}")

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-7)
    # Update scheduler về đúng epoch hiện tại
    if start_epoch > 0:
        scheduler.step(start_epoch) 
        
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))
    patience_counter = 0
    
    PRIOR_WARMUP_EPOCHS = args.prior_warmup_epochs

    print("\n🏋️ START TRAINING")
    for epoch in range(start_epoch, args.epochs):
        flow_matcher.train()
        ae.eval()
        
        current_prior_scale = 0.0 if epoch < PRIOR_WARMUP_EPOCHS else 1.0
        
        metrics = {"total": 0.0, "flow": 0.0, "length": 0.0, "sync": 0.0, "prior": 0.0}
        n_batches = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1} [PriorScale={current_prior_scale}]")
        
        for batch in pbar:
            try:
                poses, batch_dict = prepare_batch(batch, device)
                if poses is None: continue
                
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
                
                if torch.isnan(total_loss):
                    print(f"\n⚠️ NaN Loss at Ep {epoch} Batch {n_batches}!")
                    optimizer.zero_grad()
                    continue
                
                scaler.scale(total_loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(flow_matcher.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                for k, v in metrics.items(): metrics[k] += safe_float(losses.get(k, 0))
                n_batches += 1
                pbar.set_postfix({"L": f"{metrics['total']/n_batches:.4f}", "F": f"{metrics['flow']/n_batches:.3f}"})
                
            except Exception as e:
                # In lỗi chi tiết để debug nếu còn lỗi
                print(f"Error: {e}") 
                continue
        
        if n_batches == 0: continue
        
        for k in metrics: writer.add_scalar(f"Train/{k}", metrics[k]/n_batches, epoch)
        
        val_res = validate(flow_matcher, ae, val_loader, device, latent_scale)
        if val_res:
            writer.add_scalar("Val/Total", val_res['total'], epoch)
            print(f"📊 Val: {val_res['total']:.4f} (Flow: {val_res['flow']:.4f})")
            
            if val_res['total'] < best_val_loss:
                best_val_loss = val_res['total']
                patience_counter = 0
                torch.save({
                    "epoch": epoch + 1,
                    "model_state_dict": flow_matcher.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "latent_scale_factor": latent_scale,
                    "val_loss": best_val_loss
                }, os.path.join(args.save_dir, "best_model.pt"))
                print("💾 Saved Best!")
            else:
                patience_counter += 1
                print(f"⏳ Patience: {patience_counter}/{args.patience}")
                if patience_counter >= args.patience: break
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": flow_matcher.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "latent_scale_factor": latent_scale
            }, os.path.join(args.save_dir, f"ckpt_epoch_{epoch+1}.pt"))
            
        scheduler.step(epoch + 1)
    
    writer.close()
    print("✅ DONE")

if __name__ == "__main__":
    main()