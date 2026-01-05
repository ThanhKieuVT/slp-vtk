#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
STAGE 3: STUDENT DISTILLATION
✅ FIXED: max_seq_len = 400, ensure latent_scale consistency
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
from torch.utils.tensorboard import SummaryWriter
import numpy as np

sys.path.append(os.getcwd())
from dataset import SignLanguageDataset, collate_fn
from models.fml.autoencoder import UnifiedPoseAutoencoder
from models.fml.latent_flow_matcher import LatentFlowMatcher

def safe_float(x):
    if isinstance(x, (int, float, np.floating)):
        return float(x)
    elif hasattr(x, "item"):
        return x.item()
    else:
        return float(x)

def prepare_batch(batch, device):
    if isinstance(batch, dict):
        for k, v in batch.items():
            if isinstance(v, torch.Tensor): 
                batch[k] = v.to(device)
        
        poses = batch.get("poses")
        if poses is None:
            return None, None
            
        if "seq_lengths" not in batch:
            B, T, _ = poses.shape
            batch["seq_lengths"] = torch.full((B,), T, device=device, dtype=torch.long)
            
        return poses, batch
    else:
        if len(batch) < 3:
            return None, None
            
        poses = batch[0].to(device)
        B, T, _ = poses.shape
        
        if len(batch) >= 4:
            seq_lens = batch[3].to(device)
        else:
            seq_lens = torch.full((B,), T, device=device, dtype=torch.long)
        
        return poses, {
            'text_tokens': batch[1].to(device), 
            'attention_mask': batch[2].to(device), 
            'seq_lengths': seq_lens
        }

def distillation_loss(student_out, teacher_out, alpha=0.5, beta=0.5, gamma=0.3):
    """Multi-level distillation loss"""
    L_task = F.mse_loss(
        student_out['velocity_pred'], 
        student_out['velocity_target']
    )
    
    L_latent = F.mse_loss(
        student_out['predicted_latent'],
        teacher_out['predicted_latent'].detach()
    )
    
    L_velocity = F.mse_loss(
        student_out['velocity_pred'],
        teacher_out['velocity_pred'].detach()
    )
    
    L_v_flow = F.mse_loss(
        student_out['v_flow'],
        teacher_out['v_flow'].detach()
    )
    
    total = alpha * L_task + beta * L_latent + gamma * L_velocity + 0.1 * L_v_flow
    
    return total, {
        'task': safe_float(L_task),
        'latent': safe_float(L_latent),
        'velocity': safe_float(L_velocity),
        'v_flow': safe_float(L_v_flow)
    }

@torch.no_grad()
def validate(student, teacher, ae, val_loader, device, latent_scale, alpha, beta, gamma):
    student.eval()
    teacher.eval()
    
    metrics = {
        'total': 0.0,
        'task': 0.0,
        'latent': 0.0,
        'velocity': 0.0,
        'v_flow': 0.0,
        'student_flow': 0.0,
        'teacher_flow': 0.0
    }
    n = 0
    
    for batch in val_loader:
        try:
            poses, batch_dict = prepare_batch(batch, device)
            if poses is None:
                continue
            
            gt_latent = ae.encode(poses) / latent_scale
            
            teacher_out = teacher(batch_dict, gt_latent, pose_gt=poses.detach(), mode="train", prior_scale=1.0)
            student_out = student(batch_dict, gt_latent, pose_gt=poses.detach(), mode="train", prior_scale=1.0)
            
            loss, loss_dict = distillation_loss(student_out, teacher_out, alpha, beta, gamma)
            
            metrics['total'] += loss.item()
            metrics['task'] += loss_dict['task']
            metrics['latent'] += loss_dict['latent']
            metrics['velocity'] += loss_dict['velocity']
            metrics['v_flow'] += loss_dict['v_flow']
            metrics['student_flow'] += safe_float(student_out.get('flow', 0))
            metrics['teacher_flow'] += safe_float(teacher_out.get('flow', 0))
            n += 1
            
        except Exception as e:
            print(f"Validation error: {e}")
            continue
    
    if n == 0:
        return None
    
    return {k: v / n for k, v in metrics.items()}

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--ae_ckpt", type=str, required=True)
    p.add_argument("--teacher_ckpt", type=str, required=True)
    p.add_argument("--save_dir", type=str, default="./ckpts_stage3_student")
    p.add_argument("--student_ckpt", type=str, default=None)
    
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--ae_hidden_dim", type=int, default=512)
    p.add_argument("--teacher_hidden", type=int, default=768)
    p.add_argument("--student_hidden", type=int, default=384)
    
    p.add_argument("--epochs", type=int, default=150)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_seq_len", type=int, default=400)  # ✅ FIXED: 120 → 400
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--beta", type=float, default=0.5)
    p.add_argument("--gamma", type=float, default=0.3)
    
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    print(f"🚀 STAGE 3: STUDENT DISTILLATION")
    print(f"   Teacher hidden_dim: {args.teacher_hidden}")
    print(f"   Student hidden_dim: {args.student_hidden}")
    print(f"   Max seq len: {args.max_seq_len}")  # ✅ Log max_seq_len
    print(f"   Distillation weights: α={args.alpha}, β={args.beta}, γ={args.gamma}")

    print("⏳ Loading data...")
    train_dataset = SignLanguageDataset(
        data_dir=args.data_dir, 
        split="train", 
        max_seq_len=args.max_seq_len
    )
    
    val_dataset = None
    for split in ["dev", "test", "val"]:
        path = os.path.join(args.data_dir, split)
        if os.path.exists(path):
            val_dataset = SignLanguageDataset(
                data_dir=args.data_dir, 
                split=split, 
                max_seq_len=args.max_seq_len
            )
            break
    
    if not val_dataset:
        full = len(train_dataset)
        val_sz = int(full * 0.1)
        train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [full - val_sz, val_sz])
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=2,
        collate_fn=collate_fn, 
        drop_last=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn
    )

    print("📦 Loading models...")
    
    # Autoencoder (Frozen)
    ae = UnifiedPoseAutoencoder(
        latent_dim=args.latent_dim, 
        hidden_dim=args.ae_hidden_dim
    ).to(device)
    ae_ckpt = torch.load(args.ae_ckpt, map_location=device)
    ae.load_state_dict(ae_ckpt['model_state_dict'], strict=False)
    ae.eval()
    for param in ae.parameters():
        param.requires_grad = False
    print("   ✅ Autoencoder loaded (frozen)")
    
    # Teacher (Frozen)
    teacher = LatentFlowMatcher(
        latent_dim=args.latent_dim, 
        hidden_dim=args.teacher_hidden
    ).to(device)
    teacher_ckpt = torch.load(args.teacher_ckpt, map_location=device)
    teacher.load_state_dict(teacher_ckpt['model_state_dict'], strict=False)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad = False
    
    # ✅ Load latent_scale từ teacher checkpoint
    latent_scale = float(teacher_ckpt.get("latent_scale_factor", 1.0))
    print(f"   ✅ Teacher loaded (frozen), latent_scale={latent_scale:.6f}")
    
    if latent_scale == 1.0:
        print("   ⚠️ WARNING: latent_scale = 1.0, teacher checkpoint may not have saved it!")
    
    # Student (Trainable)
    student = LatentFlowMatcher(
        latent_dim=args.latent_dim, 
        hidden_dim=args.student_hidden
    ).to(device)
    student.train()
    print(f"   ✅ Student initialized (trainable)")
    
    teacher_params = sum(p.numel() for p in teacher.parameters())
    student_params = sum(p.numel() for p in student.parameters())
    print(f"   📊 Teacher params: {teacher_params:,} | Student params: {student_params:,}")
    print(f"   📊 Compression ratio: {teacher_params / student_params:.2f}x")

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr, weight_decay=1e-6)
    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=2, eta_min=1e-7)
    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    
    start_epoch = 0
    best_val_loss = float('inf')
    patience_counter = 0
    
    if args.student_ckpt and os.path.exists(args.student_ckpt):
        print(f"📄 Resuming from: {args.student_ckpt}")
        ckpt = torch.load(args.student_ckpt, map_location=device)
        student.load_state_dict(ckpt['model_state_dict'], strict=False)
        
        if 'optimizer_state_dict' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                print("   ✅ Optimizer state loaded")
            except:
                print("   ⚠️ Could not load optimizer state")
        
        start_epoch = ckpt.get("epoch", 0)
        best_val_loss = float(ckpt.get("val_loss", float('inf')))
        print(f"   Resuming from epoch {start_epoch}")
    
    if start_epoch > 0:
        scheduler.step(start_epoch)
    
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, "logs"))
    
    print("\n🏋️ START DISTILLATION TRAINING")
    for epoch in range(start_epoch, args.epochs):
        student.train()
        teacher.eval()
        ae.eval()
        
        metrics = {
            'total': 0.0,
            'task': 0.0,
            'latent': 0.0,
            'velocity': 0.0,
            'v_flow': 0.0
        }
        n_batches = 0
        
        pbar = tqdm(train_loader, desc=f"[STUDENT] Epoch {epoch+1}/{args.epochs}")
        
        for batch in pbar:
            try:
                poses, batch_dict = prepare_batch(batch, device)
                if poses is None:
                    continue
                
                with torch.no_grad():
                    gt_latent = ae.encode(poses) / latent_scale
                    
                    teacher_out = teacher(
                        batch_dict, 
                        gt_latent, 
                        pose_gt=poses.detach(), 
                        mode="train",
                        prior_scale=1.0
                    )
                
                optimizer.zero_grad()
                
                with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
                    student_out = student(
                        batch_dict,
                        gt_latent,
                        pose_gt=poses.detach(),
                        mode="train",
                        prior_scale=1.0
                    )
                    
                    loss, loss_dict = distillation_loss(
                        student_out, 
                        teacher_out,
                        args.alpha,
                        args.beta,
                        args.gamma
                    )
                
                if torch.isnan(loss):
                    print(f"\n⚠️ NaN loss detected, skipping batch")
                    optimizer.zero_grad()
                    continue
                
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(student.parameters(), args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                
                for k in metrics:
                    metrics[k] += loss_dict.get(k, 0)
                n_batches += 1
                
                pbar.set_postfix({
                    'Loss': f"{metrics['total']/n_batches:.4f}",
                    'Task': f"{metrics['task']/n_batches:.3f}",
                    'Vel': f"{metrics['velocity']/n_batches:.3f}"
                })
                
            except Exception as e:
                print(f"\n❌ Error in batch: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        if n_batches == 0:
            print(f"⚠️ No valid batches in epoch {epoch+1}")
            continue
        
        for k in metrics:
            writer.add_scalar(f"Train/{k}", metrics[k]/n_batches, epoch)
        
        val_res = validate(
            student, teacher, ae, val_loader, device, latent_scale,
            args.alpha, args.beta, args.gamma
        )
        
        if val_res:
            writer.add_scalar("Val/Total", val_res['total'], epoch)
            writer.add_scalar("Val/StudentFlow", val_res['student_flow'], epoch)
            writer.add_scalar("Val/TeacherFlow", val_res['teacher_flow'], epoch)
            
            retention = (1 - abs(val_res['student_flow'] - val_res['teacher_flow']) / val_res['teacher_flow']) * 100
            retention = max(0, min(100, retention))
            
            print(f"\n📊 Validation Results:")
            print(f"   Total Loss: {val_res['total']:.4f}")
            print(f"   Task: {val_res['task']:.4f} | Latent: {val_res['latent']:.4f} | Velocity: {val_res['velocity']:.4f}")
            print(f"   Student Flow: {val_res['student_flow']:.4f} | Teacher Flow: {val_res['teacher_flow']:.4f}")
            print(f"   Quality Retention: {retention:.2f}%")
            
            if val_res['total'] < best_val_loss:
                best_val_loss = val_res['total']
                patience_counter = 0
                
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': student.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'latent_scale_factor': latent_scale,  # ✅ Save same scale as teacher
                    'val_loss': best_val_loss,
                    'hidden_dim': args.student_hidden,
                    'teacher_hidden_dim': args.teacher_hidden,
                    'max_seq_len': args.max_seq_len,
                    'distillation_params': {
                        'alpha': args.alpha,
                        'beta': args.beta,
                        'gamma': args.gamma
                    },
                    'retention': retention
                }, os.path.join(args.save_dir, "best_student.pt"))
                
                print(f"   💾 Saved best model!")
            else:
                patience_counter += 1
                print(f"   ⏳ Patience: {patience_counter}/{args.patience}")
                
                if patience_counter >= args.patience:
                    print(f"\n⏹️ Early stopping triggered!")
                    break
        
        if (epoch + 1) % 10 == 0:
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': student.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'latent_scale_factor': latent_scale,
                'hidden_dim': args.student_hidden,
                'max_seq_len': args.max_seq_len
            }, os.path.join(args.save_dir, f"student_epoch_{epoch+1}.pt"))
        
        scheduler.step(epoch + 1)
    
    writer.close()
    print("\n✅ DISTILLATION TRAINING COMPLETED!")
    print(f"   Best validation loss: {best_val_loss:.4f}")
    print(f"   Model saved to: {args.save_dir}/best_student.pt")

if __name__ == "__main__":
    main()