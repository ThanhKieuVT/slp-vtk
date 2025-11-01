# training/train_joint_sync.py
"""
Train temporal synchronization module
Requires trained manual and NMM models
(FIXED: Added periodic checkpointing and resume logic)
"""
import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import argparse

import sys
sys.path.append('..')

from models.hierarchical_flow import HierarchicalFlowMatcher
from models.nmm_generator import NMMFlowGenerator
from models.temporal_sync import TemporalSynchronizer
from data.phoenix_dataset import PhoenixFlowDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str,
                       default='../data/RWTH/processed_data/final')
    parser.add_argument('--manual_checkpoint', type=str, required=True,
                       help='Path to trained manual flow checkpoint')
    parser.add_argument('--nmm_checkpoint', type=str, required=True,
                       help='Path to trained NMM flow checkpoint')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/sync')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=5)
    # --- (THÊM LOGIC RESUME) ---
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--eval_every', type=int, default=1) # Thêm eval_every
    # -----------------------------
    parser.add_argument('--wandb_project', type=str, default='slp-hierarchical-flow')
    parser.add_argument('--wandb_run_name', type=str, default='joint_sync')
    return parser.parse_args()

def train():
    args = parse_args()
    
    accelerator = Accelerator(mixed_precision='fp16')
    
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Datasets
    print("Loading datasets...")
    train_dataset = PhoenixFlowDataset(split='train', data_root=args.data_root)
    dev_dataset = PhoenixFlowDataset(split='dev', data_root=args.data_root)
    
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True
    )
    dev_loader = DataLoader(
        dev_dataset, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=args.num_workers
    )
    
    # Load pretrained models
    print("Loading pretrained models...")
    
    # Manual flow (frozen)
    manual_model = HierarchicalFlowMatcher()
    manual_ckpt = torch.load(args.manual_checkpoint, map_location='cpu')
    manual_model.load_state_dict(manual_ckpt['model'])
    manual_model.eval()
    for param in manual_model.parameters():
        param.requires_grad = False
    print("  ✅ Manual flow loaded (frozen)")
    
    # NMM flow (frozen)
    nmm_model = NMMFlowGenerator()
    nmm_ckpt = torch.load(args.nmm_checkpoint, map_location='cpu')
    nmm_model.load_state_dict(nmm_ckpt['model'])
    nmm_model.eval()
    for param in nmm_model.parameters():
        param.requires_grad = False
    print("  ✅ NMM flow loaded (frozen)")
    
    # Synchronization module (trainable)
    sync_model = TemporalSynchronizer()
    
    # Optimizer (only sync model)
    optimizer = torch.optim.AdamW(
        sync_model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.num_epochs, eta_min=1e-6
    )
    
    # Prepare
    # (Lưu ý: 2 model "frozen" vẫn cần .prepare() để đảm bảo data-parallel)
    manual_model, nmm_model, sync_model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        manual_model, nmm_model, sync_model, optimizer, train_loader, dev_loader, scheduler
    )
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # --- (FIX 1: THÊM LOGIC RESUME) ---
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        # Chỉ load state của sync_model và optimizer
        accelerator.unwrap_model(sync_model).load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    # --- (HẾT FIX 1) ---

    # Training
    print("\nStarting joint training...")
    print("="*70)
        
    for epoch in range(start_epoch, args.num_epochs): # Sửa: bắt đầu từ start_epoch
        sync_model.train()
        train_loss_total = 0.0
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Generate poses with frozen models
            with torch.no_grad():
                manual_output = manual_model(batch, mode='inference', num_inference_steps=10)
                nmm_output = nmm_model(batch, mode='inference', num_inference_steps=10)
            
            # Combine manual levels
            manual_seq = manual_output['pose_sequence']  # [B, T, 150]
            
            # Combine NMM components
            nmm_seq = torch.cat([
                nmm_output['facial_aus'],
                nmm_output['head_pose'],
                nmm_output['eye_gaze'],
                nmm_output['mouth_shape']
            ], dim=-1)  # [B, T, 64]
            
            # Ground truth
            gt_manual = torch.cat([
                batch['manual_coarse'],
                batch['manual_medium'],
                batch['manual_fine']
            ], dim=-1)
            
            gt_nmm = torch.cat([
                batch['nmm_facial_aus'],
                batch['nmm_head_pose'],
                batch['nmm_eye_gaze'],
                batch['nmm_mouth_shape']
            ], dim=-1)
            
            # Mask
            # (Lỗi logic nhỏ: T_max phải tính từ GT, vì 'inference' có thể trả T=50)
            T_max = gt_manual.shape[1] 
            mask = torch.arange(T_max, device=gt_manual.device)[None, :] >= batch['seq_lengths'][:, None]
            
            # (Phải crop/pad output của inference cho khớp T_max của GT)
            T_pred = manual_seq.shape[1]
            if T_pred > T_max:
                manual_seq = manual_seq[:, :T_max, :]
                nmm_seq = nmm_seq[:, :T_max, :]
            elif T_pred < T_max:
                pad_manual = torch.zeros(manual_seq.shape[0], T_max - T_pred, manual_seq.shape[2]).to(accelerator.device)
                pad_nmm = torch.zeros(nmm_seq.shape[0], T_max - T_pred, nmm_seq.shape[2]).to(accelerator.device)
                manual_seq = torch.cat([manual_seq, pad_manual], dim=1)
                nmm_seq = torch.cat([nmm_seq, pad_nmm], dim=1)
            
            # Synchronization
            losses = sync_model.compute_sync_loss(
                manual_seq, nmm_seq,
                gt_manual, gt_nmm,
                mask=mask
            )
            
            # Backward
            accelerator.backward(losses['total'])
            accelerator.clip_grad_norm_(sync_model.parameters(), max_norm=1.0)
            optimizer.step()
            optimizer.zero_grad()
            
            train_loss_total += losses['total'].item()
            progress_bar.set_postfix({'loss': losses['total'].item()})
            
            if accelerator.is_main_process and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss_total': losses['total'].item(),
                    'train/loss_manual': losses['manual'].item(),
                    'train/loss_nmm': losses['nmm'].item(),
                    'train/loss_correlation': losses['correlation'].item(),
                    'epoch': epoch + batch_idx / len(train_loader)
                })
        
        avg_train_loss = train_loss_total / len(train_loader)
        
        # --- Validation (thêm eval_every) ---
        if (epoch + 1) % args.eval_every == 0:
            sync_model.eval()
            val_loss_total = 0.0
            
            with torch.no_grad():
                for batch in tqdm(dev_loader, desc="Validating",
                                 disable=not accelerator.is_local_main_process):
                    # (Copy logic giống train loop)
                    manual_output = manual_model(batch, mode='inference', num_inference_steps=10)
                    nmm_output = nmm_model(batch, mode='inference', num_inference_steps=10)
                    
                    manual_seq = manual_output['pose_sequence']
                    nmm_seq = torch.cat([
                        nmm_output['facial_aus'], nmm_output['head_pose'],
                        nmm_output['eye_gaze'], nmm_output['mouth_shape']
                    ], dim=-1)
                    
                    gt_manual = torch.cat([
                        batch['manual_coarse'], batch['manual_medium'], batch['manual_fine']
                    ], dim=-1)
                    gt_nmm = torch.cat([
                        batch['nmm_facial_aus'], batch['nmm_head_pose'],
                        batch['nmm_eye_gaze'], batch['nmm_mouth_shape']
                    ], dim=-1)
                    
                    T_max = gt_manual.shape[1]
                    mask = torch.arange(T_max, device=gt_manual.device)[None, :] >= batch['seq_lengths'][:, None]

                    T_pred = manual_seq.shape[1]
                    if T_pred > T_max:
                        manual_seq = manual_seq[:, :T_max, :]
                        nmm_seq = nmm_seq[:, :T_max, :]
                    elif T_pred < T_max:
                        pad_manual = torch.zeros(manual_seq.shape[0], T_max - T_pred, manual_seq.shape[2]).to(accelerator.device)
                        pad_nmm = torch.zeros(nmm_seq.shape[0], T_max - T_pred, nmm_seq.shape[2]).to(accelerator.device)
                        manual_seq = torch.cat([manual_seq, pad_manual], dim=1)
                        nmm_seq = torch.cat([nmm_seq, pad_nmm], dim=1)

                    losses = sync_model.compute_sync_loss(
                        manual_seq, nmm_seq, gt_manual, gt_nmm, mask=mask
                    )
                    val_loss_total += losses['total'].item()
            
            avg_val_loss = val_loss_total / len(dev_loader)
            
            if accelerator.is_main_process:
                print(f"\nEpoch {epoch+1}:")
                print(f"  Train Loss: {avg_train_loss:.4f}")
                print(f"  Val Loss:   {avg_val_loss:.4f}")
                
                wandb.log({
                    'val/loss_total': avg_val_loss,
                    'epoch': epoch + 1
                })
            
            # Save best
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                if accelerator.is_main_process:
                    # (Lưu state của model được unwrap)
                    torch.save({
                        'epoch': epoch,
                        'model': accelerator.unwrap_model(sync_model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss
                    }, os.path.join(args.output_dir, 'best_model.pt'))
                    print(f"  ✅ Saved best model")
        
        # --- (FIX 2: THÊM LOGIC LƯU "AN TOÀN") ---
        if (epoch + 1) % args.save_every == 0:
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model': accelerator.unwrap_model(sync_model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': vars(args) # (Thêm config cho đầy đủ)
                }, save_path)
                print(f"  💾 Saved checkpoint to {save_path}")
        # --- (HẾT FIX 2) ---
        
        scheduler.step()
        print("="*70)
    
    if accelerator.is_main_process:
        print("\n✅ Joint training complete!")
        # (Lưu file cuối cùng)
        save_path = os.path.join(args.output_dir, 'final_model.pt')
        torch.save({
            'epoch': args.num_epochs - 1,
            'model': accelerator.unwrap_model(sync_model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': vars(args)
        }, save_path)
        print(f"Final model saved to {save_path}")
        wandb.finish()

if __name__ == '__main__':
    train()
