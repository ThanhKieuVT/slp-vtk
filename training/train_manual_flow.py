# training/train_manual_flow.py
"""
Train hierarchical manual flow models
Can train all 3 levels together or separately
"""
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import argparse

import sys
sys.path.append('..')

from models.hierarchical_flow import HierarchicalFlowMatcher
from data.phoenix_dataset import PhoenixFlowDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                       default='../data/RWTH/processed_data/final')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/manual_flow')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--freeze_text_encoder', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='slp-hierarchical-flow')
    parser.add_argument('--wandb_run_name', type=str, default='manual_flow')
    return parser.parse_args()

def train():
    args = parse_args()
    
    # Setup accelerator
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=2
    )
    
    # WandB logging (only on main process)
    if accelerator.is_main_process:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )
    
    # Datasets
    print("Loading datasets...")
    train_dataset = PhoenixFlowDataset(split='train', data_root=args.data_root, augment=True)
    dev_dataset = PhoenixFlowDataset(split='dev', data_root=args.data_root)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers
    )
    
    # Model
    print("Building model...")
    model = HierarchicalFlowMatcher(
        freeze_text_encoder=args.freeze_text_encoder
    )
    
    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # LR scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # Prepare for distributed training
    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, scheduler)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"Resumed from epoch {start_epoch}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print("="*70)
    
    for epoch in range(start_epoch, args.num_epochs):
        # ========== TRAINING ==========
        model.train()
        train_losses = {
            'total': 0.0,
            'coarse': 0.0,
            'medium': 0.0,
            'fine': 0.0
        }
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            losses = model(batch, mode='train')
            
            # Backward
            accelerator.backward(losses['total'])
            
            # Gradient clipping
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Optimizer step
            optimizer.step()
            optimizer.zero_grad()
            
            # Accumulate losses
            for key in train_losses:
                train_losses[key] += losses[key].item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': losses['total'].item(),
                'coarse': losses['coarse'].item(),
                'medium': losses['medium'].item(),
                'fine': losses['fine'].item()
            })
            
            # Log to wandb
            if accelerator.is_main_process and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss_total': losses['total'].item(),
                    'train/loss_coarse': losses['coarse'].item(),
                    'train/loss_medium': losses['medium'].item(),
                    'train/loss_fine': losses['fine'].item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'epoch': epoch + batch_idx / len(train_loader)
                })
        
        # Average training losses
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        # ========== VALIDATION ==========
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            val_losses = {
                'total': 0.0,
                'coarse': 0.0,
                'medium': 0.0,
                'fine': 0.0
            }
            
            with torch.no_grad():
                for batch in tqdm(dev_loader, desc="Validating", 
                                 disable=not accelerator.is_local_main_process):
                    losses = model(batch, mode='train')
                    
                    for key in val_losses:
                        val_losses[key] += losses[key].item()
            
            # Average validation losses
            for key in val_losses:
                val_losses[key] /= len(dev_loader)
            
            # Print epoch summary
            if accelerator.is_main_process:
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Train Loss: {train_losses['total']:.4f} "
                      f"(C: {train_losses['coarse']:.4f}, "
                      f"M: {train_losses['medium']:.4f}, "
                      f"F: {train_losses['fine']:.4f})")
                print(f"  Val Loss:   {val_losses['total']:.4f} "
                      f"(C: {val_losses['coarse']:.4f}, "
                      f"M: {val_losses['medium']:.4f}, "
                      f"F: {val_losses['fine']:.4f})")
                
                # Log to wandb
                wandb.log({
                    'val/loss_total': val_losses['total'],
                    'val/loss_coarse': val_losses['coarse'],
                    'val/loss_medium': val_losses['medium'],
                    'val/loss_fine': val_losses['fine'],
                    'epoch': epoch + 1
                })
            
            # Save best model
            if val_losses['total'] < best_val_loss:
                best_val_loss = val_losses['total']
                if accelerator.is_main_process:
                    save_path = os.path.join(args.output_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'config': vars(args)
                    }, save_path)
                    print(f"  ✅ Saved best model (val_loss: {best_val_loss:.4f})")
        
        # ========== SAVE CHECKPOINT ==========
        if (epoch + 1) % args.save_every == 0:
            if accelerator.is_main_process:
                save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
                torch.save({
                    'epoch': epoch,
                    'model': accelerator.unwrap_model(model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'best_val_loss': best_val_loss,
                    'config': vars(args)
                }, save_path)
                print(f"  💾 Saved checkpoint to {save_path}")
        
        # LR scheduler step
        scheduler.step()
        
        print("="*70)
    
    # Final save
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, 'final_model.pt')
        torch.save({
            'epoch': args.num_epochs - 1,
            'model': accelerator.unwrap_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': vars(args)
        }, save_path)
        print(f"\n✅ Training complete! Final model saved to {save_path}")
        wandb.finish()

if __name__ == '__main__':
    train()