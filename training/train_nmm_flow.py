# training/train_nmm_flow.py
"""
Train NMM flow models
Can share text encoder with manual branch
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

from models.nmm_generator import NMMFlowGenerator
from models.hierarchical_flow import HierarchicalFlowMatcher
from data.phoenix_dataset import PhoenixFlowDataset, collate_fn

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, 
                       default='../data/RWTH/processed_data/final')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/nmm_flow')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--share_text_encoder', type=str, default=None,
                       help='Path to manual flow checkpoint to share text encoder')
    parser.add_argument('--freeze_text_encoder', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='slp-hierarchical-flow')
    parser.add_argument('--wandb_run_name', type=str, default='nmm_flow')
    return parser.parse_args()

def train():
    args = parse_args()
    
    # Setup
    accelerator = Accelerator(mixed_precision='fp16')
    
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
    
    shared_text_encoder = None
    if args.share_text_encoder:
        # Load text encoder from manual flow checkpoint
        print(f"Loading text encoder from {args.share_text_encoder}")
        manual_checkpoint = torch.load(args.share_text_encoder, map_location='cpu')
        manual_model = HierarchicalFlowMatcher()
        manual_model.load_state_dict(manual_checkpoint['model'])
        shared_text_encoder = manual_model.text_encoder
        print("Text encoder loaded and will be shared")
    
    model = NMMFlowGenerator(
        freeze_text_encoder=args.freeze_text_encoder,
        share_text_encoder=shared_text_encoder
    )
    
    # Optimizer (only optimize NMM flow parameters)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=0.01
    )
    
    # Scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=args.num_epochs,
        eta_min=1e-6
    )
    
    # Prepare
    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, scheduler
    )
    
    # Resume
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"Resuming from {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Training loop
    print(f"\nStarting training from epoch {start_epoch}...")
    print("="*70)
    
    for epoch in range(start_epoch, args.num_epochs):
        # Training
        model.train()
        train_losses = {
            'total': 0.0,
            'facial_au': 0.0,
            'head_pose': 0.0,
            'eye_gaze': 0.0,
            'mouth_shape': 0.0
        }
        
        progress_bar = tqdm(
            train_loader,
            desc=f"Epoch {epoch+1}/{args.num_epochs}",
            disable=not accelerator.is_local_main_process
        )
        
        for batch_idx, batch in enumerate(progress_bar):
            losses = model(batch, mode='train')
            
            accelerator.backward(losses['total'])
            
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            optimizer.zero_grad()
            
            for key in train_losses:
                train_losses[key] += losses[key].item()
            
            progress_bar.set_postfix({
                'loss': losses['total'].item(),
                'au': losses['facial_au'].item(),
                'head': losses['head_pose'].item()
            })
            
            if accelerator.is_main_process and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss_total': losses['total'].item(),
                    'train/loss_facial_au': losses['facial_au'].item(),
                    'train/loss_head_pose': losses['head_pose'].item(),
                    'train/loss_eye_gaze': losses['eye_gaze'].item(),
                    'train/loss_mouth_shape': losses['mouth_shape'].item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'epoch': epoch + batch_idx / len(train_loader)
                })
        
        for key in train_losses:
            train_losses[key] /= len(train_loader)
        
        # Validation
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            val_losses = {
                'total': 0.0,
                'facial_au': 0.0,
                'head_pose': 0.0,
                'eye_gaze': 0.0,
                'mouth_shape': 0.0
            }
            
            with torch.no_grad():
                for batch in tqdm(dev_loader, desc="Validating",
                                 disable=not accelerator.is_local_main_process):
                    losses = model(batch, mode='train')
                    for key in val_losses:
                        val_losses[key] += losses[key].item()
            
            for key in val_losses:
                val_losses[key] /= len(dev_loader)
            
            if accelerator.is_main_process:
                print(f"\nEpoch {epoch+1} Summary:")
                print(f"  Train Loss: {train_losses['total']:.4f}")
                print(f"  Val Loss:   {val_losses['total']:.4f}")
                
                wandb.log({
                    'val/loss_total': val_losses['total'],
                    'val/loss_facial_au': val_losses['facial_au'],
                    'val/loss_head_pose': val_losses['head_pose'],
                    'val/loss_eye_gaze': val_losses['eye_gaze'],
                    'val/loss_mouth_shape': val_losses['mouth_shape'],
                    'epoch': epoch + 1
                })
            
            # Save best
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
        
        # Save checkpoint
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
                print(f"  💾 Saved checkpoint")
        
        scheduler.step()
        print("="*70)
    
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, 'final_model.pt')
        torch.save({
            'epoch': args.num_epochs - 1,
            'model': accelerator.unwrap_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': vars(args)
        }, save_path)
        print(f"\n✅ Training complete!")
        wandb.finish()

if __name__ == '__main__':
    train()