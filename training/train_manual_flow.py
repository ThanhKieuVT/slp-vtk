# training/train_manual_flow_colab.py
"""
Train hierarchical manual flow models
(Colab-optimized version: single GPU, fp16, wandb optional)
"""

import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse

# ====== Optional WandB ======
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("⚠️ WandB not found. Logging disabled.")

from accelerate import Accelerator

# Add project root to path
sys.path.append('..')

# ====== Import custom modules ======
from models.hierarchical_flow import HierarchicalFlowMatcher
from data.phoenix_dataset import PhoenixFlowDataset, collate_fn


# ========== Parse arguments ==========
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/RWTH/processed_data/final')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/manual_flow')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)  # ✅ 0 cho Colab
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--freeze_text_encoder', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='slp-hierarchical-flow')
    parser.add_argument('--wandb_run_name', type=str, default='manual_flow')
    parser.add_argument('--disable_wandb', action='store_true', help='Disable WandB logging')
    return parser.parse_args()


# ========== Main training ==========
def train():
    args = parse_args()

    # Create output dir
    os.makedirs(args.output_dir, exist_ok=True)

    # ====== Accelerator setup ======
    accelerator = Accelerator(
        mixed_precision='fp16',
        gradient_accumulation_steps=2
    )

    # ====== WandB setup ======
    use_wandb = WANDB_AVAILABLE and not args.disable_wandb and accelerator.is_main_process
    if use_wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args)
        )

    # ====== Dataset ======
    print("📦 Loading datasets...")
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

    # ====== Model ======
    print("🧠 Building model...")
    model = HierarchicalFlowMatcher(freeze_text_encoder=args.freeze_text_encoder)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    # Prepare with accelerator
    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, scheduler
    )

    # ====== Resume checkpoint ======
    start_epoch, best_val_loss = 0, float('inf')
    if args.resume_from and os.path.exists(args.resume_from):
        print(f"🔁 Resuming from checkpoint: {args.resume_from}")
        checkpoint = torch.load(args.resume_from, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        print(f"✅ Resumed from epoch {start_epoch}")

    # ====== Training loop ======
    print(f"\n🚀 Starting training at epoch {start_epoch}")
    print("=" * 70)

    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        train_losses = {'total': 0.0, 'coarse': 0.0, 'medium': 0.0, 'fine': 0.0}

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}", disable=not accelerator.is_local_main_process)

        for batch_idx, batch in enumerate(progress_bar):
            # Forward pass
            losses = model(batch, mode='train')

            # Backward
            accelerator.backward(losses['total'])
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)

            optimizer.step()
            optimizer.zero_grad()

            for key in train_losses:
                train_losses[key] += losses[key].item()

            progress_bar.set_postfix({k: f"{v.item():.4f}" for k, v in losses.items()})

            # WandB log
            if use_wandb and batch_idx % 10 == 0:
                wandb.log({
                    'train/loss_total': losses['total'].item(),
                    'train/lr': optimizer.param_groups[0]['lr'],
                    'epoch': epoch + batch_idx / len(train_loader)
                })

        for k in train_losses:
            train_losses[k] /= len(train_loader)

        # ====== Validation ======
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            val_losses = {'total': 0.0, 'coarse': 0.0, 'medium': 0.0, 'fine': 0.0}

            with torch.no_grad():
                for batch in tqdm(dev_loader, desc="Validating", disable=not accelerator.is_local_main_process):
                    losses = model(batch, mode='train')
                    for k in val_losses:
                        val_losses[k] += losses[k].item()

            for k in val_losses:
                val_losses[k] /= len(dev_loader)

            if accelerator.is_main_process:
                print(f"\n📊 Epoch {epoch+1}")
                print(f"Train Loss: {train_losses['total']:.4f} | Val Loss: {val_losses['total']:.4f}")
                print("=" * 70)

                if use_wandb:
                    wandb.log({
                        'val/loss_total': val_losses['total'],
                        'epoch': epoch + 1
                    })

                # Save best model
                if val_losses['total'] < best_val_loss:
                    best_val_loss = val_losses['total']
                    save_path = os.path.join(args.output_dir, 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss,
                        'config': vars(args)
                    }, save_path)
                    print(f"💾 Saved best model (val_loss={best_val_loss:.4f})")

        # ====== Periodic save ======
        if (epoch + 1) % args.save_every == 0 and accelerator.is_main_process:
            save_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': vars(args)
            }, save_path)
            print(f"💾 Saved checkpoint: {save_path}")

        scheduler.step()

    # ====== Final save ======
    if accelerator.is_main_process:
        save_path = os.path.join(args.output_dir, 'final_model.pt')
        torch.save({
            'epoch': args.num_epochs - 1,
            'model': accelerator.unwrap_model(model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': vars(args)
        }, save_path)
        print(f"✅ Training complete. Final model saved to {save_path}")
        if use_wandb:
            wandb.finish()


if __name__ == '__main__':
    train()
