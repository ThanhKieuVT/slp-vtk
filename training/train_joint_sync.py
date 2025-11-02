# training/train_joint_sync.py
"""
Train temporal synchronization module
✅ Auto resume from last/best checkpoint
✅ Periodic checkpointing + best model saving
✅ Safe resume even after Colab disconnects
"""
import os
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import argparse
import glob

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
    parser.add_argument('--manual_checkpoint', type=str, required=True)
    parser.add_argument('--nmm_checkpoint', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='./checkpoints/sync')
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_epochs', type=int, default=30)
    parser.add_argument('--learning_rate', type=float, default=5e-5)
    parser.add_argument('--num_workers', type=int, default=0)  # ✅ Colab-safe
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='slp-hierarchical-flow')
    parser.add_argument('--wandb_run_name', type=str, default='joint_sync')
    return parser.parse_args()


def find_latest_checkpoint(output_dir):
    """Tự động tìm checkpoint mới nhất trong thư mục."""
    ckpts = sorted(glob.glob(os.path.join(output_dir, 'checkpoint_epoch_*.pt')))
    return ckpts[-1] if ckpts else None


def train():
    args = parse_args()
    accelerator = Accelerator(mixed_precision='fp16')

    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # === Load datasets ===
    print("Loading datasets...")
    train_dataset = PhoenixFlowDataset(split='train', data_root=args.data_root)
    dev_dataset = PhoenixFlowDataset(split='dev', data_root=args.data_root)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    # === Load pretrained frozen models ===
    print("Loading pretrained models...")
    manual_model = HierarchicalFlowMatcher()
    manual_model.load_state_dict(torch.load(args.manual_checkpoint, map_location='cpu')['model'])
    manual_model.eval()
    for p in manual_model.parameters(): p.requires_grad = False

    nmm_model = NMMFlowGenerator()
    nmm_model.load_state_dict(torch.load(args.nmm_checkpoint, map_location='cpu')['model'])
    nmm_model.eval()
    for p in nmm_model.parameters(): p.requires_grad = False

    sync_model = TemporalSynchronizer()

    optimizer = torch.optim.AdamW(sync_model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    manual_model, nmm_model, sync_model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        manual_model, nmm_model, sync_model, optimizer, train_loader, dev_loader, scheduler
    )

    os.makedirs(args.output_dir, exist_ok=True)

    # === Resume logic ===
    start_epoch = 0
    best_val_loss = float('inf')
    resume_path = args.resume_from or find_latest_checkpoint(args.output_dir)

    if resume_path and os.path.exists(resume_path):
        print(f"🔁 Resuming from {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu')
        accelerator.unwrap_model(sync_model).load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"Resumed at epoch {start_epoch}, best_val_loss={best_val_loss:.4f}")

    print("\n🚀 Starting training...")
    print("=" * 70)

    for epoch in range(start_epoch, args.num_epochs):
        sync_model.train()
        train_loss = 0.0

        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}",
                        disable=not accelerator.is_local_main_process)

        for batch in progress:
            with torch.no_grad():
                manual_out = manual_model(batch, mode='inference', num_inference_steps=10)
                nmm_out = nmm_model(batch, mode='inference', num_inference_steps=10)

            manual_seq = manual_out['pose_sequence']
            nmm_seq = torch.cat([
                nmm_out['facial_aus'], nmm_out['head_pose'],
                nmm_out['eye_gaze'], nmm_out['mouth_shape']
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

            # Align T dimension
            T_pred = manual_seq.shape[1]
            if T_pred > T_max:
                manual_seq, nmm_seq = manual_seq[:, :T_max], nmm_seq[:, :T_max]
            elif T_pred < T_max:
                pad_m = torch.zeros(manual_seq.shape[0], T_max - T_pred, manual_seq.shape[2]).to(accelerator.device)
                pad_n = torch.zeros(nmm_seq.shape[0], T_max - T_pred, nmm_seq.shape[2]).to(accelerator.device)
                manual_seq, nmm_seq = torch.cat([manual_seq, pad_m], 1), torch.cat([nmm_seq, pad_n], 1)

            losses = sync_model.compute_sync_loss(manual_seq, nmm_seq, gt_manual, gt_nmm, mask=mask)
            accelerator.backward(losses['total'])
            accelerator.clip_grad_norm_(sync_model.parameters(), max_norm=1.0)
            optimizer.step(); optimizer.zero_grad()

            train_loss += losses['total'].item()
            progress.set_postfix({'loss': losses['total'].item()})

            if accelerator.is_main_process:
                wandb.log({'train/loss_total': losses['total'].item(),
                           'epoch': epoch + len(progress)/len(train_loader)})

        avg_train = train_loss / len(train_loader)

        # === Validation ===
        if (epoch + 1) % args.eval_every == 0:
            sync_model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for batch in tqdm(dev_loader, desc="Validating", disable=not accelerator.is_local_main_process):
                    manual_out = manual_model(batch, mode='inference', num_inference_steps=10)
                    nmm_out = nmm_model(batch, mode='inference', num_inference_steps=10)
                    manual_seq = manual_out['pose_sequence']
                    nmm_seq = torch.cat([
                        nmm_out['facial_aus'], nmm_out['head_pose'],
                        nmm_out['eye_gaze'], nmm_out['mouth_shape']
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
                    losses = sync_model.compute_sync_loss(manual_seq, nmm_seq, gt_manual, gt_nmm, mask=mask)
                    val_loss += losses['total'].item()
            avg_val = val_loss / len(dev_loader)

            if accelerator.is_main_process:
                print(f"\nEpoch {epoch+1}: Train={avg_train:.4f} | Val={avg_val:.4f}")
                wandb.log({'val/loss_total': avg_val, 'epoch': epoch + 1})

            # === Save best ===
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                if accelerator.is_main_process:
                    torch.save({
                        'epoch': epoch,
                        'model': accelerator.unwrap_model(sync_model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'best_val_loss': best_val_loss
                    }, os.path.join(args.output_dir, 'best_model.pt'))
                    print(f"✅ Saved best model (val_loss={best_val_loss:.4f})")

        # === Save checkpoint periodically ===
        if (epoch + 1) % args.save_every == 0 and accelerator.is_main_process:
            ckpt_path = os.path.join(args.output_dir, f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model': accelerator.unwrap_model(sync_model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'best_val_loss': best_val_loss,
                'config': vars(args)
            }, ckpt_path)
            print(f"💾 Saved checkpoint: {ckpt_path}")

        scheduler.step()
        print("=" * 70)

    # === Final save ===
    if accelerator.is_main_process:
        torch.save({
            'epoch': args.num_epochs - 1,
            'model': accelerator.unwrap_model(sync_model).state_dict(),
            'optimizer': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'config': vars(args)
        }, os.path.join(args.output_dir, 'final_model.pt'))
        print("\n🎯 Training complete. Final model saved.")
        wandb.finish()


if __name__ == '__main__':
    train()
