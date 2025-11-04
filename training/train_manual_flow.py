"""
Train Manual Flow model with resume + best checkpoint saving (Optimized)
"""
import os
import glob
import torch
from torch.utils.data import DataLoader
from accelerate import Accelerator
from tqdm import tqdm
import wandb
import argparse
import sys
from torch.utils.data import Subset
import numpy as np
sys.path.append('..')

from models.hierarchical_flow import HierarchicalFlowMatcher
from data.phoenix_dataset import PhoenixFlowDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/RWTH/processed_data/final')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/manual_flow')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--wandb_project', type=str, default='slp-hierarchical-flow')
    parser.add_argument('--wandb_run_name', type=str, default='manual_flow')
    return parser.parse_args()


# =================================================================
# === ĐÃ SỬA LỖI SẮP XẾP (BUG FIX) ===
# =================================================================
def find_latest_checkpoint(output_dir):
    """
    Finds the latest checkpoint (highest epoch number) in the output directory.
    Sorts numerically, not alphabetically (e.g., 10 > 5).
    """
    ckpts = glob.glob(os.path.join(output_dir, 'checkpoint_epoch_*.pt'))
    if not ckpts:
        return None
    
    # Sắp xếp dựa trên SỐ epoch (int) thay vì tên file (string)
    # Tách số từ tên file: '.../checkpoint_epoch_10.pt' -> 10
    ckpts.sort(key=lambda f: int(f.split('_')[-1].split('.')[0]))
    
    # Trả về file cuối cùng (có epoch lớn nhất)
    return ckpts[-1]
# =================================================================


def train():
    args = parse_args()
    accelerator = Accelerator(mixed_precision='fp16')

    # === Init W&B ===
    if accelerator.is_main_process:
        wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))

    # === Dataset ===
    #print("Loading datasets...")
    #train_dataset = PhoenixFlowDataset(split='train', data_root=args.data_root, augment=True)
    #dev_dataset = PhoenixFlowDataset(split='dev', data_root=args.data_root)
    #train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                            #  collate_fn=collate_fn, num_workers=args.num_workers)
    #dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
    #                        collate_fn=collate_fn, num_workers=args.num_workers)
# === Dataset ===
    print("Loading datasets...")
    # 1. Load full data nhưng đổi tên biến
    train_dataset_full = PhoenixFlowDataset(split='train', data_root=args.data_root, augment=True) 
    dev_dataset = PhoenixFlowDataset(split='dev', data_root=args.data_root)

    # === (THÊM MỚI) TẠO SUBSET ĐỂ CHẠY THỬ ===
    # Chị có thể đổi số 0.1 (10%) thành số khác, ví dụ 0.2 (20%)
    subset_percentage = 0.1 
    
    num_train_samples = int(len(train_dataset_full) * subset_percentage)
    print(f"--- ⚠️  ĐANG CHẠY TEST TRÊN SUBSET ---")
    print(f"--- Chỉ sử dụng {num_train_samples} / {len(train_dataset_full)} mẫu training. ---")
    
    # Lấy ngẫu nhiên các index để train
    indices = np.random.permutation(len(train_dataset_full))[:num_train_samples]
    # Tạo dataset con
    train_dataset = Subset(train_dataset_full, indices)
    # ==========================================

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers)
    # === Model ===
    print("Building HierarchicalFlowMatcher...")
    model = HierarchicalFlowMatcher()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, scheduler
    )

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_best = os.path.join(args.output_dir, "best_model.pt")

    # === Resume logic ===
    start_epoch, best_val_loss = 0, float('inf')
    resume_path = args.resume_from or find_latest_checkpoint(args.output_dir) or ckpt_best
    if resume_path and os.path.exists(resume_path):
        print(f"🔁 Resuming from: {resume_path}")
        ckpt = torch.load(resume_path, map_location='cpu')
        accelerator.unwrap_model(model).load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        if 'scheduler' in ckpt:
            scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"✅ Resume thành công từ epoch {start_epoch} (best_loss={best_val_loss:.4f})")

    # === Training loop ===
    print(f"\n🚀 Training from epoch {start_epoch}/{args.num_epochs}")
    for epoch in range(start_epoch, args.num_epochs):
        model.train()
        train_loss_total = 0.0
        progress = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.num_epochs}",
                        disable=not accelerator.is_local_main_process)

        for batch in progress:
            losses = model(batch, mode='train')
            accelerator.backward(losses['total'])
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            train_loss_total += losses['total'].item()
            progress.set_postfix({'loss': losses['total'].item()})

        avg_train_loss = train_loss_total / len(train_loader)
        scheduler.step()

        # === Validation ===
        if (epoch + 1) % args.eval_every == 0:
            model.eval()
            val_loss_total = 0.0
            with torch.no_grad():
                for batch in tqdm(dev_loader, desc="Validating",
                                  disable=not accelerator.is_local_main_process):
                    losses = model(batch, mode='train')
                    val_loss_total += losses['total'].item()
            avg_val_loss = val_loss_total / len(dev_loader)

            if accelerator.is_main_process:
                print(f"\nEpoch {epoch+1} | Train={avg_train_loss:.4f} | Val={avg_val_loss:.4f}")
                wandb.log({'train/loss_total': avg_train_loss,
                           'val/loss_total': avg_val_loss,
                           'epoch': epoch + 1})

                # === Save best ===
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save({
                        'epoch': epoch,
                        'model': accelerator.unwrap_model(model).state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': scheduler.state_dict(),
                        'best_val_loss': best_val_loss
                    }, ckpt_best)
                    print(f"🌟 New BEST model saved! (val_loss={best_val_loss:.4f})")

        # === Save periodic checkpoint ===
        if (epoch + 1) % args.save_every == 0 and accelerator.is_main_process:
            path = os.path.join(args.output_dir, f"checkpoint_epoch_{epoch+1}.pt")
            torch.save({
                'epoch': epoch,
                'model': accelerator.unwrap_model(model).state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_val_loss': best_val_loss
            }, path)
            print(f"💾 Saved checkpoint: {path}")

    if accelerator.is_main_process:
        print(f"\n🏁 Training complete! Best val_loss={best_val_loss:.4f}")
        wandb.finish()


if __name__ == "__main__":
    train()