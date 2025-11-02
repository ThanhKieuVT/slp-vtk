"""
Train NMM flow models with resume + best checkpoint saving (Robust + Colab-friendly)
- auto-find resume checkpoint (args.resume_from > best_model.pt > last_checkpoint.pt > latest periodic)
- try strict load, fallback to strict=False and report missing/unexpected keys
- save best_model.pt, last_checkpoint.pt and periodic checkpoint_epoch_*.pt
- defaults num_workers=0 for Colab
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

sys.path.append('..')

from models.nmm_generator import NMMFlowGenerator
from models.hierarchical_flow import HierarchicalFlowMatcher
from data.phoenix_dataset import PhoenixFlowDataset, collate_fn


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='../data/RWTH/processed_data/final')
    parser.add_argument('--output_dir', type=str, default='./checkpoints/nmm_flow')
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--num_epochs', type=int, default=40)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--num_workers', type=int, default=0)  # safe for Colab
    parser.add_argument('--save_every', type=int, default=5)
    parser.add_argument('--eval_every', type=int, default=1)
    parser.add_argument('--resume_from', type=str, default=None)
    parser.add_argument('--share_text_encoder', type=str, default=None)
    parser.add_argument('--freeze_text_encoder', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='slp-hierarchical-flow')
    parser.add_argument('--wandb_run_name', type=str, default='nmm_flow')
    return parser.parse_args()


def find_latest_checkpoint(output_dir):
    """Return latest periodic checkpoint or last_checkpoint if exists, else None"""
    last_ckpt = os.path.join(output_dir, "last_checkpoint.pt")
    if os.path.exists(last_ckpt):
        return last_ckpt
    ckpts = sorted(glob.glob(os.path.join(output_dir, 'checkpoint_epoch_*.pt')))
    return ckpts[-1] if ckpts else None


def try_load_state(unwrap_model, state_dict):
    """
    Try to load state_dict strictly; on failure, fallback to strict=False and
    return tuple(success_bool, missing_keys, unexpected_keys)
    """
    try:
        unwrap_model.load_state_dict(state_dict, strict=True)
        return True, [], []
    except RuntimeError as e:
        # Try non-strict load and report keys
        try:
            res = unwrap_model.load_state_dict(state_dict, strict=False)
            # PyTorch returns a NamedTuple (missing_keys, unexpected_keys) from load_state_dict
            missing = getattr(res, 'missing_keys', None)
            unexpected = getattr(res, 'unexpected_keys', None)
            return True, missing or [], unexpected or []
        except Exception as e2:
            print("❌ Failed to load state even with strict=False:", e2)
            return False, None, None


def train():
    args = parse_args()
    accelerator = Accelerator(mixed_precision='fp16')

    # Init W&B only on main process
    if accelerator.is_main_process:
        try:
            wandb.init(project=args.wandb_project, name=args.wandb_run_name, config=vars(args))
        except Exception as e:
            print("⚠️ WandB init failed:", e)

    # Dataset
    print("Loading datasets...")
    train_dataset = PhoenixFlowDataset(split='train', data_root=args.data_root, augment=True)
    dev_dataset = PhoenixFlowDataset(split='dev', data_root=args.data_root)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=collate_fn, num_workers=args.num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False,
                            collate_fn=collate_fn, num_workers=args.num_workers)

    # Model
    print("Building model...")
    shared_text_encoder = None
    if args.share_text_encoder:
        if os.path.exists(args.share_text_encoder):
            print(f"🔗 Sharing text encoder from {args.share_text_encoder}")
            manual_ckpt = torch.load(args.share_text_encoder, map_location='cpu')
            manual_model = HierarchicalFlowMatcher()
            # load manual checkpoint into manual_model safely (allow strict=False fallback)
            success, missing, unexpected = try_load_state(manual_model, manual_ckpt.get('model', manual_ckpt))
            if not success:
                raise RuntimeError("Failed to load provided manual checkpoint for shared text encoder.")
            shared_text_encoder = manual_model.text_encoder
            print("✅ Shared text encoder loaded")
        else:
            print(f"⚠️ share_text_encoder path not found: {args.share_text_encoder}. Ignoring share_text_encoder.")

    model = NMMFlowGenerator(
        freeze_text_encoder=args.freeze_text_encoder,
        share_text_encoder=shared_text_encoder
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.num_epochs, eta_min=1e-6)

    # Prepare with accelerator (wraps model, optimizer, loaders, scheduler)
    model, optimizer, train_loader, dev_loader, scheduler = accelerator.prepare(
        model, optimizer, train_loader, dev_loader, scheduler
    )

    os.makedirs(args.output_dir, exist_ok=True)
    ckpt_best = os.path.join(args.output_dir, "best_model.pt")
    ckpt_last = os.path.join(args.output_dir, "last_checkpoint.pt")

    # Resume logic: prefer explicit args.resume_from > best_model.pt > last_checkpoint.pt > latest periodic
    start_epoch, best_val_loss = 0, float('inf')
    candidate = None
    if args.resume_from:
        candidate = args.resume_from
    elif os.path.exists(ckpt_best):
        candidate = ckpt_best
    else:
        candidate = find_latest_checkpoint(args.output_dir)

    if candidate and os.path.exists(candidate):
        print(f"🔁 Attempting to resume from: {candidate}")
        ckpt = torch.load(candidate, map_location='cpu')
        # load model weights into unwrapped model (safe attempt strict then fallback)
        unwrap_model = accelerator.unwrap_model(model)
        # state dict may be nested under 'model' or be the checkpoint itself
        state_dict = ckpt.get('model', ckpt)
        try:
            # first try strict load
            unwrap_model.load_state_dict(state_dict, strict=True)
            print("✅ Model state loaded with strict=True")
            missing_keys, unexpected_keys = [], []
        except RuntimeError as e:
            print("⚠️ strict=True failed:", e)
            print("➡️ Trying strict=False and reporting missing/unexpected keys...")
            try:
                res = unwrap_model.load_state_dict(state_dict, strict=False)
                missing_keys = getattr(res, 'missing_keys', []) or []
                unexpected_keys = getattr(res, 'unexpected_keys', []) or []
                print("  Missing keys (will be randomly initialized):", missing_keys)
                print("  Unexpected keys (ignored):", unexpected_keys)
            except Exception as e2:
                print("❌ Loading failed even with strict=False:", e2)
                raise RuntimeError("Unable to load checkpoint into model (incompatible).")
        # load optimizer and scheduler if available (best-effort)
        if 'optimizer' in ckpt:
            try:
                optimizer.load_state_dict(ckpt['optimizer'])
            except Exception as e:
                print("⚠️ Could not load optimizer state cleanly:", e)
        if 'scheduler' in ckpt:
            try:
                scheduler.load_state_dict(ckpt['scheduler'])
            except Exception as e:
                print("⚠️ Could not load scheduler state cleanly:", e)
        start_epoch = ckpt.get('epoch', 0) + 1
        best_val_loss = ckpt.get('best_val_loss', float('inf'))
        print(f"✅ Resume successful (starting epoch {start_epoch}, best_val_loss={best_val_loss:.4f})")

    # Training loop
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

        # Validation
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
                try:
                    wandb.log({'train/loss_total': avg_train_loss,
                               'val/loss_total': avg_val_loss,
                               'epoch': epoch + 1})
                except Exception:
                    pass

                # Save best
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

        # Always save last checkpoint (safe for interruptions)
        if accelerator.is_main_process:
            try:
                torch.save({
                    'epoch': epoch,
                    'model': accelerator.unwrap_model(model).state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'best_val_loss': best_val_loss
                }, ckpt_last)
            except Exception as e:
                print("⚠️ Failed to save last_checkpoint.pt:", e)

            # periodic checkpoint
            if (epoch + 1) % args.save_every == 0:
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
        try:
            wandb.finish()
        except Exception:
            pass


if __name__ == "__main__":
    train()
