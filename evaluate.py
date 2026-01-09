#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Evaluation Script for Sign Language Generation Model
Run this after training to measure BLEU/ROUGE/DTW on test set
"""
import os
import sys
import argparse
import torch
import numpy as np
import json
from tqdm import tqdm
from torch.utils.data import DataLoader

sys.path.append(os.getcwd())

from dataset import SignLanguageDataset, collate_fn
from models.fml.autoencoder import UnifiedPoseAutoencoder
from models.fml.latent_flow_matcher import LatentFlowMatcher
from utils.evaluation_metrics import SignLanguageEvaluator


def load_models(args, device):
    """Load trained models"""
    print("\n📦 Loading models...")
    
    # Autoencoder
    ae = UnifiedPoseAutoencoder(
        latent_dim=args.latent_dim,
        hidden_dim=args.ae_hidden_dim
    ).to(device)
    
    ae_ckpt = torch.load(args.ae_ckpt, map_location=device)
    ae.load_state_dict(ae_ckpt.get("model_state_dict", ae_ckpt), strict=False)
    ae.eval()
    
    print(f"   ✅ Autoencoder loaded from {args.ae_ckpt}")
    
    # Flow Matcher
    flow = LatentFlowMatcher(
        latent_dim=args.latent_dim,
        hidden_dim=args.flow_hidden_dim
    ).to(device)
    
    flow_ckpt = torch.load(args.flow_ckpt, map_location=device)
    flow.load_state_dict(flow_ckpt['model_state_dict'], strict=False)
    flow.eval()
    
    latent_scale = float(flow_ckpt.get('latent_scale_factor', 1.0))
    
    print(f"   ✅ Flow Matcher loaded from {args.flow_ckpt}")
    print(f"   📏 Latent scale: {latent_scale:.6f}")
    
    return ae, flow, latent_scale


def decode_gloss_tokens(tokens, vocab=None):
    """Convert token IDs to gloss strings"""
    # Placeholder - implement based on your tokenizer
    if vocab is None:
        return [str(t) for t in tokens if t > 0]  # Filter padding
    return [vocab[t] for t in tokens if t > 0]


@torch.no_grad()
def evaluate_model(args):
    """Main evaluation function"""
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load models
    ae, flow, latent_scale = load_models(args, device)
    
    # Load test dataset
    print(f"\n📁 Loading test dataset from {args.data_dir}")
    test_dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        split=args.test_split,
        max_seq_len=args.max_seq_len
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    print(f"   📊 Test samples: {len(test_dataset)}")
    
    # Initialize evaluator
    evaluator = SignLanguageEvaluator()
    
    # Storage for results
    all_gt_poses = []
    all_pred_poses = []
    all_gt_glosses = []
    all_pred_glosses = []
    
    print("\n🔮 Running inference...")
    for batch in tqdm(test_loader, desc="Evaluating"):
        # Move to device
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(device)
        
        # Ground truth
        gt_poses = batch['poses']  # [B, T, 214]
        
        # Generate predictions
        with torch.cuda.amp.autocast(enabled=(device.type == "cuda")):
            # Text -> Latent (via Flow Matching)
            pred_latent = flow.sample(
                batch,
                steps=args.inference_steps,
                device=device,
                cfg_scale=args.cfg_scale,
                latent_scale=latent_scale,
                max_seq_len=args.max_seq_len
            )
            
            # Latent -> Pose (via Decoder)
            pred_poses, _ = ae.decode(pred_latent * latent_scale)
        
        # Convert to numpy
        gt_poses_np = gt_poses.cpu().numpy()
        pred_poses_np = pred_poses.cpu().numpy()
        
        # Store (per sample in batch)
        B = gt_poses.shape[0]
        for i in range(B):
            # Get actual sequence length
            if 'seq_lengths' in batch:
                actual_len = batch['seq_lengths'][i].item()
            else:
                actual_len = gt_poses.shape[1]
            
            all_gt_poses.append(gt_poses_np[i, :actual_len])
            all_pred_poses.append(pred_poses_np[i, :actual_len])
            
            # Store glosses (if available)
            if 'text_tokens' in batch:
                gt_gloss = decode_gloss_tokens(batch['text_tokens'][i].cpu().numpy())
                all_gt_glosses.append(gt_gloss)
                # Predicted gloss would need back-translation (skip for now)
    
    # Compute metrics
    print("\n📊 Computing metrics...")
    
    results = evaluator.evaluate_batch(
        gt_poses=all_gt_poses,
        pred_poses=all_pred_poses,
        gt_glosses=all_gt_glosses if all_gt_glosses else None,
        pred_glosses=None  # Would need back-translation model
    )
    
    # Print results
    evaluator.print_results(results, title=f"Test Set Evaluation (cfg_scale={args.cfg_scale})")
    
    # Save to file
    output_path = os.path.join(args.output_dir, f"eval_results_cfg{args.cfg_scale}.json")
    os.makedirs(args.output_dir, exist_ok=True)
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to: {output_path}")
    
    return results


def parse_args():
    p = argparse.ArgumentParser(description="Evaluate Sign Language Generation Model")
    
    # Data
    p.add_argument("--data_dir", type=str, required=True,
                   help="Path to processed dataset")
    p.add_argument("--test_split", type=str, default="test",
                   help="Test split name")
    p.add_argument("--max_seq_len", type=int, default=400)
    
    # Model checkpoints
    p.add_argument("--ae_ckpt", type=str, required=True,
                   help="Path to autoencoder checkpoint")
    p.add_argument("--flow_ckpt", type=str, required=True,
                   help="Path to flow matcher checkpoint")
    
    # Model config
    p.add_argument("--latent_dim", type=int, default=256)
    p.add_argument("--ae_hidden_dim", type=int, default=512)
    p.add_argument("--flow_hidden_dim", type=int, default=512)
    
    # Inference
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--inference_steps", type=int, default=50,
                   help="Number of flow matching steps")
    p.add_argument("--cfg_scale", type=float, default=1.5,
                   help="Classifier-free guidance scale")
    p.add_argument("--device", type=str, default="cuda")
    
    # Output
    p.add_argument("--output_dir", type=str, default="./eval_results",
                   help="Directory to save evaluation results")
    
    return p.parse_args()


def main():
    args = parse_args()
    
    print("=" * 70)
    print("🎯 SIGN LANGUAGE GENERATION MODEL EVALUATION")
    print("=" * 70)
    print(f"Data: {args.data_dir}")
    print(f"Test split: {args.test_split}")
    print(f"AE checkpoint: {args.ae_ckpt}")
    print(f"Flow checkpoint: {args.flow_ckpt}")
    print(f"CFG scale: {args.cfg_scale}")
    print(f"Inference steps: {args.inference_steps}")
    print("=" * 70)
    
    results = evaluate_model(args)
    
    print("\n✅ Evaluation completed!")
    
    # Optional: CFG scale sweep
    if input("\nRun CFG scale sweep? (y/n): ").lower() == 'y':
        print("\n🔄 Running CFG scale sweep...")
        for scale in [1.0, 1.5, 2.0, 2.5, 3.0]:
            print(f"\n--- Testing cfg_scale={scale} ---")
            args.cfg_scale = scale
            evaluate_model(args)


if __name__ == "__main__":
    main()
