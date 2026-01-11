#!/usr/bin/env python3
"""
Diagnostic Script: Check Latent Distribution from Stage 1 Autoencoder
Usage: python diagnose_latents.py --ae_ckpt path/to/best_model.pt --data_dir path/to/data
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(os.getcwd())

from dataset import SignLanguageDataset, collate_fn
from models.fml.autoencoder import UnifiedPoseAutoencoder


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ae_ckpt', type=str, required=True, help='Path to autoencoder checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to processed data directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=1024, help='Max samples to analyze')
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load dataset
    print("📂 Loading dataset...")
    dataset = SignLanguageDataset(args.data_dir, split='train', max_seq_len=400)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"   Loaded {len(dataset)} samples\n")
    
    # Load autoencoder
    print(f"🔧 Loading autoencoder from: {args.ae_ckpt}")
    ae = UnifiedPoseAutoencoder(latent_dim=256, hidden_dim=512).to(device)
    
    ckpt = torch.load(args.ae_ckpt, map_location=device)
    if 'model_state_dict' in ckpt:
        ae.load_state_dict(ckpt['model_state_dict'])
    else:
        ae.load_state_dict(ckpt)
    
    ae.eval()
    print("   ✅ Loaded\n")
    
    # Analyze latents
    print("🔍 Analyzing latent distribution...")
    latents = []
    poses_stats = {'min': [], 'max': [], 'has_nan': 0, 'has_inf': 0}
    latent_stats = {'min': [], 'max': [], 'has_nan': 0, 'has_inf': 0}
    
    seen = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches"):
            poses = batch['poses'].to(device)
            mask = batch['pose_mask'].to(device)
            
            # Check input poses
            if torch.isnan(poses).any():
                poses_stats['has_nan'] += 1
            if torch.isinf(poses).any():
                poses_stats['has_inf'] += 1
            
            poses_stats['min'].append(poses.min().item())
            poses_stats['max'].append(poses.max().item())
            
            # Encode
            z = ae.encode(poses, mask)
            
            # Check latents
            if torch.isnan(z).any():
                latent_stats['has_nan'] += 1
            if torch.isinf(z).any():
                latent_stats['has_inf'] += 1
            
            latent_stats['min'].append(z.min().item())
            latent_stats['max'].append(z.max().item())
            
            latents.append(z.detach().cpu().flatten())
            
            seen += poses.shape[0]
            if seen >= args.max_samples:
                break
    
    # Concatenate all latents
    latents = torch.cat(latents, dim=0)
    
    # Print report
    print("\n" + "="*70)
    print("📊 DIAGNOSTIC REPORT")
    print("="*70)
    
    print("\n🔹 INPUT POSES:")
    print(f"   Min value: {min(poses_stats['min']):.4f}")
    print(f"   Max value: {max(poses_stats['max']):.4f}")
    print(f"   Batches with NaN: {poses_stats['has_nan']}")
    print(f"   Batches with Inf: {poses_stats['has_inf']}")
    
    print("\n🔹 ENCODED LATENTS:")
    print(f"   Min value: {min(latent_stats['min']):.4f}")
    print(f"   Max value: {max(latent_stats['max']):.4f}")
    print(f"   Mean: {latents.mean():.4f}")
    print(f"   Std: {latents.std():.4f}")
    print(f"   Batches with NaN: {latent_stats['has_nan']}")
    print(f"   Batches with Inf: {latent_stats['has_inf']}")
    
    # Percentiles
    q25, q50, q75, q95, q99 = torch.quantile(
        latents, torch.tensor([0.25, 0.50, 0.75, 0.95, 0.99])
    )
    print(f"\n🔹 LATENT DISTRIBUTION (Percentiles):")
    print(f"   25th: {q25:.4f}")
    print(f"   50th (Median): {q50:.4f}")
    print(f"   75th: {q75:.4f}")
    print(f"   95th: {q95:.4f}")
    print(f"   99th: {q99:.4f}")
    
    # Assessment
    print("\n" + "="*70)
    print("🩺 ASSESSMENT")
    print("="*70)
    
    issues = []
    
    if poses_stats['has_nan'] > 0:
        issues.append("❌ INPUT DATA CONTAINS NaN - Critical issue!")
    if poses_stats['has_inf'] > 0:
        issues.append("❌ INPUT DATA CONTAINS Inf - Critical issue!")
    
    if latent_stats['has_nan'] > 0:
        issues.append("❌ ENCODER PRODUCES NaN - Model is broken!")
    if latent_stats['has_inf'] > 0:
        issues.append("❌ ENCODER PRODUCES Inf - Model is broken!")
    
    latent_range = max(latent_stats['max']) - min(latent_stats['min'])
    if latent_range > 50:
        issues.append(f"⚠️ Latent range too large ({latent_range:.1f}) - May need clamping")
    
    if abs(latents.mean()) > 5:
        issues.append(f"⚠️ Latent mean far from zero ({latents.mean():.2f}) - Poor centering")
    
    if latents.std() > 5 or latents.std() < 0.1:
        issues.append(f"⚠️ Latent std unusual ({latents.std():.2f}) - Expected: 0.5-2.0")
    
    if not issues:
        print("✅ All checks passed! Latent space looks healthy.")
    else:
        print("Found the following issues:\n")
        for issue in issues:
            print(f"   {issue}")
    
    print("\n" + "="*70)
    
    # Recommendations
    if issues:
        print("\n💡 RECOMMENDATIONS:")
        if latent_stats['has_nan'] > 0 or latent_stats['has_inf'] > 0:
            print("   1. Add latent clamping: latent.clamp(-10, 10) in encoder")
            print("   2. Re-train Stage 1 with input validation")
        elif latent_range > 50:
            print("   1. Add latent clamping: latent.clamp(-10, 10) in encoder")
        if latents.std() > 5:
            print("   2. Add L2 regularization to Stage 1 training")
        if poses_stats['has_nan'] > 0 or poses_stats['has_inf'] > 0:
            print("   3. Clean dataset - run data validation script")
        print("="*70)


if __name__ == '__main__':
    main()
