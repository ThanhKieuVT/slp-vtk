#!/usr/bin/env python3
"""
Enhanced Diagnostic Script for Sign Language Production Models
Includes: FGD, comprehensive latent analysis, visualization
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import json

import sys
sys.path.append(os.getcwd())

from training.dataset import SignLanguageDataset, collate_fn
from models.fml.autoencoder import UnifiedPoseAutoencoder


def compute_fgd(real_features, gen_features):
    """
    Compute Fréchet Gesture Distance (FGD) similar to FID
    
    Args:
        real_features: [N, D] numpy array
        gen_features: [M, D] numpy array
    
    Returns:
        fgd: scalar
    """
    mu_real = np.mean(real_features, axis=0)
    mu_gen = np.mean(gen_features, axis=0)
    
    sigma_real = np.cov(real_features, rowvar=False)
    sigma_gen = np.cov(gen_features, rowvar=False)
    
    # Fréchet distance
    diff = mu_real - mu_gen
    covmean = sqrtm(sigma_real @ sigma_gen)
    
    # Handle numerical errors
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fgd = np.sum(diff ** 2) + np.trace(sigma_real + sigma_gen - 2 * covmean)
    return float(fgd)


def analyze_latent_distribution(latents, output_dir):
    """
    Comprehensive latent space analysis with visualizations
    
    Args:
        latents: [N, D] tensor
        output_dir: directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    latents_np = latents.cpu().numpy()
    
    results = {}
    
    # 1. Basic statistics
    results['mean'] = float(latents.mean().item())
    results['std'] = float(latents.std().item())
    results['min'] = float(latents.min().item())
    results['max'] = float(latents.max().item())
    
    # 2. Normality test (Shapiro-Wilk on sample)
    sample_size = min(5000, len(latents_np))
    sample_idx = np.random.choice(len(latents_np), sample_size, replace=False)
    _, p_value = stats.shapiro(latents_np[sample_idx])
    results['normality_p_value'] = float(p_value)
    results['is_normal'] = bool(p_value > 0.05)
    
    # 3. Per-dimension analysis
    latents_reshaped = latents_np.reshape(-1, latents_np.shape[-1])  # [N*T, D]
    dim_means = np.mean(latents_reshaped, axis=0)
    dim_stds = np.std(latents_reshaped, axis=0)
    dim_usage = np.sum(np.abs(latents_reshaped) > 0.1, axis=0) / len(latents_reshaped)
    
    results['unused_dims'] = int(np.sum(dim_usage < 0.01))
    results['dim_usage_mean'] = float(np.mean(dim_usage))
    
    # 4. Correlation between dimensions
    corr_matrix = np.corrcoef(latents_reshaped.T)
    off_diagonal_corr = corr_matrix[np.triu_indices(len(corr_matrix), k=1)]
    results['mean_dim_correlation'] = float(np.mean(np.abs(off_diagonal_corr)))
    results['max_dim_correlation'] = float(np.max(np.abs(off_diagonal_corr)))
    
    # 5. Visualizations
    
    # Plot 1: Distribution histogram
    plt.figure(figsize=(10, 6))
    plt.hist(latents_np.flatten(), bins=100, alpha=0.7, edgecolor='black')
    plt.axvline(results['mean'], color='red', linestyle='--', label=f"Mean: {results['mean']:.3f}")
    plt.xlabel('Latent Value')
    plt.ylabel('Frequency')
    plt.title('Latent Space Distribution')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'latent_distribution.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 2: Per-dimension statistics
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].plot(dim_means)
    axes[0].set_title('Per-Dimension Mean')
    axes[0].set_xlabel('Dimension')
    axes[0].set_ylabel('Mean')
    axes[0].grid(alpha=0.3)
    
    axes[1].plot(dim_stds)
    axes[1].set_title('Per-Dimension Std')
    axes[1].set_xlabel('Dimension')
    axes[1].set_ylabel('Std')
    axes[1].grid(alpha=0.3)
    
    axes[2].plot(dim_usage)
    axes[2].axhline(0.01, color='red', linestyle='--', label='Usage threshold')
    axes[2].set_title('Dimension Usage')
    axes[2].set_xlabel('Dimension')
    axes[2].set_ylabel('Usage Rate')
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'per_dimension_stats.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot 3: Correlation heatmap (sample of dimensions for readability)
    if latents_reshaped.shape[1] > 64:
        sample_dims = np.random.choice(latents_reshaped.shape[1], 64, replace=False)
        sample_dims = np.sort(sample_dims)
        corr_sample = np.corrcoef(latents_reshaped[:, sample_dims].T)
        title_suffix = " (64 sampled dims)"
    else:
        corr_sample = corr_matrix
        title_suffix = ""
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_sample, cmap='coolwarm', center=0, vmin=-1, vmax=1,
                cbar_kws={'label': 'Correlation'})
    plt.title(f'Latent Dimension Correlation{title_suffix}')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
    return results


def main():
    parser = argparse.ArgumentParser(description='Enhanced diagnostics for autoencoder')
    parser.add_argument('--ae_ckpt', type=str, required=True, help='Path to autoencoder checkpoint')
    parser.add_argument('--data_dir', type=str, required=True, help='Path to processed data directory')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--max_samples', type=int, default=1024, help='Max samples to analyze')
    parser.add_argument('--output_dir', type=str, default='diagnostics', help='Output directory')
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load dataset
    print("📂 Loading dataset...")
    dataset = SignLanguageDataset(
        args.data_dir, 
        split='train', 
        max_seq_len=400,
        normalize=True
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn)
    print(f"   Loaded {len(dataset)} samples\n")
    
    # Load autoencoder
    print(f"🔧 Loading autoencoder from: {args.ae_ckpt}")
    ae = UnifiedPoseAutoencoder(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    
    ckpt = torch.load(args.ae_ckpt, map_location=device)
    if 'model_state_dict' in ckpt:
        ae.load_state_dict(ckpt['model_state_dict'])
        print(f"   Checkpoint: Epoch {ckpt.get('epoch', 'unknown')}, "
              f"Best Val Loss: {ckpt.get('best_val_loss', 'unknown'):.4f}")
    elif 'model' in ckpt:
        ae.load_state_dict(ckpt['model'])
        print(f"   Checkpoint: Epoch {ckpt.get('epoch', 'unknown')}")
    else:
        ae.load_state_dict(ckpt)
    
    ae.eval()
    print("   ✅ Loaded\n")
    
    # Collect data
    print("🔍 Analyzing autoencoder quality...")
    latents = []
    real_poses = []
    recon_poses = []
    
    poses_stats = {'min': [], 'max': [], 'has_nan': 0, 'has_inf': 0}
    latent_stats = {'min': [], 'max': [], 'has_nan': 0, 'has_inf': 0}
    recon_errors = []
    
    seen = 0
    with torch.no_grad():
        for batch in tqdm(loader, desc="Processing batches"):
            poses = batch['poses'].to(device)
            mask = batch['pose_mask'].to(device)
            
            # Check input
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
            
            # Decode
            recon = ae.decode(z, mask)
            
            # Compute reconstruction error
            valid_recon = recon[mask]
            valid_poses = poses[mask]
            batch_error = torch.mean((valid_recon - valid_poses) ** 2).item()
            recon_errors.append(batch_error)
            
            # Store for FGD computation
            latents.append(z.cpu())
            real_poses.append(valid_poses.cpu().numpy())
            recon_poses.append(valid_recon.cpu().numpy())
            
            seen += poses.shape[0]
            if seen >= args.max_samples:
                break
    
    # Concatenate results
    latents = torch.cat(latents, dim=0)
    real_poses_concat = np.concatenate(real_poses, axis=0)
    recon_poses_concat = np.concatenate(recon_poses, axis=0)
    
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
    latents_flat = latents.flatten()
    print(f"   Min value: {min(latent_stats['min']):.4f}")
    print(f"   Max value: {max(latent_stats['max']):.4f}")
    print(f"   Mean: {latents_flat.mean():.4f}")
    print(f"   Std: {latents_flat.std():.4f}")
    print(f"   Batches with NaN: {latent_stats['has_nan']}")
    print(f"   Batches with Inf: {latent_stats['has_inf']}")
    
    # Percentiles
    q25, q50, q75, q95, q99 = torch.quantile(
        latents_flat, torch.tensor([0.25, 0.50, 0.75, 0.95, 0.99])
    )
    print(f"\n🔹 LATENT DISTRIBUTION (Percentiles):")
    print(f"   25th: {q25:.4f}")
    print(f"   50th (Median): {q50:.4f}")
    print(f"   75th: {q75:.4f}")
    print(f"   95th: {q95:.4f}")
    print(f"   99th: {q99:.4f}")
    
    print(f"\n🔹 RECONSTRUCTION QUALITY:")
    mean_recon_error = np.mean(recon_errors)
    print(f"   Mean MSE: {mean_recon_error:.6f}")
    print(f"   Std MSE: {np.std(recon_errors):.6f}")
    
    # FGD
    print(f"\n🔹 FRECHET GESTURE DISTANCE (FGD):")
    try:
        from scipy.linalg import sqrtm
        fgd = compute_fgd(real_poses_concat, recon_poses_concat)
        print(f"   FGD: {fgd:.4f}")
    except Exception as e:
        print(f"   ⚠️ Could not compute FGD: {e}")
        fgd = None
    
    # Latent space analysis
    print(f"\n🔬 LATENT SPACE ANALYSIS:")
    latent_analysis = analyze_latent_distribution(latents, args.output_dir)
    print(f"   Unused dimensions (< 1% usage): {latent_analysis['unused_dims']}")
    print(f"   Mean dimension usage: {latent_analysis['dim_usage_mean']:.2%}")
    print(f"   Mean inter-dimension correlation: {latent_analysis['mean_dim_correlation']:.4f}")
    print(f"   Is normal distribution: {latent_analysis['is_normal']} "
          f"(p={latent_analysis['normality_p_value']:.4f})")
    
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
    
    if abs(latents_flat.mean()) > 5:
        issues.append(f"⚠️ Latent mean far from zero ({latents_flat.mean():.2f}) - Poor centering")
    
    if latents_flat.std() > 5 or latents_flat.std() < 0.1:
        issues.append(f"⚠️ Latent std unusual ({latents_flat.std():.2f}) - Expected: 0.5-2.0")
    
    if mean_recon_error > 0.1:
        issues.append(f"⚠️ High reconstruction error ({mean_recon_error:.4f}) - Model underfitting")
    
    if latent_analysis['unused_dims'] > latent_analysis.get('latent_dim', 256) * 0.2:
        issues.append(f"⚠️ Many unused latent dimensions ({latent_analysis['unused_dims']}) - Inefficient")
    
    if not issues:
        print("✅ All checks passed! Autoencoder looks healthy.")
    else:
        print("Found the following issues:\n")
        for issue in issues:
            print(f"   {issue}")
    
    print("\n" + "="*70)
    
    # Save results
    results = {
        'timestamp': str(torch.cuda.get_device_properties(0).name if torch.cuda.is_available() else 'CPU'),
        'checkpoint': args.ae_ckpt,
        'samples_analyzed': seen,
        'input_stats': poses_stats,
        'latent_stats': {k: v for k, v in latent_stats.items() if k not in ['min', 'max']},
        'latent_range': latent_range,
        'latent_mean': float(latents_flat.mean()),
        'latent_std': float(latents_flat.std()),
        'reconstruction_mse': mean_recon_error,
        'fgd': fgd,
        'latent_analysis': latent_analysis,
        'issues': issues
    }
    
    output_json = os.path.join(args.output_dir, 'diagnostic_results.json')
    with open(output_json, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n💾 Results saved to:")
    print(f"   JSON: {output_json}")
    print(f"   Plots: {args.output_dir}/")
    print("="*70)


if __name__ == '__main__':
    main()
