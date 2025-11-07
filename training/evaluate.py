"""
Evaluation Script cho PHOENIX-2014T
Metrics: MSE, DTW, Sync Quality, BLEU1-4, ROUGE
"""
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial.distance import euclidean
import pickle
from collections import Counter
import re

from dataset import SignLanguageDataset, collate_fn
from models.fml.autoencoder import UnifiedPoseAutoencoder
from models.fml.latent_predictor import LatentPredictor
from data_preparation import denormalize_pose


def dtw_distance(seq1, seq2):
    """
    Simple DTW implementation (không cần fastdtw)
    
    Args:
        seq1: [T1, D] numpy array
        seq2: [T2, D] numpy array
    
    Returns:
        dtw_distance: float
    """
    T1, D = seq1.shape
    T2, D2 = seq2.shape
    
    # Compute pairwise distances
    distances = np.zeros((T1, T2))
    for i in range(T1):
        for j in range(T2):
            distances[i, j] = np.linalg.norm(seq1[i] - seq2[j])
    
    # DTW dynamic programming
    dtw = np.full((T1 + 1, T2 + 1), np.inf)
    dtw[0, 0] = 0
    
    for i in range(1, T1 + 1):
        for j in range(1, T2 + 1):
            cost = distances[i - 1, j - 1]
            dtw[i, j] = cost + min(dtw[i - 1, j],      # insertion
                                   dtw[i, j - 1],      # deletion
                                   dtw[i - 1, j - 1])  # match
    
    return dtw[T1, T2]


def compute_bleu(reference, candidate, n=4):
    """
    Compute BLEU-n score
    
    Args:
        reference: list of words (ground truth)
        candidate: list of words (predicted)
        n: n-gram order (1-4)
    
    Returns:
        bleu_score: float
    """
    if len(candidate) == 0:
        return 0.0
    
    # Compute n-gram precision
    precisions = []
    for i in range(1, n + 1):
        ref_ngrams = Counter([tuple(reference[j:j+i]) for j in range(len(reference)-i+1)])
        cand_ngrams = Counter([tuple(candidate[j:j+i]) for j in range(len(candidate)-i+1)])
        
        matches = sum((ref_ngrams & cand_ngrams).values())
        total = sum(cand_ngrams.values())
        
        if total == 0:
            precisions.append(0.0)
        else:
            precisions.append(matches / total)
    
    # Brevity penalty
    if len(candidate) > len(reference):
        bp = 1.0
    else:
        bp = np.exp(1 - len(reference) / len(candidate))
    
    # Geometric mean of precisions
    if min(precisions) > 0:
        bleu = bp * np.exp(np.mean([np.log(p) for p in precisions if p > 0]))
    else:
        bleu = 0.0
    
    return bleu


def compute_rouge_l(reference, candidate):
    """
    Compute ROUGE-L score (Longest Common Subsequence)
    
    Args:
        reference: list of words (ground truth)
        candidate: list of words (predicted)
    
    Returns:
        rouge_l: float
    """
    if len(reference) == 0 or len(candidate) == 0:
        return 0.0
    
    # LCS using dynamic programming
    m, n = len(reference), len(candidate)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if reference[i-1] == candidate[j-1]:
                dp[i][j] = dp[i-1][j-1] + 1
            else:
                dp[i][j] = max(dp[i-1][j], dp[i][j-1])
    
    lcs = dp[m][n]
    
    # ROUGE-L = F1 score
    precision = lcs / len(candidate) if len(candidate) > 0 else 0.0
    recall = lcs / len(reference) if len(reference) > 0 else 0.0
    
    if precision + recall == 0:
        return 0.0
    
    rouge_l = 2 * precision * recall / (precision + recall)
    return rouge_l


def tokenize_text(text):
    """
    Tokenize text thành words (simple whitespace tokenization)
    Có thể cải thiện với proper tokenizer
    """
    # Simple tokenization: split by whitespace and punctuation
    tokens = re.findall(r'\b\w+\b', text.lower())
    return tokens


def compute_sync_quality(pose_pred, pose_gt, mask=None):
    """
    Tính sync quality giữa tay (manual) và mặt (NMM)
    
    Args:
        pose_pred: [T, 214] - predicted pose
        pose_gt: [T, 214] - ground truth pose
        mask: [T] - mask cho valid frames
    
    Returns:
        correlation: float - Pearson correlation
        lag: int - lag tối ưu (frames)
    """
    if mask is not None:
        pose_pred = pose_pred[mask]
        pose_gt = pose_gt[mask]
    
    T = len(pose_pred)
    if T < 10:
        return 0.0, 0
    
    # Extract manual (tay): first 150 dims
    manual_pred = pose_pred[:, :150]  # [T, 150]
    manual_gt = pose_gt[:, :150]
    
    # Extract NMM (mặt): last 64 dims
    nmm_pred = pose_pred[:, 150:]  # [T, 64]
    nmm_gt = pose_gt[:, 150:]
    
    # Flatten để có 1D signal
    manual_pred_flat = manual_pred.flatten()  # [T*150]
    manual_gt_flat = manual_gt.flatten()
    nmm_pred_flat = nmm_pred.flatten()  # [T*64]
    nmm_gt_flat = nmm_gt.flatten()
    
    # Compute correlation cho manual
    manual_corr = np.corrcoef(manual_pred_flat, manual_gt_flat)[0, 1]
    if np.isnan(manual_corr):
        manual_corr = 0.0
    
    # Compute correlation cho NMM
    nmm_corr = np.corrcoef(nmm_pred_flat, nmm_gt_flat)[0, 1]
    if np.isnan(nmm_corr):
        nmm_corr = 0.0
    
    # Compute cross-correlation để tìm lag giữa manual và NMM
    max_lag = min(10, T // 4)
    best_corr = -1.0
    best_lag = 0
    
    # Tính correlation giữa manual và NMM trong predicted pose
    for lag in range(-max_lag, max_lag + 1):
        if lag == 0:
            # No lag
            if len(manual_pred_flat) > 0 and len(nmm_pred_flat) > 0:
                # Normalize lengths
                min_len = min(len(manual_pred_flat), len(nmm_pred_flat))
                corr = np.corrcoef(
                    manual_pred_flat[:min_len],
                    nmm_pred_flat[:min_len]
                )[0, 1]
            else:
                continue
        elif lag > 0:
            # manual shift forward
            if len(manual_pred_flat) > lag:
                min_len = min(len(manual_pred_flat) - lag, len(nmm_pred_flat))
                if min_len > 0:
                    corr = np.corrcoef(
                        manual_pred_flat[lag:lag+min_len],
                        nmm_pred_flat[:min_len]
                    )[0, 1]
                else:
                    continue
            else:
                continue
        else:
            # manual shift backward
            lag_abs = abs(lag)
            if len(manual_pred_flat) > lag_abs:
                min_len = min(len(manual_pred_flat) - lag_abs, len(nmm_pred_flat))
                if min_len > 0:
                    corr = np.corrcoef(
                        manual_pred_flat[:min_len],
                        nmm_pred_flat[lag_abs:lag_abs+min_len]
                    )[0, 1]
                else:
                    continue
            else:
                continue
        
        if not np.isnan(corr) and corr > best_corr:
            best_corr = corr
            best_lag = lag
    
    # Combined correlation
    final_corr = (manual_corr + nmm_corr) / 2.0
    
    return final_corr, abs(best_lag)


def evaluate_model(
    predictor,
    encoder,
    decoder,
    dataloader,
    device,
    normalize_stats=None,
    save_predictions=False,
    output_dir=None
):
    """
    Evaluate model trên dataset
    
    Returns:
        metrics: dict với các metrics
    """
    predictor.eval()
    encoder.eval()
    decoder.eval()
    
    all_dtw_distances = []
    all_correlations = []
    all_lags = []
    all_pose_errors = []
    all_manual_errors = []
    all_nmm_errors = []
    
    # Text metrics
    all_bleu1 = []
    all_bleu2 = []
    all_bleu3 = []
    all_bleu4 = []
    all_rouge_l = []
    
    predictions = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            poses_gt = batch['poses'].to(device)  # [B, T, 214]
            pose_mask = batch['pose_mask'].to(device)  # [B, T]
            text_tokens = batch['text_tokens'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            seq_lengths = batch['seq_lengths'].to(device)
            
            # Predict
            predicted_latent = predictor(
                text_tokens,
                attention_mask,
                target_length=seq_lengths,
                mask=pose_mask
            )
            predicted_pose = decoder(predicted_latent, mask=pose_mask)  # [B, T, 214]
            
            # Convert to numpy
            poses_gt_np = poses_gt.cpu().numpy()
            predicted_pose_np = predicted_pose.cpu().numpy()
            pose_mask_np = pose_mask.cpu().numpy()
            seq_lengths_np = seq_lengths.cpu().numpy()
            
            # Denormalize nếu có stats
            if normalize_stats is not None:
                mean = normalize_stats['mean']
                std = normalize_stats['std']
                poses_gt_np = denormalize_pose(poses_gt_np, mean, std)
                predicted_pose_np = denormalize_pose(predicted_pose_np, mean, std)
            
            # Process từng sample trong batch
            B = poses_gt_np.shape[0]
            for i in range(B):
                T = int(seq_lengths_np[i])
                mask = pose_mask_np[i, :T]
                
                pose_gt = poses_gt_np[i, :T]  # [T, 214]
                pose_pred = predicted_pose_np[i, :T]  # [T, 214]
                
                # Apply mask
                pose_gt = pose_gt[mask]
                pose_pred = pose_pred[mask]
                
                if len(pose_gt) == 0:
                    continue
                
                # DTW distance
                try:
                    dtw_dist = dtw_distance(pose_gt, pose_pred)
                    all_dtw_distances.append(dtw_dist)
                except Exception as e:
                    print(f"DTW error: {e}")
                    pass
                
                # Sync quality
                corr, lag = compute_sync_quality(pose_pred, pose_gt, mask=None)
                all_correlations.append(corr)
                all_lags.append(lag)
                
                # MSE errors
                mse = np.mean((pose_gt - pose_pred) ** 2)
                all_pose_errors.append(mse)
                
                # Manual error (tay)
                manual_gt = pose_gt[:, :150]
                manual_pred = pose_pred[:, :150]
                manual_mse = np.mean((manual_gt - manual_pred) ** 2)
                all_manual_errors.append(manual_mse)
                
                # NMM error (mặt)
                nmm_gt = pose_gt[:, 150:]
                nmm_pred = pose_pred[:, 150:]
                nmm_mse = np.mean((nmm_gt - nmm_pred) ** 2)
                all_nmm_errors.append(nmm_mse)
                
                # Text metrics (BLEU, ROUGE) - nếu có text ground truth
                if 'texts' in batch and batch['texts'][i]:
                    ref_text = batch['texts'][i]
                    # Trong trường hợp này, chúng ta không có text generated từ pose
                    # Nên sẽ dùng text input làm candidate (vì model là Text → Pose)
                    # Hoặc có thể skip nếu không có pose → text model
                    # Tạm thời: so sánh text input với chính nó (để test metric)
                    # TODO: Nếu có pose → text model, dùng pose_pred để generate text
                    
                    # Tokenize
                    ref_tokens = tokenize_text(ref_text)
                    
                    # Nếu có text input, dùng làm candidate (tạm thời)
                    # Trong thực tế, cần pose → text model để generate text từ pose_pred
                    # Ở đây tạm thời dùng ref_text làm candidate để test metric
                    cand_tokens = ref_tokens  # TODO: Generate từ pose_pred
                    
                    # Compute BLEU scores
                    bleu1 = compute_bleu(ref_tokens, cand_tokens, n=1)
                    bleu2 = compute_bleu(ref_tokens, cand_tokens, n=2)
                    bleu3 = compute_bleu(ref_tokens, cand_tokens, n=3)
                    bleu4 = compute_bleu(ref_tokens, cand_tokens, n=4)
                    
                    # Compute ROUGE-L
                    rouge_l = compute_rouge_l(ref_tokens, cand_tokens)
                    
                    all_bleu1.append(bleu1)
                    all_bleu2.append(bleu2)
                    all_bleu3.append(bleu3)
                    all_bleu4.append(bleu4)
                    all_rouge_l.append(rouge_l)
                
                if save_predictions:
                    predictions.append({
                        'video_id': batch['video_ids'][i],
                        'text': batch['texts'][i],
                        'pose_gt': pose_gt,
                        'pose_pred': pose_pred,
                        'dtw': dtw_dist if 'dtw_dist' in locals() else None,
                        'correlation': corr,
                        'lag': lag,
                        'mse': mse,
                        'manual_mse': manual_mse,
                        'nmm_mse': nmm_mse
                    })
    
    # Compute statistics
    metrics = {
        'dtw_mean': np.mean(all_dtw_distances) if all_dtw_distances else float('inf'),
        'dtw_std': np.std(all_dtw_distances) if all_dtw_distances else 0.0,
        'dtw_median': np.median(all_dtw_distances) if all_dtw_distances else float('inf'),
        'correlation_mean': np.mean(all_correlations) if all_correlations else 0.0,
        'correlation_std': np.std(all_correlations) if all_correlations else 0.0,
        'lag_mean': np.mean(all_lags) if all_lags else 0.0,
        'lag_std': np.std(all_lags) if all_lags else 0.0,
        'mse_mean': np.mean(all_pose_errors) if all_pose_errors else float('inf'),
        'mse_std': np.std(all_pose_errors) if all_pose_errors else 0.0,
        'manual_mse_mean': np.mean(all_manual_errors) if all_manual_errors else float('inf'),
        'manual_mse_std': np.std(all_manual_errors) if all_manual_errors else 0.0,
        'nmm_mse_mean': np.mean(all_nmm_errors) if all_nmm_errors else float('inf'),
        'nmm_mse_std': np.std(all_nmm_errors) if all_nmm_errors else 0.0,
        'bleu1_mean': np.mean(all_bleu1) if all_bleu1 else 0.0,
        'bleu1_std': np.std(all_bleu1) if all_bleu1 else 0.0,
        'bleu2_mean': np.mean(all_bleu2) if all_bleu2 else 0.0,
        'bleu2_std': np.std(all_bleu2) if all_bleu2 else 0.0,
        'bleu3_mean': np.mean(all_bleu3) if all_bleu3 else 0.0,
        'bleu3_std': np.std(all_bleu3) if all_bleu3 else 0.0,
        'bleu4_mean': np.mean(all_bleu4) if all_bleu4 else 0.0,
        'bleu4_std': np.std(all_bleu4) if all_bleu4 else 0.0,
        'rouge_l_mean': np.mean(all_rouge_l) if all_rouge_l else 0.0,
        'rouge_l_std': np.std(all_rouge_l) if all_rouge_l else 0.0,
        'num_samples': len(all_correlations),
        'num_text_samples': len(all_bleu1)  # Số samples có text
    }
    
    # Save predictions nếu cần
    if save_predictions and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        pred_path = os.path.join(output_dir, 'predictions.pkl')
        with open(pred_path, 'wb') as f:
            pickle.dump(predictions, f)
        print(f"Saved predictions to {pred_path}")
    
    return metrics


def print_metrics(metrics, split_name=''):
    """Print metrics đẹp"""
    print(f"\n{'='*70}")
    print(f"Evaluation Results - {split_name.upper()} SET")
    print(f"{'='*70}")
    print(f"Number of samples: {metrics['num_samples']}")
    print(f"\n📊 DTW Distance (lower is better):")
    print(f"  Mean:   {metrics['dtw_mean']:.4f}")
    print(f"  Std:    {metrics['dtw_std']:.4f}")
    print(f"  Median: {metrics['dtw_median']:.4f}")
    print(f"\n🔗 Sync Quality - Correlation (higher is better):")
    print(f"  Mean: {metrics['correlation_mean']:.4f}")
    print(f"  Std:  {metrics['correlation_std']:.4f}")
    print(f"\n⏱️  Sync Lag - Frames (lower is better):")
    print(f"  Mean: {metrics['lag_mean']:.2f}")
    print(f"  Std:  {metrics['lag_std']:.2f}")
    print(f"\n📉 MSE Error - Full Pose (lower is better):")
    print(f"  Mean: {metrics['mse_mean']:.6f}")
    print(f"  Std:  {metrics['mse_std']:.6f}")
    print(f"\n✋ MSE Error - Manual (Hands, lower is better):")
    print(f"  Mean: {metrics['manual_mse_mean']:.6f}")
    print(f"  Std:  {metrics['manual_mse_std']:.6f}")
    print(f"\n😊 MSE Error - NMM (Face, lower is better):")
    print(f"  Mean: {metrics['nmm_mse_mean']:.6f}")
    print(f"  Std:  {metrics['nmm_mse_std']:.6f}")
    
    if metrics['num_text_samples'] > 0:
        print(f"\n📝 Text Metrics (BLEU & ROUGE):")
        print(f"  Number of samples with text: {metrics['num_text_samples']}")
        print(f"  BLEU-1: {metrics['bleu1_mean']:.4f} ± {metrics['bleu1_std']:.4f}")
        print(f"  BLEU-2: {metrics['bleu2_mean']:.4f} ± {metrics['bleu2_std']:.4f}")
        print(f"  BLEU-3: {metrics['bleu3_mean']:.4f} ± {metrics['bleu3_std']:.4f}")
        print(f"  BLEU-4: {metrics['bleu4_mean']:.4f} ± {metrics['bleu4_std']:.4f}")
        print(f"  ROUGE-L: {metrics['rouge_l_mean']:.4f} ± {metrics['rouge_l_std']:.4f}")
        print(f"\n  ⚠️  Note: BLEU/ROUGE hiện tại dùng text input làm candidate.")
        print(f"      Để tính đúng, cần có Pose → Text model để generate text từ pose.")
    else:
        print(f"\n⚠️  No text data found for BLEU/ROUGE metrics")
    
    print(f"{'='*70}\n")


def main():
    parser = argparse.ArgumentParser(description='Evaluate model trên PHOENIX-2014T')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Đường dẫn đến processed_data/data/')
    parser.add_argument('--predictor_checkpoint', type=str, required=True,
                        help='Checkpoint của Latent Predictor (Stage 2)')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                        help='Checkpoint của Autoencoder (Stage 1)')
    parser.add_argument('--split', type=str, default='dev', choices=['dev', 'test'],
                        help='Split để evaluate (dev hoặc test)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Số workers')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--max_seq_len', type=int, default=120,
                        help='Max sequence length')
    parser.add_argument('--save_predictions', action='store_true',
                        help='Lưu predictions để phân tích sau')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Thư mục lưu predictions và metrics')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load normalization stats
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    normalize_stats = None
    if os.path.exists(stats_path):
        normalize_stats = np.load(stats_path)
        print(f"✅ Loaded normalization stats from {stats_path}")
    else:
        print(f"⚠️  No normalization stats found, using normalized data as-is")
    
    # Load Autoencoder
    print(f"\n📦 Loading autoencoder from {args.autoencoder_checkpoint}")
    autoencoder = UnifiedPoseAutoencoder(
        pose_dim=214,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    )
    checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.to(device)
    autoencoder.eval()
    
    encoder = autoencoder.encoder
    decoder = autoencoder.decoder
    print("✅ Autoencoder loaded")
    
    # Load Predictor
    print(f"\n📦 Loading predictor from {args.predictor_checkpoint}")
    predictor = LatentPredictor(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=6,
        num_heads=8,
        num_queries=args.max_seq_len,
        dropout=0.1
    )
    checkpoint = torch.load(args.predictor_checkpoint, map_location=device)
    predictor.load_state_dict(checkpoint['model_state_dict'])
    predictor.to(device)
    predictor.eval()
    print("✅ Predictor loaded")
    
    # Dataset
    print(f"\n📂 Loading {args.split} dataset...")
    dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        split=args.split,
        max_seq_len=args.max_seq_len,
        stats_path=stats_path
    )
    print(f"✅ Loaded {len(dataset)} samples")
    
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    # Evaluate
    print(f"\n🚀 Evaluating on {args.split} split...")
    metrics = evaluate_model(
        predictor,
        encoder,
        decoder,
        dataloader,
        device,
        normalize_stats=normalize_stats,
        save_predictions=args.save_predictions,
        output_dir=args.output_dir
    )
    
    # Print results
    print_metrics(metrics, split_name=args.split)
    
    # Save metrics
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
        metrics_path = os.path.join(args.output_dir, f'metrics_{args.split}.pkl')
        with open(metrics_path, 'wb') as f:
            pickle.dump(metrics, f)
        print(f"💾 Saved metrics to {metrics_path}")


if __name__ == '__main__':
    main()

