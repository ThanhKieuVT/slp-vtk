"""
Script: Check Autoencoder Reconstruction Quality (Stage 1)
✅ FIXED: Tương thích với grouped normalization (manual + NMM)
Mục đích: Kiểm tra chất lượng tái tạo của Autoencoder
"""

import torch
import numpy as np
import argparse
import os
import sys

# Import modules
try:
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from data_preparation import load_sample, normalize_pose, denormalize_pose
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("💡 Đảm bảo file này ở thư mục gốc, cùng cấp với 'models/' và 'data_preparation.py'")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description='Kiểm tra Autoencoder Stage 1')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Đường dẫn đến processed_data/data/')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                        help='Checkpoint của Autoencoder (.pt)')
    parser.add_argument('--output_dir', type=str, default='check_stage1_output',
                        help='Thư mục lưu kết quả')
    parser.add_argument('--split', type=str, default='dev',
                        choices=['train', 'dev', 'test'],
                        help='Split để lấy mẫu test')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Chỉ số mẫu để test (default: mẫu đầu tiên)')
    
    # Model config (phải khớp với lúc train)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--encoder_layers', type=int, default=6)
    parser.add_argument('--decoder_coarse_layers', type=int, default=4)
    parser.add_argument('--decoder_medium_layers', type=int, default=4)
    parser.add_argument('--decoder_fine_layers', type=int, default=6)
    parser.add_argument('--num_heads', type=int, default=8)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)
    
    # ============================================================
    # 1. LOAD GROUPED NORMALIZATION STATS
    # ============================================================
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        print(f"❌ Không tìm thấy stats tại: {stats_path}")
        print("💡 Chạy data_preparation.py để tạo stats trước!")
        return
    
    print(f"📊 Loading grouped stats từ: {stats_path}")
    stats_data = np.load(stats_path)
    stats = {
        'manual_mean': float(stats_data['manual_mean']),
        'manual_std': float(stats_data['manual_std']),
        'nmm_mean': stats_data['nmm_mean'],
        'nmm_std': stats_data['nmm_std']
    }
    print(f"   ✅ Manual: mean={stats['manual_mean']:.4f}, std={stats['manual_std']:.4f}")
    print(f"   ✅ NMM: shape={stats['nmm_mean'].shape}")
    
    # ============================================================
    # 2. LOAD AUTOENCODER MODEL
    # ============================================================
    print(f"\n📦 Loading Autoencoder từ: {args.autoencoder_checkpoint}")
    model = UnifiedPoseAutoencoder(
        pose_dim=214,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        encoder_layers=args.encoder_layers,
        decoder_coarse_layers=args.decoder_coarse_layers,
        decoder_medium_layers=args.decoder_medium_layers,
        decoder_fine_layers=args.decoder_fine_layers,
        num_heads=args.num_heads,
        dropout=0.1
    ).to(device)
    
    try:
        checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
        state_dict = checkpoint.get('model_state_dict', checkpoint)
        model.load_state_dict(state_dict, strict=True)
        print("   ✅ Load weights thành công!")
        
        # In thông tin checkpoint
        if 'epoch' in checkpoint:
            print(f"   📌 Epoch: {checkpoint['epoch']}")
        if 'val_loss' in checkpoint:
            print(f"   📌 Val Loss: {checkpoint['val_loss']:.6f}")
        
        model.eval()
    except Exception as e:
        print(f"❌ Lỗi load checkpoint: {e}")
        return
    
    # ============================================================
    # 3. LOAD SAMPLE DATA
    # ============================================================
    split_dir = os.path.join(args.data_dir, args.split)
    poses_dir = os.path.join(split_dir, "poses")
    
    if not os.path.exists(poses_dir):
        print(f"❌ Không tìm thấy: {poses_dir}")
        return
    
    # Lấy danh sách video IDs
    video_files = sorted([f.replace('.npz', '') for f in os.listdir(poses_dir) 
                         if f.endswith('.npz')])
    
    if not video_files:
        print(f"❌ Không tìm thấy file .npz nào trong {poses_dir}")
        return
    
    if args.sample_idx >= len(video_files):
        print(f"⚠️ sample_idx={args.sample_idx} vượt quá số lượng mẫu ({len(video_files)})")
        args.sample_idx = 0
    
    video_id = video_files[args.sample_idx]
    print(f"\n🔍 Đang test mẫu: {video_id} (idx={args.sample_idx}/{len(video_files)})")
    
    # Load pose data [T, 214]
    pose_original, T = load_sample(video_id, split_dir)
    
    if pose_original is None:
        print(f"❌ Không load được sample {video_id}")
        return
    
    print(f"   ✅ Loaded: shape={pose_original.shape}, length={T}")
    
    # ============================================================
    # 4. NORMALIZE (như trong training)
    # ============================================================
    print("\n🔄 Normalizing với grouped stats...")
    pose_normalized = normalize_pose(pose_original, stats)
    
    # Convert to tensor
    pose_tensor = torch.from_numpy(pose_normalized).float().unsqueeze(0).to(device)  # [1, T, 214]
    
    # Create mask (all valid)
    mask = torch.ones(1, T, dtype=torch.bool, device=device)  # [1, T]
    
    # ============================================================
    # 5. INFERENCE - RECONSTRUCTION
    # ============================================================
    print("🤖 Đang chạy Autoencoder...")
    with torch.no_grad():
        reconstructed, latent = model(pose_tensor, mask=mask)
    
    print(f"   ✅ Reconstructed shape: {reconstructed.shape}")
    print(f"   ✅ Latent shape: {latent.shape}")
    
    # ============================================================
    # 6. DENORMALIZE
    # ============================================================
    print("\n🔄 Denormalizing về tọa độ gốc...")
    reconstructed_np = reconstructed.squeeze(0).cpu().numpy()  # [T, 214]
    
    # Denormalize với grouped stats
    reconstructed_denorm = denormalize_pose(reconstructed_np, stats)
    
    # ============================================================
    # 7. COMPUTE METRICS
    # ============================================================
    print("\n📊 Computing metrics...")
    
    # MSE per feature
    mse_per_feature = np.mean((pose_original - reconstructed_denorm) ** 2, axis=0)  # [214]
    
    # Split metrics
    manual_mse = np.mean(mse_per_feature[:150])  # Manual (150 dims)
    nmm_mse = np.mean(mse_per_feature[150:])     # NMM (64 dims)
    total_mse = np.mean(mse_per_feature)
    
    print(f"   📌 Total MSE: {total_mse:.6f}")
    print(f"   📌 Manual MSE (pose): {manual_mse:.6f}")
    print(f"   📌 NMM MSE (facial): {nmm_mse:.6f}")
    
    # MAE
    mae_per_feature = np.mean(np.abs(pose_original - reconstructed_denorm), axis=0)
    manual_mae = np.mean(mae_per_feature[:150])
    nmm_mae = np.mean(mae_per_feature[150:])
    total_mae = np.mean(mae_per_feature)
    
    print(f"   📌 Total MAE: {total_mae:.6f}")
    print(f"   📌 Manual MAE: {manual_mae:.6f}")
    print(f"   📌 NMM MAE: {nmm_mae:.6f}")
    
    # ============================================================
    # 8. SAVE RESULTS
    # ============================================================
    print("\n💾 Đang lưu kết quả...")
    
    # Save original
    original_path = os.path.join(args.output_dir, f"{video_id}_original.npy")
    np.save(original_path, pose_original)
    print(f"   ✅ Saved: {original_path}")
    
    # Save reconstructed
    recon_path = os.path.join(args.output_dir, f"{video_id}_reconstructed.npy")
    np.save(recon_path, reconstructed_denorm)
    print(f"   ✅ Saved: {recon_path}")
    
    # Save latent
    latent_path = os.path.join(args.output_dir, f"{video_id}_latent.npy")
    np.save(latent_path, latent.squeeze(0).cpu().numpy())
    print(f"   ✅ Saved: {latent_path}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, f"{video_id}_metrics.txt")
    with open(metrics_path, 'w') as f:
        f.write(f"Video ID: {video_id}\n")
        f.write(f"Sequence Length: {T}\n")
        f.write(f"\n=== MSE ===\n")
        f.write(f"Total MSE: {total_mse:.6f}\n")
        f.write(f"Manual MSE: {manual_mse:.6f}\n")
        f.write(f"NMM MSE: {nmm_mse:.6f}\n")
        f.write(f"\n=== MAE ===\n")
        f.write(f"Total MAE: {total_mae:.6f}\n")
        f.write(f"Manual MAE: {manual_mae:.6f}\n")
        f.write(f"NMM MAE: {nmm_mae:.6f}\n")
        f.write(f"\n=== Per-Feature Stats ===\n")
        f.write(f"Manual features (0-149):\n")
        f.write(f"  MSE range: [{mse_per_feature[:150].min():.6f}, {mse_per_feature[:150].max():.6f}]\n")
        f.write(f"  MAE range: [{mae_per_feature[:150].min():.6f}, {mae_per_feature[:150].max():.6f}]\n")
        f.write(f"\nNMM features (150-213):\n")
        f.write(f"  MSE range: [{mse_per_feature[150:].min():.6f}, {mse_per_feature[150:].max():.6f}]\n")
        f.write(f"  MAE range: [{mae_per_feature[150:].min():.6f}, {mae_per_feature[150:].max():.6f}]\n")
    
    print(f"   ✅ Saved: {metrics_path}")
    
    # ============================================================
    # 9. SUMMARY
    # ============================================================
    print("\n" + "="*60)
    print("✅ HOÀN TẤT!")
    print("="*60)
    print(f"📂 Kết quả lưu tại: {args.output_dir}/")
    print(f"   • Original:      {video_id}_original.npy")
    print(f"   • Reconstructed: {video_id}_reconstructed.npy")
    print(f"   • Latent:        {video_id}_latent.npy")
    print(f"   • Metrics:       {video_id}_metrics.txt")
    print("="*60)
    
    # Đánh giá chất lượng
    print("\n📋 ĐÁNH GIÁ CHẤT LƯỢNG:")
    if total_mse < 0.01:
        print("   ✅ Xuất sắc! MSE < 0.01")
    elif total_mse < 0.05:
        print("   ✅ Tốt! MSE < 0.05")
    elif total_mse < 0.1:
        print("   ⚠️  Chấp nhận được. MSE < 0.1")
    else:
        print("   ❌ Kém! MSE > 0.1 - Cần train thêm")
    
    print("\n💡 HƯỚNG DẪN TIẾP THEO:")
    print("1. Xem metrics chi tiết:")
    print(f"   cat {metrics_path}")
    print("\n2. Visualize so sánh (nếu có script):")
    print(f"   python visualize_comparison.py \\")
    print(f"     --original {original_path} \\")
    print(f"     --reconstructed {recon_path}")
    print("\n3. Test với nhiều mẫu khác:")
    print(f"   python check_stage1_autoencoder.py \\")
    print(f"     --data_dir {args.data_dir} \\")
    print(f"     --autoencoder_checkpoint {args.autoencoder_checkpoint} \\")
    print(f"     --sample_idx [0-{len(video_files)-1}]")


if __name__ == '__main__':
    main()