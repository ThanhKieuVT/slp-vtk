# Tên file: check_autoencoder.py
import os
import argparse
import torch
import numpy as np
from torch.utils.data import DataLoader

# Import từ các file của bạn
from dataset import SignLanguageDataset, collate_fn
from models.fml.autoencoder import UnifiedPoseAutoencoder
from data_preparation import denormalize_pose

def check_reconstruction(args):
    """
    Tải mô hình Stage 1, chạy tái tạo trên 1 sample,
    giải chuẩn hóa và lưu kết quả.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Sử dụng thiết bị: {device}")

    # --- 1. Tải Normalization Stats ---
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        print(f"Lỗi: Không tìm thấy {stats_path}")
        return
    
    stats = np.load(stats_path)
    mean = stats['mean']
    std = stats['std']
    print(f"✅ Đã tải stats từ {stats_path}")

    # --- 2. Tải Autoencoder (Stage 1) ---
    print(f"📦 Đang tải autoencoder từ {args.autoencoder_checkpoint}")
    autoencoder = UnifiedPoseAutoencoder(
        pose_dim=214,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    )
    checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.to(device)
    autoencoder.eval()
    print("✅ Autoencoder đã được tải")

    # --- 3. Tải Dataset (dùng tập 'dev' để kiểm tra) ---
    print(f"📂 Đang tải {args.split} dataset...")
    dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        split=args.split,
        max_seq_len=args.max_seq_len,
        stats_path=stats_path
    )
    
    if args.sample_idx >= len(dataset):
        print(f"Lỗi: sample_idx {args.sample_idx} vượt quá số lượng mẫu {len(dataset)}")
        return

    # Lấy 1 sample
    sample = dataset[args.sample_idx]
    
    # Tạo batch (batch_size=1)
    batch = collate_fn([sample])
    
    # Chuyển batch lên device
    poses_gt_norm = batch['poses'].to(device) # Đây là pose đã chuẩn hóa
    pose_mask = batch['pose_mask'].to(device)
    seq_length = batch['seq_lengths'][0].item()
    video_id = batch['video_ids'][0]
    
    print(f"✅ Đã tải sample: {video_id} (index {args.sample_idx}), độ dài: {seq_length} frames")

    # --- 4. Chạy tái tạo ---
    with torch.no_grad():
        reconstructed_pose_norm, _ = autoencoder(poses_gt_norm, mask=pose_mask)
    
    # Chuyển về numpy
    poses_gt_norm_np = poses_gt_norm.squeeze(0).cpu().numpy()
    reconstructed_pose_norm_np = reconstructed_pose_norm.squeeze(0).cpu().numpy()
    
    # Cắt bỏ padding
    poses_gt_norm_np = poses_gt_norm_np[:seq_length]
    reconstructed_pose_norm_np = reconstructed_pose_norm_np[:seq_length]

    # --- 5. Giải chuẩn hóa (QUAN TRỌNG) ---
    pose_gt_denorm = denormalize_pose(poses_gt_norm_np, mean, std)
    pose_recon_denorm = denormalize_pose(reconstructed_pose_norm_np, mean, std)
    print("✅ Đã giải chuẩn hóa (denormalize) 2 poses")

    # --- 6. Lưu kết quả ---
    os.makedirs(args.output_dir, exist_ok=True)
    gt_path = os.path.join(args.output_dir, f"{video_id}_gt.npy")
    recon_path = os.path.join(args.output_dir, f"{video_id}_recon.npy")
    
    np.save(gt_path, pose_gt_denorm)
    np.save(recon_path, pose_recon_denorm)
    
    print(f"\n🎉 Thành công!")
    print(f"  Đã lưu Ground Truth: {gt_path}")
    print(f"  Đã lưu Reconstructed: {recon_path}")
    print(f"\n👉 Bước tiếp theo: Chạy visualize_pose.py để xem kết quả:")
    print(f"python visualize_pose.py --gt_path {gt_path} --recon_path {recon_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Kiểm tra chất lượng Autoencoder (Stage 1)')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Đường dẫn đến processed_data/data/')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                        help='Checkpoint của Autoencoder (best_model.pt của Stage 1)')
    parser.add_argument('--output_dir', type=str, default='check_stage1_output',
                        help='Thư mục lưu 2 file .npy')
    
    # Các tham số này phải khớp với lúc bạn train Stage 1
    parser.add_argument('--latent_dim', type=int, default=256, help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=512, help='Hidden dimension')
    parser.add_argument('--max_seq_len', type=int, default=120, help='Max sequence length')
    
    parser.add_argument('--split', type=str, default='dev', choices=['train', 'dev', 'test'],
                        help='Dataset split để lấy mẫu')
    parser.add_argument('--sample_idx', type=int, default=0,
                        help='Index của sample trong dataset')
    
    args = parser.parse_args()
    
    check_reconstruction(args)