import torch
import numpy as np
import argparse
import os
import sys
import glob

# --- Import Modules (Xử lý đường dẫn Colab) ---
try:
    sys.path.append(os.getcwd()) 
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    # Import hàm chuẩn từ data_preparation của chị
    from data_preparation import denormalize_pose, load_sample 
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("💡 Gợi ý: Chị nhớ đặt file này ở thư mục gốc (cùng cấp với folder 'models' và file 'data_preparation.py')")
    sys.exit(1)

def main():
    # --- 1. CẤU HÌNH THAM SỐ ---
    parser = argparse.ArgumentParser(description='Kiểm tra chất lượng Autoencoder (Stage 1) - Grouped Version')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Folder chứa file normalization_stats.npz và các folder poses/nmms')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                        help='Đường dẫn file .pt của Autoencoder')
    parser.add_argument('--output_dir', type=str, default='check_stage1_output',
                        help='Nơi lưu file kết quả')
    
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--pose_dim', type=int, default=214)

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Đang kiểm tra trên device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. LOAD MODEL ---
    print(f"📦 Loading Autoencoder (pose_dim={args.pose_dim})...")
    ae = UnifiedPoseAutoencoder(
        pose_dim=args.pose_dim, 
        latent_dim=args.latent_dim, 
        hidden_dim=args.hidden_dim
    ).to(device)

    try:
        ckpt = torch.load(args.autoencoder_checkpoint, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        ae.load_state_dict(state_dict, strict=True)
        print("✅ Load weights thành công!")
        ae.eval()
    except Exception as e:
        print(f"❌ Lỗi load checkpoint: {e}")
        return

    # --- 3. LOAD & PREPARE GROUPED STATS ---
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        # Thử tìm ở thư mục cha nếu data_dir trỏ vào subfolder train/test
        stats_path = os.path.join(os.path.dirname(args.data_dir), "normalization_stats.npz")
    
    if not os.path.exists(stats_path):
        print(f"❌ Không tìm thấy normalization_stats.npz tại {stats_path}")
        return
        
    print(f"📊 Loading Grouped Stats từ: {stats_path}")
    stats_raw = np.load(stats_path)
    
    # Chuyển đổi format sang dict để dùng cho hàm denormalize_pose của chị
    stats_dict = {
        'manual_mean': float(stats_raw['manual_mean']),
        'manual_std': float(stats_raw['manual_std']),
        'nmm_mean': stats_raw['nmm_mean'],
        'nmm_std': stats_raw['nmm_std']
    }

    # Tạo tensor Full 214D để Normalize đầu vào
    m_mean = np.full(150, stats_dict['manual_mean'])
    m_std = np.full(150, stats_dict['manual_std'])
    full_mean = torch.tensor(np.concatenate([m_mean, stats_dict['nmm_mean']])).float().to(device)
    full_std = torch.tensor(np.concatenate([m_std, stats_dict['nmm_std']])).float().to(device)

    # --- 4. TÌM & LOAD FILE DATA MẪU ---
    # Quét trong folder poses để lấy video_id
    poses_dir = os.path.join(args.data_dir, "poses")
    if not os.path.exists(poses_dir):
        poses_dir = args.data_dir # Fallback
        
    npy_files = glob.glob(os.path.join(poses_dir, "*.npz"))
    if not npy_files:
        print(f"❌ Không tìm thấy file .npz nào trong {poses_dir}")
        return

    sample_id = os.path.basename(npy_files[0]).replace('.npz', '')
    print(f"🔍 Đang test với Video ID: {sample_id}")
    
    # Dùng hàm load_sample chuẩn của chị để lấy đủ 214D
    real_pose_np, T = load_sample(sample_id, args.data_dir)
    
    if real_pose_np is None:
        print("❌ Lỗi load_sample. Vui lòng check đường dẫn data_dir!")
        return

    real_pose_np = np.nan_to_num(real_pose_np)

    # --- 5. NORMALIZE & INFERENCE ---
    real_pose = torch.tensor(real_pose_np, dtype=torch.float32).to(device)
    # Normalize: (X - Mean) / Std
    real_pose_norm = (real_pose - full_mean) / (full_std + 1e-8)
    real_pose_input = real_pose_norm.unsqueeze(0)

    print("🔄 Đang chạy qua Autoencoder...")
    with torch.no_grad():
        recon_norm, _ = ae(real_pose_input)

    # --- 6. DENORMALIZE & EVALUATE ---
    recon_np = recon_norm.squeeze(0).cpu().numpy()
    # Dùng hàm của chị để giải chuẩn hóa theo nhóm
    recon_final = denormalize_pose(recon_np, stats_dict) 

    # Tính MSE trên giá trị gốc
    mse = np.mean((real_pose_np - recon_final)**2)
    
    # --- 7. LƯU KẾT QUẢ ---
    real_save_path = os.path.join(args.output_dir, f"{sample_id}_orig.npy")
    recon_save_path = os.path.join(args.output_dir, f"{sample_id}_recon.npy")
    
    np.save(real_save_path, real_pose_np)
    np.save(recon_save_path, recon_final)
    
    print("\n" + "="*50)
    print(f"📊 KẾT QUẢ KIỂM TRA (MSE): {mse:.8f}")
    if mse < 0.001:
        print("✅ Đánh giá: RẤT TỐT (Stage 1 hoàn hảo)")
    elif mse < 0.01:
        print("⚠️ Đánh giá: TẠM ỔN (Có thể mất chi tiết nhỏ)")
    else:
        print("❌ Đánh giá: KÉM (Cần kiểm tra lại Normalize hoặc Training)")
    print("="*50)
    
    print(f"\n👉 1. File gốc: {real_save_path}")
    print(f"👉 2. File tái tạo: {recon_save_path}")
    print(f"\n💡 Chạy lệnh visualize để xem video:")
    print(f"python training/visualize_single_pose.py --npy_path {recon_save_path} --output_video {sample_id}_check.mp4")

if __name__ == '__main__':
    main()