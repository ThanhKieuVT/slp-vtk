import torch
import numpy as np
import argparse
import os
import sys
import glob

# --- Import Modules ---
try:
    sys.path.append(os.getcwd()) 
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    # Import logic chuẩn từ file chị gửi
    from data_preparation import normalize_pose, denormalize_pose, load_sample
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit(1)

def main():
    # --- 1. CẤU HÌNH THAM SỐ ---
    parser = argparse.ArgumentParser(description='Kiểm tra chất lượng Autoencoder (Stage 1)')
    parser.add_argument('--data_dir', type=str, required=True, help='Folder chứa normalization_stats.npz và train/dev/test')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True, help='Đường dẫn file .pt')
    parser.add_argument('--output_dir', type=str, default='check_stage1_output')
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. LOAD STATS (Theo đúng dictionary 4 keys chị đã viết) ---
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        print(f"❌ Không tìm thấy stats tại {stats_path}")
        return
        
    print(f"📊 Loading Grouped Stats từ: {stats_path}")
    s = np.load(stats_path)
    # Tạo dict stats đúng signature hàm của chị
    stats = {
        'manual_mean': s['manual_mean'], 'manual_std': s['manual_std'],
        'nmm_mean': s['nmm_mean'], 'nmm_std': s['nmm_std']
    }

    # --- 3. LOAD MODEL ---
    ae = UnifiedPoseAutoencoder(pose_dim=214, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    ckpt = torch.load(args.autoencoder_checkpoint, map_location=device)
    ae.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
    ae.eval()
    print("✅ Load model thành công!")

    # --- 4. LOAD FILE DATA MẪU ---
    # Sử dụng hàm load_sample của chị để lấy đúng 214D (Pose + NMM)
    print(f"🔍 Đang tìm file mẫu trong {args.data_dir}/train/poses...")
    pose_files = glob.glob(os.path.join(args.data_dir, "train/poses/*.npz"))
    if not pose_files:
        print("❌ Không tìm thấy file dữ liệu mẫu!")
        return
    
    # Lấy video_id từ file đầu tiên tìm thấy
    video_id = os.path.basename(pose_files[0]).replace('.npz', '')
    real_pose_np, T = load_sample(video_id, os.path.join(args.data_dir, "train"))
    print(f"✅ Đã load video: {video_id} (T={T})")

    # --- 5. NORMALIZE (Dùng hàm normalize_pose của chị) ---
    real_pose_norm = normalize_pose(real_pose_np, stats)
    real_pose_input = torch.tensor(real_pose_norm, dtype=torch.float32).to(device).unsqueeze(0)

    # --- 6. INFERENCE (TÁI TẠO) ---
    print("🔄 Đang chạy qua Autoencoder...")
    with torch.no_grad():
        recon_norm, _ = ae(real_pose_input)

    # --- 7. DENORMALIZE & SAVE ---
    recon_norm_np = recon_norm.squeeze(0).cpu().numpy()
    
    # Dùng hàm denormalize_pose của chị để giải chuẩn hóa về tọa độ gốc
    recon_final = denormalize_pose(recon_norm_np, stats) 
    
    # Lưu kết quả
    real_save_path = os.path.join(args.output_dir, "original_sample.npy")
    recon_save_path = os.path.join(args.output_dir, "reconstructed_sample.npy")
    np.save(real_save_path, real_pose_np)
    np.save(recon_save_path, recon_final)
    
    # Tính MSE nhanh để chị xem "sức khỏe"
    mse = np.mean((real_pose_np - recon_final)**2)
    print("\n" + "="*40)
    print(f"📉 Reconstruction MSE: {mse:.8f}")
    print(f"📂 Kết quả lưu tại: {args.output_dir}")
    print("="*40)

if __name__ == '__main__':
    main()