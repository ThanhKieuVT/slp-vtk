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
    # Import 2 hàm chuẩn từ file của chị
    from data_preparation import normalize_pose, denormalize_pose 
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("💡 Gợi ý: Đặt file này ở thư mục gốc (cùng cấp với folder 'models' và 'data_preparation.py')")
    sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Kiểm tra chất lượng Autoencoder (Stage 1)')
    parser.add_argument('--data_dir', type=str, required=True, help='Thư mục chứa normalization_stats.npz')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True, help='Đường dẫn file .pt')
    parser.add_argument('--output_dir', type=str, default='check_stage1_output')
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 1. LOAD STATS (Theo cấu trúc mới của chị) ---
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        print(f"❌ Không tìm thấy stats tại {stats_path}")
        return
        
    print(f"📊 Loading Grouped Stats từ: {stats_path}")
    # Load stats thành dictionary để truyền vào hàm của chị
    stats_npz = np.load(stats_path)
    stats = {k: stats_npz[k] for k in stats_npz.files}

    # --- 2. LOAD MODEL ---
    print(f"📦 Loading Autoencoder...")
    ae = UnifiedPoseAutoencoder(
        pose_dim=214, 
        latent_dim=args.latent_dim, 
        hidden_dim=args.hidden_dim
    ).to(device)

    try:
        ckpt = torch.load(args.autoencoder_checkpoint, map_location=device)
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        ae.load_state_dict(state_dict, strict=True)
        ae.eval()
        print("✅ Load weights thành công!")
    except Exception as e:
        print(f"❌ Lỗi load checkpoint: {e}")
        return

    # --- 3. LOAD DATA MẪU ---
    # Tìm file .npz (cấu trúc chị dùng trong data_preparation) hoặc .npy
    files = glob.glob(os.path.join(args.data_dir, "**/*.npz"), recursive=True) + \
            glob.glob(os.path.join(args.data_dir, "**/*.npy"), recursive=True)
    
    valid_files = [f for f in files if "stats" not in f and "output" not in f]
    
    if not valid_files:
        print("❌ Không tìm thấy dữ liệu mẫu!")
        return

    sample_file = valid_files[0]
    print(f"✅ Chọn file mẫu: {sample_file}")
    
    data = np.load(sample_file)
    # Nếu là file .npz của chị, lấy key 'keypoints', nếu là .npy thì lấy trực tiếp
    real_pose_np = data['keypoints'] if sample_file.endswith('.npz') and 'keypoints' in data else data
    
    # Nếu pose đang là [T, 75, 2], cần flatten về [T, 150] rồi nối với NMM nếu cần
    # Lưu ý: Ở đây script giả định file mẫu đã là 214D. 
    # Nếu file mẫu chỉ là 150D, chị cần dùng hàm load_sample trong data_preparation.py của chị.
    if real_pose_np.shape[-1] != 214:
        print(f"⚠️ Cảnh báo: File mẫu có shape {real_pose_np.shape}, không phải 214D.")
        print("💡 Script sẽ cố gắng chạy nếu bạn đã xử lý concat trước đó.")

    # --- 4. CHUẨN HÓA & INFERENCE ---
    # Sử dụng hàm normalize của chị (Hỗ trợ cả numpy/torch)
    real_pose_norm = normalize_pose(real_pose_np, stats)
    real_pose_tensor = torch.tensor(real_pose_norm, dtype=torch.float32).to(device).unsqueeze(0)

    print("🔄 Đang tái tạo qua Autoencoder...")
    with torch.no_grad():
        recon_tensor, _ = ae(real_pose_tensor)

    # --- 5. GIẢI CHUẨN HÓA & LƯU ---
    recon_norm_np = recon_tensor.squeeze(0).cpu().numpy()
    
    # Sử dụng hàm denormalize của chị
    recon_final = denormalize_pose(recon_norm_np, stats)
    
    # Tính lỗi MSE đơn giản để check nhanh
    mse = np.mean((real_pose_np - recon_final)**2)
    print(f"📉 Reconstruction MSE: {mse:.6f}")

    # Lưu kết quả
    original_path = os.path.join(args.output_dir, "original.npy")
    recon_path = os.path.join(args.output_dir, "reconstructed.npy")
    
    np.save(original_path, real_pose_np)
    np.save(recon_path, recon_final)
    
    print(f"\n✅ Đã lưu file gốc tại: {original_path}")
    print(f"✅ Đã lưu file tái tạo tại: {recon_path}")
    print(f"\n👉 Chị chạy lệnh visualize để xem kết quả: \npython training/visualize_single_pose.py --npy_path {recon_path}")

if __name__ == '__main__':
    main()