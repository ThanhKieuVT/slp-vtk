import torch
import numpy as np
import argparse
import os
import sys
import glob

# --- Import Modules (Xử lý đường dẫn Colab/Local) ---
try:
    sys.path.append(os.getcwd()) 
    # Import Model của Stage 1
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    # Import logic chuẩn từ file chị gửi
    from data_preparation import normalize_pose, denormalize_pose 
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("💡 Gợi ý: Chị nhớ đặt file này ở thư mục gốc (cùng cấp với folder 'models' và 'data_preparation.py')")
    sys.exit(1)

def main():
    # --- 1. CẤU HÌNH THAM SỐ ---
    parser = argparse.ArgumentParser(description='Kiểm tra chất lượng Autoencoder (Stage 1)')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Folder chứa file normalization_stats.npz và các folder train/dev/test')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                        help='Đường dẫn file .pt của Autoencoder')
    parser.add_argument('--output_dir', type=str, default='check_stage1_output',
                        help='Nơi lưu file kết quả')
    
    # Config Model (Phải khớp lúc chị train Stage 1)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)

    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Đang kiểm tra trên device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. LOAD STATS (Theo cấu trúc Grouped Stats của chị) ---
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        print(f"❌ Không tìm thấy normalization_stats.npz tại {args.data_dir}")
        return
        
    print(f"📊 Loading Grouped Stats từ: {stats_path}")
    stats_npz = np.load(stats_path)
    # Chuyển về dictionary để truyền vào hàm normalize_pose
    stats = {k: stats_npz[k] for k in stats_npz.files}

    # --- 3. LOAD MODEL ---
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
        print("✅ Load weights thành công!")
        ae.eval()
    except Exception as e:
        print(f"❌ Lỗi load checkpoint: {e}")
        return

    # --- 4. TÌM & LOAD FILE DATA MẪU ---
    # Tự động tìm 1 file .npz (theo format load_sample của chị) hoặc .npy
    print(f"🔍 Đang tìm file mẫu...")
    files = glob.glob(os.path.join(args.data_dir, "**/*.npz"), recursive=True) + \
            glob.glob(os.path.join(args.data_dir, "**/*.npy"), recursive=True)
    
    valid_files = [f for f in files if "stats" not in f and "output" not in f]
    
    if not valid_files:
        print("❌ Không tìm thấy file dữ liệu nào để test!")
        return

    sample_file = valid_files[0]
    print(f"✅ Đã chọn file mẫu: {sample_file}")
    
    # Ở Stage 1, Model nhận đầu vào 214D. 
    # Nếu file chị chọn là file raw, chị nên dùng hàm load_sample của chị để có đúng 214D
    # Ở đây em giả định chị trỏ vào file đã xử lý hoặc em sẽ lấy key keypoints nếu có.
    data = np.load(sample_file)
    if isinstance(data, np.lib.npyio.NpzFile):
        # Ưu tiên lấy keypoints nếu là file pose 
        real_pose_np = data['keypoints'] if 'keypoints' in data else data[data.files[0]]
    else:
        real_pose_np = data

    # Flatten nếu cần (phải đưa về [T, 214])
    if len(real_pose_np.shape) == 3: # [T, 75, 2] -> [T, 150]
        real_pose_np = real_pose_np.reshape(real_pose_np.shape[0], -1)
        print(f"⚠️ Cảnh báo: Dữ liệu tự động flatten về {real_pose_np.shape}. Hãy chắc chắn nó là 214D.")

    # --- 5. NORMALIZE & INFERENCE ---
    # Sử dụng hàm của chị để đảm bảo đúng logic Grouped Normalization
    real_pose_norm = normalize_pose(real_pose_np, stats)
    
    # Chuyển sang Tensor và thêm Batch Dimension [1, T, 214]
    real_pose_input = torch.tensor(real_pose_norm, dtype=torch.float32).to(device).unsqueeze(0)

    print("🔄 Đang chạy qua Autoencoder...")
    with torch.no_grad():
        recon_norm, _ = ae(real_pose_input)

    # --- 6. DENORMALIZE & SAVE ---
    recon_norm_np = recon_norm.squeeze(0).cpu().numpy()
    
    # Giải chuẩn hóa về scale gốc bằng chính hàm của chị
    recon_final = denormalize_pose(recon_norm_np, stats)
    
    # Tính lỗi MSE cơ bản để chị đánh giá nhanh qua terminal
    mse = np.mean((real_pose_np - recon_final)**2)
    print(f"📉 Reconstruction MSE (Original Scale): {mse:.6f}")

    # Lưu kết quả để so sánh
    real_save_path = os.path.join(args.output_dir, "original_sample.npy")
    recon_save_path = os.path.join(args.output_dir, "reconstructed_sample.npy")
    
    np.save(real_save_path, real_pose_np)
    np.save(recon_save_path, recon_final)
    
    print("\n" + "="*40)
    print("🎉 HOÀN TẤT!")
    print(f"   1. Gốc:      {real_save_path}")
    print(f"   2. Tái tạo:  {recon_save_path}")
    print("="*40)
    print(f"\n👉 Chị chạy lệnh sau để xem video so sánh:")
    print(f"python training/visualize_single_pose.py --npy_path {recon_save_path}")

if __name__ == '__main__':
    main()