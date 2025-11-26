"""
Script: Check Autoencoder Reconstruction Quality (Colab Compatible)
Mục đích: "Khám sức khỏe" Autoencoder Stage 1.
Logic: 
1. Lấy 1 file pose thật (.npy).
2. Chuẩn hóa (Normalize) bằng stats của chị.
3. Đưa qua Autoencoder để tái tạo.
4. Giải chuẩn hóa (Denormalize) về tọa độ gốc.
5. Lưu 2 file (Gốc vs Tái tạo) để so sánh.
"""

import torch
import numpy as np
import argparse
import os
import sys
import glob

# --- Import Modules (Xử lý đường dẫn Colab) ---
try:
    sys.path.append(os.getcwd()) 
    # Import Model
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    # Import hàm denormalize chuẩn từ file chị gửi
    from data_preparation import denormalize_pose 
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("💡 Gợi ý: Chị nhớ đặt file này ở thư mục gốc (cùng cấp với folder 'models' và file 'data_preparation.py')")
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
    
    # Config Model (Phải khớp lúc train)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Đang kiểm tra trên device: {device}")
    os.makedirs(args.output_dir, exist_ok=True)

    # --- 2. LOAD MODEL ---
    print(f"📦 Loading Autoencoder từ: {args.autoencoder_checkpoint}")
    ae = UnifiedPoseAutoencoder(
        pose_dim=214, # -> Combine 214D
        latent_dim=args.latent_dim, 
        hidden_dim=args.hidden_dim
    ).to(device)

    try:
        ckpt = torch.load(args.autoencoder_checkpoint, map_location=device)
        # Xử lý dict an toàn
        state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
        ae.load_state_dict(state_dict, strict=True)
        print("✅ Load weights thành công (Strict Mode)!")
        ae.eval()
    except Exception as e:
        print(f"❌ Lỗi load checkpoint: {e}")
        return

    # --- 3. LOAD STATS ---
    # Code sẽ tự tìm file stats trong data_dir hoặc thư mục cha
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(args.data_dir, "../normalization_stats.npz")
    
    if not os.path.exists(stats_path):
        print(f"❌ Không tìm thấy normalization_stats.npz tại {args.data_dir}")
        return
        
    print(f"📊 Loading Stats từ: {stats_path}")
    stats = np.load(stats_path)
    # Load lên GPU để tính toán cho nhanh
    mean = torch.tensor(stats['mean']).float().to(device)
    std = torch.tensor(stats['std']).float().to(device)

    # --- 4. TÌM & LOAD FILE DATA MẪU ---
    # Tự động quét tìm 1 file .npy bất kỳ trong data_dir để test
    print(f"🔍 Đang tìm file .npy mẫu...")
    npy_files = glob.glob(os.path.join(args.data_dir, "**/*.npy"), recursive=True)
    
    # Lọc bỏ file rác (stats, output cũ)
    valid_files = [f for f in npy_files if "stats" not in f and "output" not in f and "check" not in f]
    
    if not valid_files:
        print("❌ Không tìm thấy file .npy dữ liệu nào để test!")
        return

    sample_file = valid_files[0]
    print(f"✅ Đã chọn file mẫu: {sample_file}")
    
    try:
        real_pose_np = np.load(sample_file) # [T, 214]
    except Exception as e:
        print(f"❌ Lỗi đọc file .npy: {e}")
        return

    # Chuyển sang Tensor
    real_pose = torch.tensor(real_pose_np, dtype=torch.float32).to(device)

    # --- 5. NORMALIZE (Mô phỏng đầu vào Model) ---
    # Công thức: (X - Mean) / Std
    real_pose_norm = (real_pose - mean) / (std + 1e-6)
    
    # Thêm batch dimension: [T, 214] -> [1, T, 214]
    real_pose_input = real_pose_norm.unsqueeze(0)

    # --- 6. INFERENCE (TÁI TẠO) ---
    print("🔄 Đang chạy qua Autoencoder...")
    with torch.no_grad():
        recon_norm, _ = ae(real_pose_input)

    # --- 7. DENORMALIZE & SAVE ---
    print("💾 Đang lưu kết quả...")
    
    # 7.1. Lưu Ground Truth (Dữ liệu gốc)
    real_save_path = os.path.join(args.output_dir, "original_sample.npy")
    np.save(real_save_path, real_pose_np)
    
    # 7.2. Lưu Reconstruction (Kết quả tái tạo)
    recon_np = recon_norm.squeeze(0).cpu().numpy()
    
    # Dùng hàm của chị để giải chuẩn hóa: X_new * Std + Mean
    recon_final = denormalize_pose(recon_np, stats['mean'], stats['std']) 
    
    recon_save_path = os.path.join(args.output_dir, "reconstructed_sample.npy")
    np.save(recon_save_path, recon_final)
    
    print("\n" + "="*40)
    print("🎉 HOÀN TẤT! Kết quả đã lưu tại:")
    print(f"   1. Gốc (GT):      {real_save_path}")
    print(f"   2. Tái tạo (Rec): {recon_save_path}")
    print("="*40)
    print("\n👉 COPY LỆNH SAU ĐỂ XEM VIDEO SO SÁNH:")
    print(f"python training/visualize_single_pose.py --npy_path {recon_save_path} --output_video check_ae_result.mp4")

if __name__ == '__main__':
    main()