"""
Script: Check Autoencoder Reconstruction Quality (CLI Version)
Mục đích: Kiểm tra xem Autoencoder có bị "hỏng" (mất tay) không bằng cách tái tạo 1 file pose thật.
"""

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
    from data_preparation import denormalize_pose
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit(1)

def main():
    # --- 1. CẤU HÌNH ARGPARSE (Theo ý chị) ---
    parser = argparse.ArgumentParser(description='Kiểm tra chất lượng Autoencoder (Stage 1)')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Đường dẫn thư mục chứa normalization_stats.npz và các file data .npy')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                        help='Đường dẫn file checkpoint .pt của Autoencoder')
    parser.add_argument('--output_dir', type=str, default='check_stage1_output',
                        help='Thư mục lưu file kết quả')
    
    args = parser.parse_args()

    # Tạo thư mục output nếu chưa có
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Đang kiểm tra trên device: {device}")

    # --- 2. TÌM FILE DỮ LIỆU ĐỂ TEST ---
    # Tự động tìm 1 file .npy bất kỳ trong data_dir để làm mẫu
    print(f"🔍 Đang tìm file .npy mẫu trong: {args.data_dir}")
    npy_files = glob.glob(os.path.join(args.data_dir, "**/*.npy"), recursive=True)
    # Lọc bỏ các file không phải data (như stats hay output cũ)
    valid_files = [f for f in npy_files if 'stats' not in f and 'output' not in f and 'check' not in f]
    
    if not valid_files:
        print("❌ Không tìm thấy file .npy dữ liệu nào trong thư mục data_dir!")
        return
    
    # Lấy file đầu tiên tìm được
    sample_file = valid_files[0]
    print(f"✅ Đã chọn file mẫu để test: {sample_file}")

    # --- 3. LOAD AUTOENCODER ---
    print(f"📦 Loading Autoencoder từ: {args.autoencoder_checkpoint}")
    ae = UnifiedPoseAutoencoder(
        pose_dim=214, 
        latent_dim=256, 
        hidden_dim=512
    ).to(device)

    try:
        ckpt = torch.load(args.autoencoder_checkpoint, map_location=device)
        if 'model_state_dict' in ckpt:
            ae.load_state_dict(ckpt['model_state_dict'])
        else:
            ae.load_state_dict(ckpt)
        ae.eval()
    except Exception as e:
        print(f"❌ Lỗi load checkpoint: {e}")
        return

    # --- 4. LOAD STATS & NORMALIZE ---
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        # Thử tìm trong thư mục cha nếu không thấy
        stats_path = os.path.join(args.data_dir, "../normalization_stats.npz")
        
    if not os.path.exists(stats_path):
        print(f"❌ Không tìm thấy normalization_stats.npz! Hãy kiểm tra lại data_dir.")
        return
        
    print("📊 Loading Stats...")
    stats = np.load(stats_path)
    mean = torch.tensor(stats['mean']).float().to(device)
    std = torch.tensor(stats['std']).float().to(device)

    # Đọc file mẫu
    real_pose_np = np.load(sample_file)
    real_pose = torch.tensor(real_pose_np, dtype=torch.float32).to(device)
    
    # Normalize: (X - Mean) / Std
    real_pose_norm = (real_pose - mean) / (std + 1e-6)
    real_pose_norm = real_pose_norm.unsqueeze(0) # [1, T, 214]

    # --- 5. RECONSTRUCT (CHẠY QUA AE) ---
    print("🔄 Đang chạy qua Autoencoder...")
    with torch.no_grad():
        recon_norm, _ = ae(real_pose_norm)

    # --- 6. DENORMALIZE & SAVE ---
    print("💾 Đang lưu kết quả...")
    
    # 6.1. Lưu file GỐC (để so sánh)
    real_save_path = os.path.join(args.output_dir, "original_sample.npy")
    np.save(real_save_path, real_pose_np)
    
    # 6.2. Lưu file TÁI TẠO (qua AE)
    recon = recon_norm.squeeze(0).cpu().numpy()
    recon_final = denormalize_pose(recon, stats['mean'], stats['std'])
    
    recon_save_path = os.path.join(args.output_dir, "reconstructed_sample.npy")
    np.save(recon_save_path, recon_final)
    
    print("\n✅ HOÀN TẤT! Kết quả lưu tại:")
    print(f"   1. Gốc: {real_save_path}")
    print(f"   2. Tái tạo: {recon_save_path}")
    print("\n👉 CHẠY LỆNH VISUALIZE ĐỂ KIỂM TRA:")
    print(f"python visualize_single_pose.py --npy_path {recon_save_path} --output_video check_ae_result.mp4")

if __name__ == '__main__':
    main()