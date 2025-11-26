"""
Script: Check Autoencoder Reconstruction Quality (CLI Version) - SAFE MODE
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
    # --- 1. CẤU HÌNH ARGPARSE ---
    parser = argparse.ArgumentParser(description='Kiểm tra chất lượng Autoencoder (Stage 1)')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Đường dẫn thư mục chứa normalization_stats.npz và các file data .npy')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                        help='Đường dẫn file checkpoint .pt của Autoencoder')
    parser.add_argument('--output_dir', type=str, default='check_stage1_output',
                        help='Thư mục lưu file kết quả')
    
    # Thêm các tham số config model (để tránh lỗi nếu chị từng thay đổi khi train)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)

    args = parser.parse_args()

    # Tạo thư mục output nếu chưa có
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Đang kiểm tra trên device: {device}")

    # --- 2. TÌM FILE DỮ LIỆU ĐỂ TEST ---
    print(f"🔍 Đang tìm file .npy mẫu trong: {args.data_dir}")
    npy_files = glob.glob(os.path.join(args.data_dir, "**/*.npy"), recursive=True)
    valid_files = [f for f in npy_files if 'stats' not in f and 'output' not in f and 'check' not in f]
    
    if not valid_files:
        print("❌ Không tìm thấy file .npy dữ liệu nào trong thư mục data_dir!")
        return
    
    sample_file = valid_files[0]
    print(f"✅ Đã chọn file mẫu để test: {sample_file}")

    # --- 3. LOAD AUTOENCODER ---
    print(f"📦 Loading Autoencoder từ: {args.autoencoder_checkpoint}")
    
    # ⚠️ QUAN TRỌNG: Model này phải khởi tạo giống hệt lúc Train
    ae = UnifiedPoseAutoencoder(
        pose_dim=214, 
        latent_dim=args.latent_dim, 
        hidden_dim=args.hidden_dim
        # Nếu lúc train chị chỉnh số layers khác mặc định, phải sửa cứng ở đây
        # ví dụ: encoder_layers=4
    ).to(device)

    try:
        ckpt = torch.load(args.autoencoder_checkpoint, map_location=device)
        
        # Xử lý linh hoạt dictionary
        if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
            state_dict = ckpt['model_state_dict']
        else:
            state_dict = ckpt
            
        # Load strict=True để đảm bảo không sai lệch layer nào
        ae.load_state_dict(state_dict, strict=True)
        print("✅ Load weights thành công (Strict Mode)!")
        ae.eval()
        
    except RuntimeError as e:
        print(f"⚠️ Lỗi khớp cấu trúc Model (Shape Mismatch): {e}")
        print("👉 Gợi ý: Kiểm tra xem 'hidden_dim' hoặc số layers trong code này có khớp với file checkpoint không.")
        return
    except Exception as e:
        print(f"❌ Lỗi load checkpoint: {e}")
        return

    # --- 4. LOAD STATS & NORMALIZE ---
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(args.data_dir, "../normalization_stats.npz")
        
    if not os.path.exists(stats_path):
        print(f"❌ Không tìm thấy normalization_stats.npz! Hãy kiểm tra lại data_dir.")
        return
        
    print("📊 Loading Stats...")
    stats = np.load(stats_path)
    # Load lên GPU để tính toán normalize
    mean = torch.tensor(stats['mean']).float().to(device)
    std = torch.tensor(stats['std']).float().to(device)

    # Đọc file mẫu
    real_pose_np = np.load(sample_file)
    real_pose = torch.tensor(real_pose_np, dtype=torch.float32).to(device)
    
    # Normalize: (X - Mean) / Std
    # Thêm 1e-6 để tránh chia cho 0 nếu std có chỗ bằng 0
    real_pose_norm = (real_pose - mean) / (std + 1e-6)
    real_pose_norm = real_pose_norm.unsqueeze(0) # [1, T, 214]

    # --- 5. RECONSTRUCT ---
    print("🔄 Đang chạy qua Autoencoder...")
    with torch.no_grad():
        # Mask=None vì ta đang test 1 file trọn vẹn, không có padding thừa
        recon_norm, _ = ae(real_pose_norm)

    # --- 6. DENORMALIZE & SAVE ---
    print("💾 Đang lưu kết quả...")
    
    real_save_path = os.path.join(args.output_dir, "original_sample.npy")
    np.save(real_save_path, real_pose_np)
    
    recon = recon_norm.squeeze(0).cpu().numpy()
    # Denormalize bằng numpy array gốc trong stats (hàm denormalize thường nhận numpy)
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