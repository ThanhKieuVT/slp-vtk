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
    from data_preparation import denormalize_pose 
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("💡 Gợi ý: Chị nhớ đặt file này ở thư mục gốc (cùng cấp với folder 'models' và file 'data_preparation.py')")
    sys.exit(1)

def main():
    # --- 1. CẤU HÌNH THAM SỐ ---
    parser = argparse.ArgumentParser(description='Kiểm tra chất lượng Autoencoder (Stage 1)')
    
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Folder chứa file normalization_stats.npz')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                        help='Đường dẫn file .pt của Autoencoder')
    parser.add_argument('--output_dir', type=str, default='check_stage1_output',
                        help='Nơi lưu file kết quả')
    
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--pose_dim', type=int, default=214, help='Số lượng feature của Pose')

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

    # --- 3. LOAD STATS ---
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        stats_path = os.path.join(args.data_dir, "../normalization_stats.npz")
    
    if not os.path.exists(stats_path):
        print(f"❌ Không tìm thấy normalization_stats.npz")
        return
        
    stats = np.load(stats_path)
    mean = torch.tensor(stats['mean']).float().to(device)
    std = torch.tensor(stats['std']).float().to(device)

    # --- 4. TÌM & LOAD FILE DATA MẪU ---
    npy_files = glob.glob(os.path.join(args.data_dir, "**/*.npy"), recursive=True)
    valid_files = [f for f in npy_files if all(k not in f for k in ["stats", "output", "check"])]
    
    if not valid_files:
        print("❌ Không tìm thấy file dữ liệu nào!")
        return

    sample_file = valid_files[0]
    print(f"✅ Đã chọn file mẫu: {sample_file}")
    
    real_pose_np = np.load(sample_file)
    real_pose_np = np.nan_to_num(real_pose_np) # Xử lý NaN rác

    if real_pose_np.shape[-1] != args.pose_dim:
        print(f"⚠️ Cảnh báo: Shape file ({real_pose_np.shape[-1]}) khác với cấu hình pose_dim ({args.pose_dim})")
        # Tự động điều chỉnh nếu chị lỡ nhập sai pose_dim
        # real_pose_np = real_pose_np[:, :args.pose_dim] 

    # --- 5. NORMALIZE & INFERENCE ---
    real_pose = torch.tensor(real_pose_np, dtype=torch.float32).to(device)
    real_pose_norm = (real_pose - mean) / (std + 1e-6)
    real_pose_input = real_pose_norm.unsqueeze(0)

    print("🔄 Đang chạy qua Autoencoder...")
    with torch.no_grad():
        recon_norm, _ = ae(real_pose_input)

    # --- 6. DENORMALIZE & EVALUATE ---
    recon_np = recon_norm.squeeze(0).cpu().numpy()
    recon_final = denormalize_pose(recon_np, stats['mean'], stats['std']) 

    # TÍNH TOÁN MSE (CÀNG NHỎ CÀNG TỐT)
    mse = np.mean((real_pose_np - recon_final)**2)
    
    # --- 7. LƯU KẾT QUẢ ---
    real_save_path = os.path.join(args.output_dir, "original_sample.npy")
    recon_save_path = os.path.join(args.output_dir, "reconstructed_sample.npy")
    
    np.save(real_save_path, real_pose_np)
    np.save(recon_save_path, recon_final)
    
    print("\n" + "="*50)
    print(f"📊 KẾT QUẢ KIỂM TRA:")
    print(f"   🔹 MSE Error: {mse:.8f}")
    if mse < 0.001:
        print("   🔹 Đánh giá: RẤT TỐT (Hầu như không mất thông tin)")
    elif mse < 0.01:
        print("   🔹 Đánh giá: TỐT (Có thể dùng cho Stage 2)")
    else:
        print("   🔹 Đánh giá: CẢNH BÁO (Tái tạo kém, cần train thêm)")
    
    print("-" * 50)
    print(f"   📁 Files đã lưu tại: {args.output_dir}")
    print("="*50)
    print(f"\n👉 Chạy lệnh này để xem video so sánh:")
    print(f"python training/visualize_single_pose.py --npy_path {recon_save_path} --output_video check_ae_result.mp4")

if __name__ == '__main__':
    main()