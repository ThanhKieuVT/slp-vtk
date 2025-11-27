v"""
Script so sánh Real vs Gen (Phiên bản Point Cloud - Chuẩn không cần chỉnh)
"""
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# Import hàm load dữ liệu chuẩn của chị
sys.path.append(os.getcwd())
try:
    from data_preparation import load_sample
except ImportError:
    print("❌ Lỗi: Không tìm thấy data_preparation.py")
    sys.exit(1)

def visualize_comparison_v2(real_pose, gen_pose, output_path, title_text=""):
    # 1. Cắt ngắn về độ dài chung
    min_len = min(len(real_pose), len(gen_pose))
    real_pose = real_pose[:min_len]
    gen_pose = gen_pose[:min_len]
    
    print(f"🎬 Đang render video so sánh ({min_len} frames)...")
    
    # 2. Reshape về [T, N_points, 2]
    # Dữ liệu 214 chiều -> 107 điểm x 2 (x, y)
    real_data = real_pose.reshape(min_len, -1, 2)
    gen_data = gen_pose.reshape(min_len, -1, 2)
    
    # 3. Setup Plot (2 khung hình cạnh nhau)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Config trục (Lấy max/min từ dữ liệu thật để scale chuẩn)
    all_x = real_data[:, :, 0].flatten()
    all_y = real_data[:, :, 1].flatten()
    
    # Lọc bỏ điểm 0 (padding) để tính giới hạn khung hình chính xác
    valid_mask = (all_x > 0.01) & (all_y > 0.01)
    if valid_mask.sum() > 0:
        x_min, x_max = all_x[valid_mask].min(), all_x[valid_mask].max()
        y_min, y_max = all_y[valid_mask].min(), all_y[valid_mask].max()
    else:
        # Fallback nếu data toàn số 0
        x_min, x_max = 0, 1
        y_min, y_max = 0, 1
        
    # Nới rộng khung hình ra một chút cho đẹp
    margin = 0.1
    w = x_max - x_min
    h = y_max - y_min
    
    # Apply cho cả 2 trục
    for ax in [ax1, ax2]:
        ax.set_xlim(x_min - margin*w, x_max + margin*w)
        ax.set_ylim(y_max + margin*h, y_min - margin*h) # Đảo ngược trục Y để người đứng thẳng
        ax.axis('off') # Tắt khung viền số

    ax1.set_title("REAL (Ground Truth)", color='darkred', fontweight='bold')
    ax2.set_title("GENERATED (AI)", color='darkblue', fontweight='bold')
    fig.suptitle(f"ID: {title_text}", fontsize=10)

    # Init artists (Dùng Scatter - Chấm điểm)
    # Real: Màu đỏ, Gen: Màu xanh
    scat_real = ax1.scatter([], [], s=10, c='red', alpha=0.6, label='Body')
    scat_gen = ax2.scatter([], [], s=10, c='blue', alpha=0.6, label='Body')

    def update(frame):
        # Lấy frame t
        p_real = real_data[frame] # [107, 2]
        p_gen = gen_data[frame]   # [107, 2]
        
        # Lọc bỏ các điểm (0,0) - Điểm rác/padding
        # Giả sử tọa độ chuẩn > 0.001
        mask_r = (np.abs(p_real).sum(axis=1) > 0.001)
        mask_g = (np.abs(p_gen).sum(axis=1) > 0.001)
        
        # Cập nhật dữ liệu
        scat_real.set_offsets(p_real[mask_r])
        scat_gen.set_offsets(p_gen[mask_g])
        
        return scat_real, scat_gen

    ani = animation.FuncAnimation(fig, update, frames=min_len, blit=True, interval=50)
    ani.save(output_path, writer='ffmpeg', fps=20)
    print(f"✅ Đã fix xong! Video lưu tại: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', type=str, required=True, help="File .npy sinh ra")
    parser.add_argument('--data_dir', type=str, required=True, help="Thư mục data gốc")
    parser.add_argument('--video_id', type=str, required=True, help="ID video gốc")
    parser.add_argument('--split', type=str, default='train')
    parser.add_argument('--output_video', type=str, default='compare_fixed.mp4')
    args = parser.parse_args()

    # Load Gen
    if not os.path.exists(args.gen_path):
        print(f"❌ Không thấy file gen: {args.gen_path}")
        return
    gen_pose = np.load(args.gen_path)
    
    # Load Real
    split_dir = os.path.join(args.data_dir, args.split)
    real_pose, T = load_sample(args.video_id, split_dir)
    
    if real_pose is None:
        print(f"❌ Không tìm thấy ID {args.video_id} trong {split_dir}")
        return

    # Vẽ
    visualize_comparison_v2(real_pose, gen_pose, args.output_video, args.video_id)

if __name__ == '__main__':
    main()