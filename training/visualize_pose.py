# Tên file: visualize_pose.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle

def load_and_prepare_pose(pose_214):
    """
    Tách pose 214D thành 95 keypoints (x, y) để vẽ
    Bao gồm: 75 manual (tay, thân) + 20 mouth
    """
    # 1. Manual keypoints (75 kps)
    manual_150 = pose_214[:, :150]
    manual_kps = manual_150.reshape(-1, 75, 2)
    
    # 2. Mouth keypoints (20 kps)
    # NMM = aus(17) + head(3) + gaze(4) + mouth(40) = 64
    # Vị trí mouth_flat (40D) là từ 150 + 17 + 3 + 4 = 174
    mouth_40 = pose_214[:, 174:]
    mouth_kps = mouth_40.reshape(-1, 20, 2)
    
    # 3. Kết hợp lại
    all_kps = np.concatenate([manual_kps, mouth_kps], axis=1) # [T, 95, 2]
    
    return all_kps

def animate_poses(gt_path, recon_path, output_video):
    """
    Tạo animation so sánh Ground Truth và Reconstructed
    """
    print(f"Đang tải Ground Truth: {gt_path}")
    pose_gt_214 = np.load(gt_path)
    print(f"Đang tải Reconstructed: {recon_path}")
    pose_recon_214 = np.load(recon_path)
    
    kps_gt = load_and_prepare_pose(pose_gt_214)
    kps_recon = load_and_prepare_pose(pose_recon_214)
    
    T = len(kps_gt) # Số frames
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    def setup_ax(ax, title):
        ax.set_title(title)
        # Đảo ngược trục Y (thường (0,0) ở góc trên bên trái)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        # Set limits (bạn có thể cần điều chỉnh nếu pose bị cắt)
        # Lấy min/max từ GT
        min_vals = np.min(kps_gt.reshape(-1, 2), axis=0)
        max_vals = np.max(kps_gt.reshape(-1, 2), axis=0)
        padding = 0.1 * (max_vals - min_vals)
        ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
        ax.set_ylim(max_vals[1] + padding[1], min_vals[1] - padding[1])
        return ax.scatter([], [], s=10) # Trả về scatter plot

    scat1 = setup_ax(ax1, 'Ground Truth')
    scat2 = setup_ax(ax2, 'Reconstructed')
    
    fig.suptitle(f'Frame 0 / {T}')

    def update(frame):
        # Cập nhật data cho scatter
        scat1.set_offsets(kps_gt[frame])
        scat2.set_offsets(kps_recon[frame])
        
        fig.suptitle(f'Frame {frame} / {T}')
        return scat1, scat2

    print(f"Đang tạo animation ({T} frames)...")
    ani = animation.FuncAnimation(fig, update, frames=T, blit=True, interval=40) # 25 FPS
    
    # Lưu video
    ani.save(output_video, writer='ffmpeg', fps=25, dpi=150)
    print(f"\n🎉 Đã lưu video: {output_video}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trực quan hóa so sánh Pose')
    parser.add_argument('--gt_path', type=str, required=True,
                        help='Đường dẫn đến file .npy của Ground Truth (từ check_autoencoder.py)')
    parser.add_argument('--recon_path', type=str, required=True,
                        help='Đường dẫn đến file .npy của Reconstructed (từ check_autoencoder.py)')
    parser.add_argument('--output_video', type=str, default='pose_comparison.mp4',
                        help='Tên file video output (mp4)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gt_path):
        print(f"Lỗi: Không tìm thấy {args.gt_path}")
    elif not os.path.exists(args.recon_path):
        print(f"Lỗi: Không tìm thấy {args.recon_path}")
    else:
        animate_poses(args.gt_path, args.recon_path, args.output_video)