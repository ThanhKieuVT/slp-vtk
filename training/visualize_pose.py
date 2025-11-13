# Tên file: visualize_pose.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# --- BẢN ĐỒ KẾT NỐI (SKELETON MAP) ---

# 1. KẾT NỐI MIỆNG (20 ĐIỂM)
# Giả định: 0-11 là môi ngoài, 12-19 là môi trong
MOUTH_OUTER_LIP = list(zip(range(0, 11), range(1, 12))) + [(11, 0)]
MOUTH_INNER_LIP = list(zip(range(12, 19), range(13, 20))) + [(19, 12)]
MOUTH_CONNECTIONS_20 = MOUTH_OUTER_LIP + MOUTH_INNER_LIP

# 2. KẾT NỐI BỘ XƯƠNG (75 ĐIỂM)
# --------------------------------------------------------------------
# ⚠️  CẢNH BÁO: ĐÂY CHỈ LÀ PLACEHOLDER
# Bạn PHẢI thay thế phần này bằng bản đồ kết nối (skeleton map)
# chính xác cho 75 keypoints của bộ dữ liệu của bạn.
# (Ví dụ: OpenPose 25 body + 21 left hand + 21 right hand = 67)
# (Bộ dữ liệu của bạn có 75 điểm, có thể bao gồm cả các điểm trên mặt)
#
# Ví dụ (hoàn toàn là giả định):
# SKELETON_CONNECTIONS_75 = [
#     # Thân (ví dụ)
#     (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
#     # Tay trái (ví dụ)
#     (25, 26), (26, 27), (27, 28), (28, 29),
#     # ... (thêm tất cả các kết nối khác) ...
# ]
# --------------------------------------------------------------------
# Để chạy thử, tôi sẽ chỉ nối 10 điểm đầu tiên
SKELETON_CONNECTIONS_75 = list(zip(range(0, 10), range(1, 11)))


def load_and_prepare_pose(pose_214):
    """
    Tách pose 214D thành 95 keypoints (x, y) để vẽ
    Bao gồm: 75 manual (tay, thân) + 20 mouth
   
    """
    # 1. Manual keypoints (75 kps)
    manual_150 = pose_214[:, :150]
    manual_kps = manual_150.reshape(-1, 75, 2)
    
    # 2. Mouth keypoints (20 kps)
    # Vị trí mouth_flat (40D) là từ 174
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
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Lấy min/max từ GT
        min_vals = np.min(kps_gt.reshape(-1, 2), axis=0)
        max_vals = np.max(kps_gt.reshape(-1, 2), axis=0)
        padding = 0.1 * (max_vals - min_vals)
        ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
        ax.set_ylim(max_vals[1] + padding[1], min_vals[1] - padding[1])
        
        # Tạo các đối tượng Line2D rỗng
        lines = []
        
        # 1. Vẽ skeleton (75 điểm) - offset = 0
        for (start, end) in SKELETON_CONNECTIONS_75:
            line = Line2D([], [], color='blue', lw=2, alpha=0.7)
            ax.add_line(line)
            lines.append({'line': line, 'start': start, 'end': end, 'offset': 0})
        
        # 2. Vẽ miệng (20 điểm) - offset = 75
        for (start, end) in MOUTH_CONNECTIONS_20:
            line = Line2D([], [], color='red', lw=1, alpha=0.8)
            ax.add_line(line)
            lines.append({'line': line, 'start': start, 'end': end, 'offset': 75})
            
        # Thêm các điểm (scatter) để thấy rõ các khớp
        scatter = ax.scatter([], [], s=5)
        lines.append({'scatter': scatter})

        return lines # Trả về list các đối tượng
    
    artists1 = setup_ax(ax1, 'Ground Truth')
    artists2 = setup_ax(ax2, 'Reconstructed')
    
    fig.suptitle(f'Frame 0 / {T}')

    def update(frame):
        kps_gt_frame = kps_gt[frame]     # [95, 2]
        kps_recon_frame = kps_recon[frame] # [95, 2]
        
        all_changed_artists = []
        
        # Cập nhật cho ax1 (GT)
        for item in artists1:
            if 'line' in item:
                idx_start = item['start'] + item['offset']
                idx_end = item['end'] + item['offset']
                item['line'].set_data(
                    [kps_gt_frame[idx_start, 0], kps_gt_frame[idx_end, 0]],
                    [kps_gt_frame[idx_start, 1], kps_gt_frame[idx_end, 1]]
                )
                all_changed_artists.append(item['line'])
            elif 'scatter' in item:
                item['scatter'].set_offsets(kps_gt_frame)
                all_changed_artists.append(item['scatter'])

        # Cập nhật cho ax2 (Recon)
        for item in artists2:
            if 'line' in item:
                idx_start = item['start'] + item['offset']
                idx_end = item['end'] + item['offset']
                item['line'].set_data(
                    [kps_recon_frame[idx_start, 0], kps_recon_frame[idx_end, 0]],
                    [kps_recon_frame[idx_start, 1], kps_recon_frame[idx_end, 1]]
                )
                all_changed_artists.append(item['line'])
            elif 'scatter' in item:
                item['scatter'].set_offsets(kps_recon_frame)
                all_changed_artists.append(item['scatter'])

        fig.suptitle(f'Frame {frame} / {T}')
        
        return all_changed_artists

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
    parser.add_argument('--output_video', type=str, default='pose_comparison_skeleton.mp4',
                        help='Tên file video output (mp4)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gt_path):
        print(f"Lỗi: Không tìm thấy {args.gt_path}")
    elif not os.path.exists(args.recon_path):
        print(f"Lỗi: Không tìm thấy {args.recon_path}")
    else:
        animate_poses(args.gt_path, args.recon_path, args.output_video)