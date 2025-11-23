# Tên file: visualize_pose.py
# UPDATED: Tự động cắt độ dài để so sánh được kể cả khi Model dự đoán sai length
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# --- BẢN ĐỒ KẾT NỐI (GIỮ NGUYÊN) ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]
POSE_CONNECTIONS_UPPER_BODY = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24)
]
MOUTH_OUTER_LIP = list(zip(range(0, 11), range(1, 12))) + [(11, 0)]
MOUTH_INNER_LIP = list(zip(range(12, 19), range(13, 20))) + [(19, 12)]
MOUTH_CONNECTIONS_20 = MOUTH_OUTER_LIP + MOUTH_INNER_LIP

ALL_CONNECTIONS = []
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 0, 'color': 'gray', 'lw': 2} for (s, e) in POSE_CONNECTIONS_UPPER_BODY])
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 33, 'color': 'blue', 'lw': 1.5} for (s, e) in HAND_CONNECTIONS])
ALL_CONNECTIONS.append({'indices': (15, 0), 'offset': (0, 33), 'color': 'blue', 'lw': 2})
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 54, 'color': 'green', 'lw': 1.5} for (s, e) in HAND_CONNECTIONS])
ALL_CONNECTIONS.append({'indices': (16, 0), 'offset': (0, 54), 'color': 'green', 'lw': 2})
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 75, 'color': 'red', 'lw': 1} for (s, e) in MOUTH_CONNECTIONS_20])

MANUAL_UPPER_BODY_IDXS = list(range(23))
LEFT_HAND_IDXS = list(range(33, 54))
RIGHT_HAND_IDXS = list(range(54, 75))
MOUTH_IDXS = list(range(75, 95))
PLOT_IDXS = MANUAL_UPPER_BODY_IDXS + LEFT_HAND_IDXS + RIGHT_HAND_IDXS + MOUTH_IDXS
VALID_POINT_THRESHOLD = 0.1

def load_and_prepare_pose(pose_214):
    manual_150 = pose_214[:, :150]
    manual_kps = manual_150.reshape(-1, 75, 2)
    mouth_40 = pose_214[:, 174:]
    mouth_kps = mouth_40.reshape(-1, 20, 2)
    return np.concatenate([manual_kps, mouth_kps], axis=1)

def animate_poses(gt_path, recon_path, output_video):
    print(f"Đang tải Ground Truth: {gt_path}")
    pose_gt_214 = np.load(gt_path)
    print(f"Đang tải Reconstructed: {recon_path}")
    pose_recon_214 = np.load(recon_path)
    
    kps_gt = load_and_prepare_pose(pose_gt_214)
    kps_recon = load_and_prepare_pose(pose_recon_214)
    
    # --- MỚI: XỬ LÝ CHÊNH LỆCH ĐỘ DÀI ---
    len_gt = len(kps_gt)
    len_recon = len(kps_recon)
    
    if len_gt != len_recon:
        print(f"⚠️ CẢNH BÁO: Độ dài không khớp! GT={len_gt}, Recon={len_recon}")
        print(f"👉 Sẽ cắt về độ dài ngắn nhất để visualization không bị lỗi.")
    
    T = min(len_gt, len_recon) # Lấy min để vòng lặp không bị tràn index
    # ------------------------------------
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    def setup_ax(ax, title):
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Lấy range dựa trên frame đầu tiên của GT cho chuẩn
        plot_points_gt = kps_gt[:, PLOT_IDXS]
        valid_kps = plot_points_gt[np.sum(np.abs(plot_points_gt), axis=2) > VALID_POINT_THRESHOLD]
        if valid_kps.shape[0] > 0:
             min_vals = np.min(valid_kps, axis=0)
             max_vals = np.max(valid_kps, axis=0)
             padding = 0.2 * (max_vals - min_vals)
             padding[padding < 0.1] = 0.1
             ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
             ax.set_ylim(max_vals[1] + padding[1], min_vals[1] - padding[1])
        else:
             ax.set_xlim(-0.5, 0.5)
             ax.set_ylim(0.5, -0.5)
             
        ax.spines['top'].set_visible(True)
        ax.spines['bottom'].set_visible(True)
        ax.spines['left'].set_visible(True)
        ax.spines['right'].set_visible(True)
        
        lines = []
        for item in ALL_CONNECTIONS:
            (start, end) = item['indices']
            offset = item['offset']
            line = Line2D([], [], color=item['color'], lw=item['lw'], alpha=0.8)
            ax.add_line(line)
            if isinstance(offset, (tuple, list)):
                s_off, e_off = offset[0], offset[1]
            else:
                s_off, e_off = offset, offset
            lines.append({'line': line, 'start': start + s_off, 'end': end + e_off})
            
        scatter = ax.scatter([], [], s=2, c='black', alpha=0.4)
        lines.append({'scatter': scatter, 'plot_indices': PLOT_IDXS})
        return lines
    
    artists1 = setup_ax(ax1, 'Ground Truth')
    artists2 = setup_ax(ax2, 'Reconstructed')
    fig.suptitle(f'Frame 0 / {T}')

    def update(frame):
        # Frame này chắc chắn an toàn vì frame < T (min length)
        kps_gt_frame = kps_gt[frame]
        kps_recon_frame = kps_recon[frame] 
        
        all_changed_artists = []
        
        for ax_artists, kps_frame in [(artists1, kps_gt_frame), (artists2, kps_recon_frame)]:
            for item in ax_artists:
                if 'line' in item:
                    idx_start, idx_end = item['start'], item['end']
                    if np.sum(np.abs(kps_frame[idx_start])) > VALID_POINT_THRESHOLD and np.sum(np.abs(kps_frame[idx_end])) > VALID_POINT_THRESHOLD:
                        item['line'].set_data(
                            [kps_frame[idx_start, 0], kps_frame[idx_end, 0]],
                            [kps_frame[idx_start, 1], kps_frame[idx_end, 1]]
                        )
                    else:
                        item['line'].set_data([], [])
                    all_changed_artists.append(item['line'])
                elif 'scatter' in item:
                    pts = kps_frame[item['plot_indices']]
                    valid = pts[np.sum(np.abs(pts), axis=1) > VALID_POINT_THRESHOLD]
                    item['scatter'].set_offsets(valid)
                    all_changed_artists.append(item['scatter'])

        fig.suptitle(f'Frame {frame} / {T}')
        return all_changed_artists

    print(f"Đang tạo animation ({T} frames)...")
    ani = animation.FuncAnimation(fig, update, frames=T, blit=True, interval=40)
    ani.save(output_video, writer='ffmpeg', fps=25, dpi=150)
    print(f"\n🎉 Đã lưu video: {output_video}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--recon_path', type=str, required=True)
    parser.add_argument('--output_video', type=str, default='pose_comparison.mp4')
    args = parser.parse_args()
    
    if os.path.exists(args.gt_path) and os.path.exists(args.recon_path):
        animate_poses(args.gt_path, args.recon_path, args.output_video)
    else:
        print("Lỗi: Không tìm thấy file input.")