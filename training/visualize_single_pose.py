# Tên file: visualize_single_pose.py test cho stage 2
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# --- BẢN ĐỒ KẾT NỐI (SKELETON MAP) ---
# Dựa trên file 1_extract_all_features.py
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
ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 0, 'color': 'gray', 'lw': 2}
    for (s, e) in POSE_CONNECTIONS_UPPER_BODY
])
ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 33, 'color': 'blue', 'lw': 1.5}
    for (s, e) in HAND_CONNECTIONS
])
ALL_CONNECTIONS.append({'indices': (15, 0), 'offset': (0, 33), 'color': 'blue', 'lw': 2})
ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 54, 'color': 'green', 'lw': 1.5}
    for (s, e) in HAND_CONNECTIONS
])
ALL_CONNECTIONS.append({'indices': (16, 0), 'offset': (0, 54), 'color': 'green', 'lw': 2})
ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 75, 'color': 'red', 'lw': 1}
    for (s, e) in MOUTH_CONNECTIONS_20
])
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
    all_kps = np.concatenate([manual_kps, mouth_kps], axis=1)
    return all_kps

def animate_poses(npy_path, output_video):
    print(f"Đang tải Pose: {npy_path}")
    pose_214 = np.load(npy_path)
    
    if pose_214.shape[1] != 214:
        print(f"Lỗi: File .npy có shape {pose_214.shape}, không phải [T, 214]")
        return
        
    kps_data = load_and_prepare_pose(pose_214)
    T = len(kps_data)
    
    fig, ax = plt.subplots(1, 1, figsize=(6, 6))
    
    def setup_ax(ax, title):
        ax.set_title(title)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        
        plot_points_gt = kps_data[:, PLOT_IDXS]
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
                start_offset, end_offset = offset[0], offset[1]
            else:
                start_offset, end_offset = offset, offset
            lines.append({'line': line, 'start': start + start_offset, 'end': end + end_offset})
            
        scatter = ax.scatter([], [], s=2, c='black', alpha=0.4)
        lines.append({'scatter': scatter, 'plot_indices': PLOT_IDXS})
        return lines
    
    artists = setup_ax(ax, 'Generated Pose')
    fig.suptitle(f'Frame 0 / {T}')

    def update(frame):
        kps_frame = kps_data[frame]
        all_changed_artists = []
        
        for item in artists:
            if 'line' in item:
                idx_start = item['start']
                idx_end = item['end']
                
                if np.sum(np.abs(kps_frame[idx_start])) > VALID_POINT_THRESHOLD and np.sum(np.abs(kps_frame[idx_end])) > VALID_POINT_THRESHOLD:
                    item['line'].set_data(
                        [kps_frame[idx_start, 0], kps_frame[idx_end, 0]],
                        [kps_frame[idx_start, 1], kps_frame[idx_end, 1]]
                    )
                else:
                    item['line'].set_data([], [])
                all_changed_artists.append(item['line'])
                
            elif 'scatter' in item:
                plot_indices = item['plot_indices']
                points_to_plot = kps_frame[plot_indices]
                valid_points = points_to_plot[np.sum(np.abs(points_to_plot), axis=1) > VALID_POINT_THRESHOLD]
                item['scatter'].set_offsets(valid_points)
                all_changed_artists.append(item['scatter'])

        fig.suptitle(f'Frame {frame} / {T}')
        return all_changed_artists

    print(f"Đang tạo animation ({T} frames)...")
    ani = animation.FuncAnimation(fig, update, frames=T, blit=True, interval=40)
    ani.save(output_video, writer='ffmpeg', fps=25, dpi=150)
    print(f"\n🎉 Đã lưu video: {output_video}")
    plt.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Trực quan hóa 1 file Pose .npy')
    parser.add_argument('--npy_path', type=str, required=True,
                        help='Đường dẫn đến file .npy (từ inference_latent_flow.py)')
    parser.add_argument('--output_video', type=str, default='generated_pose_video.mp4',
                        help='Tên file video output (mp4)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.npy_path):
        print(f"Lỗi: Không tìm thấy {args.npy_path}")
    else:
        animate_poses(args.npy_path, args.output_video)