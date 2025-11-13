# Tên file: visualize_pose.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# --- BẢN ĐỒ KẾT NỐI (SKELETON MAP) ---
# Dựa trên file 1_extract_all_features.py
# 0-32: MediaPipe Holistic Pose (33 điểm)
# 33-53: MediaPipe Left Hand (21 điểm)
# 54-74: MediaPipe Right Hand (21 điểm)
# --- (TỔNG = 75 ĐIỂM "MANUAL") ---
# 75-94: 20 điểm Miệng (từ NMMs)

# 1. Kết nối 21 điểm bàn tay (dùng chung)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),         # Ngón trỏ
    (0, 9), (9, 10), (10, 11), (11, 12),    # Ngón giữa
    (0, 13), (13, 14), (14, 15), (15, 16),  # Ngón áp út
    (0, 17), (17, 18), (18, 19), (19, 20)   # Ngón út
]

# 2. Kết nối Thân + Mặt (Holistic Pose 33 điểm - BỎ QUA CHÂN)
# Chỉ số (index) tham chiếu đến MediaPipe
POSE_CONNECTIONS_UPPER_BODY = [
    # Mặt
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), 
    # Thân
    (11, 12), (12, 14), (14, 16), (11, 13), (13, 15),
    (11, 23), (12, 24), (23, 24),
    (12, 11) # Vai
]

# 3. Kết nối 20 điểm miệng (từ NMMs)
MOUTH_OUTER_LIP = list(zip(range(0, 11), range(1, 12))) + [(11, 0)]
MOUTH_INNER_LIP = list(zip(range(12, 19), range(13, 20))) + [(19, 12)]
MOUTH_CONNECTIONS_20 = MOUTH_OUTER_LIP + MOUTH_INNER_LIP

# --- TỔNG HỢP BẢN ĐỒ KẾT NỐI ---
ALL_CONNECTIONS = []

# 1. Thân + Mặt (Indices 0-32)
ALL_CONNECTIONS.extend([
    {'indices': (start, end), 'offset': 0, 'color': 'gray', 'lw': 2}
    for (start, end) in POSE_CONNECTIONS_UPPER_BODY
])

# 2. Tay Trái (Indices 33-53)
ALL_CONNECTIONS.extend([
    {'indices': (start, end), 'offset': 33, 'color': 'blue', 'lw': 1.5}
    for (start, end) in HAND_CONNECTIONS
])
# Nối cổ tay (Wrist) của Thân (điểm 15) với cổ tay Tay Trái (điểm 0)
ALL_CONNECTIONS.append({'indices': (15, 0), 'offset': (0, 33), 'color': 'blue', 'lw': 2})


# 3. Tay Phải (Indices 54-74)
ALL_CONNECTIONS.extend([
    {'indices': (start, end), 'offset': 54, 'color': 'green', 'lw': 1.5}
    for (start, end) in HAND_CONNECTIONS
])
# Nối cổ tay (Wrist) của Thân (điểm 16) với cổ tay Tay Phải (điểm 0)
ALL_CONNECTIONS.append({'indices': (16, 0), 'offset': (0, 54), 'color': 'green', 'lw': 2})


# 4. Miệng (Indices 75-94)
ALL_CONNECTIONS.extend([
    {'indices': (start, end), 'offset': 75, 'color': 'red', 'lw': 1}
    for (start, end) in MOUTH_CONNECTIONS_20
])

# --- (FIX 2) DANH SÁCH CÁC ĐIỂM CẦN VẼ (BỎ CHÂN) ---
# Bỏ qua các index 23-32 (chân) và 25-26 (hông)
# (Thực ra 23-32 là chân/hông của Holistic)
MANUAL_UPPER_BODY_IDXS = list(range(23)) # 0-22 (Mặt + Thân trên)
LEFT_HAND_IDXS = list(range(33, 54)) # 33-53
RIGHT_HAND_IDXS = list(range(54, 75)) # 54-74
MOUTH_IDXS = list(range(75, 95)) # 75-94
PLOT_IDXS = MANUAL_UPPER_BODY_IDXS + LEFT_HAND_IDXS + RIGHT_HAND_IDXS + MOUTH_IDXS


def load_and_prepare_pose(pose_214):
    """
    Tách pose 214D thành 95 keypoints (x, y) để vẽ
    Bao gồm: 75 manual (tay, thân, mặt) + 20 mouth (từ NMMs)
   
    """
    # 1. Manual keypoints (75 kps)
    manual_150 = pose_214[:, :150]
    manual_kps = manual_150.reshape(-1, 75, 2)
    
    # 2. Mouth keypoints (20 kps)
    # Vị trí mouth_flat (40D) là từ 174
    mouth_40 = pose_214[:, 174:] #
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
        
        # Lấy min/max từ GT (chỉ 75 điểm manual)
        valid_kps = kps_gt[:, :75][kps_gt[:, :75].any(axis=2)]
        if valid_kps.shape[0] > 0:
             min_vals = np.min(valid_kps, axis=0)
             max_vals = np.max(valid_kps, axis=0)
             padding = 0.2 * (max_vals - min_vals)
             ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
             ax.set_ylim(max_vals[1] + padding[1], min_vals[1] - padding[1])
        
        # Tạo các đối tượng Line2D rỗng
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
            
        # Thêm các điểm (scatter) 
        scatter = ax.scatter([], [], s=2, c='black', alpha=0.4)
        # (FIX 2) Gán PLOT_IDXS vào scatter
        lines.append({'scatter': scatter, 'plot_indices': PLOT_IDXS})

        return lines
    
    artists1 = setup_ax(ax1, 'Ground Truth')
    artists2 = setup_ax(ax2, 'Reconstructed')
    
    fig.suptitle(f'Frame 0 / {T}')

    def update(frame):
        kps_gt_frame = kps_gt[frame]
        kps_recon_frame = kps_recon[frame]
        
        all_changed_artists = []
        
        # --- Cập nhật cho ax1 (GT) ---
        for item in artists1:
            if 'line' in item:
                idx_start = item['start']
                idx_end = item['end']
                
                # (FIX 1) Chỉ vẽ nếu cả 2 điểm > 0 (không phải (0,0))
                if np.sum(np.abs(kps_gt_frame[idx_start])) > 1e-6 and np.sum(np.abs(kps_gt_frame[idx_end])) > 1e-6:
                    item['line'].set_data(
                        [kps_gt_frame[idx_start, 0], kps_gt_frame[idx_end, 0]],
                        [kps_gt_frame[idx_start, 1], kps_gt_frame[idx_end, 1]]
                    )
                else:
                    item['line'].set_data([], []) # Ẩn đường nối
                all_changed_artists.append(item['line'])
                
            elif 'scatter' in item:
                # (FIX 2) Chỉ lấy các điểm trong PLOT_IDXS
                plot_indices = item['plot_indices']
                points_to_plot = kps_gt_frame[plot_indices]
                
                # Chỉ vẽ các điểm không phải (0,0)
                valid_points = points_to_plot[np.sum(np.abs(points_to_plot), axis=1) > 1e-6]
                item['scatter'].set_offsets(valid_points)
                all_changed_artists.append(item['scatter'])

        # --- Cập nhật cho ax2 (Recon) ---
        for item in artists2:
            if 'line' in item:
                idx_start = item['start']
                idx_end = item['end']

                # (FIX 1) Chỉ vẽ nếu cả 2 điểm > 0
                if np.sum(np.abs(kps_recon_frame[idx_start])) > 1e-6 and np.sum(np.abs(kps_recon_frame[idx_end])) > 1e-6:
                    item['line'].set_data(
                        [kps_recon_frame[idx_start, 0], kps_recon_frame[idx_end, 0]],
                        [kps_recon_frame[idx_start, 1], kps_recon_frame[idx_end, 1]]
                    )
                else:
                    item['line'].set_data([], [])
                all_changed_artists.append(item['line'])
                
            elif 'scatter' in item:
                # (FIX 2) Chỉ lấy các điểm trong PLOT_IDXS
                plot_indices = item['plot_indices']
                points_to_plot = kps_recon_frame[plot_indices]
                
                # Chỉ vẽ các điểm không phải (0,0)
                valid_points = points_to_plot[np.sum(np.abs(points_to_plot), axis=1) > 1e-6]
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
    parser = argparse.ArgumentParser(description='Trực quan hóa so sánh Pose')
    parser.add_argument('--gt_path', type=str, required=True,
                        help='Đường dẫn đến file .npy của Ground Truth')
    parser.add_argument('--recon_path', type=str, required=True,
                        help='Đường dẫn đến file .npy của Reconstructed')
    parser.add_argument('--output_video', type=str, default='pose_comparison_skeleton.mp4',
                        help='Tên file video output (mp4)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gt_path):
        print(f"Lỗi: Không tìm thấy {args.gt_path}")
    elif not os.path.exists(args.recon_path):
        print(f"Lỗi: Không tìm thấy {args.recon_path}")
    else:
        animate_poses(args.gt_path, args.recon_path, args.output_video)