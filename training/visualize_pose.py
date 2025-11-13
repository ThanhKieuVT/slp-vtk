# Tên file: visualize_pose.py
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D

# --- BẢN ĐỒ KẾT NỐI (SKELETON MAP) ---
# Giả định mới dựa trên 75-keypoints (Thân trên + Tay + Mặt)
# GIẢ ĐỊNH CHỈ SỐ (INDEX ASSUMPTION):
# 0-24: 25 điểm Mặt (Face-25)
# 25: Nose
# 26: Neck
# 27-28: Shoulders (L, R)
# 29-30: Elbows (L, R)
# 31-32: Wrists (L, R)
# 33-53: Tay Trái (21 điểm)
# 54-74: Tay Phải (21 điểm)
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

# 2. Kết nối Thân trên (8 điểm)
# (Nose, Neck, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist)
# Indices: 25   26     27         28        29       30       31       32
UPPER_BODY_CONNECTIONS = [
    (25, 26), # Nose -> Neck
    (26, 27), # Neck -> LShoulder
    (26, 28), # Neck -> RShoulder
    (27, 29), # LShoulder -> LElbow
    (28, 30), # RShoulder -> RElbow
    (29, 31), # LElbow -> LWrist
    (30, 32)  # RElbow -> RWrist
]

# 3. Kết nối Mặt (25 điểm) - Giả định OpenPose-25 Face
FACE_25_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), # Lông mày
    (7, 8), (8, 9), (9, 10), (10, 11), (11, 12), # Lông mày
    (13, 14), (14, 15), (15, 16), # Mũi
    (17, 18), (18, 19), (19, 20), (20, 21), # Mắt
    (21, 22), (22, 23), (23, 24), (24, 17)  # Mắt
]

# 4. Kết nối 20 điểm miệng (từ NMMs)
MOUTH_OUTER_LIP = list(zip(range(0, 11), range(1, 12))) + [(11, 0)]
MOUTH_INNER_LIP = list(zip(range(12, 19), range(13, 20))) + [(19, 12)]
MOUTH_CONNECTIONS_20 = MOUTH_OUTER_LIP + MOUTH_INNER_LIP

# --- TỔNG HỢP BẢN ĐỒ KẾT NỐI ---
ALL_CONNECTIONS = []

# 1. Thân trên (Indices 25-32)
ALL_CONNECTIONS.extend([
    {'indices': (start, end), 'offset': 0, 'color': 'gray', 'lw': 3}
    for (start, end) in UPPER_BODY_CONNECTIONS
])

# 2. Mặt (Indices 0-24)
ALL_CONNECTIONS.extend([
    {'indices': (start, end), 'offset': 0, 'color': 'purple', 'lw': 1}
    for (start, end) in FACE_25_CONNECTIONS
])

# 3. Tay Trái (Indices 33-53)
ALL_CONNECTIONS.extend([
    {'indices': (start, end), 'offset': 33, 'color': 'blue', 'lw': 1.5}
    for (start, end) in HAND_CONNECTIONS
])

# 4. Tay Phải (Indices 54-74)
ALL_CONNECTIONS.extend([
    {'indices': (start, end), 'offset': 54, 'color': 'green', 'lw': 1.5}
    for (start, end) in HAND_CONNECTIONS
])

# 5. Miệng (Indices 75-94)
ALL_CONNECTIONS.extend([
    {'indices': (start, end), 'offset': 75, 'color': 'red', 'lw': 1}
    for (start, end) in MOUTH_CONNECTIONS_20
])


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
    # (aus[17] + head[3] + gaze[4] = 24) -> 150 + 24 = 174
    mouth_40 = pose_214[:, 174:] #
    mouth_kps = mouth_40.reshape(-1, 20, 2)
    
    # 3. Kết hợp lại
    # [T, 95, 2] (75 điểm đầu là manual, 20 điểm sau là miệng NMM)
    all_kps = np.concatenate([manual_kps, mouth_kps], axis=1) 
    
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
        min_vals = np.min(kps_gt[:, :75].reshape(-1, 2), axis=0)
        max_vals = np.max(kps_gt[:, :75].reshape(-1, 2), axis=0)
        padding_factor = 0.2 # Tăng padding
        padding = padding_factor * (max_vals - min_vals)
        ax.set_xlim(min_vals[0] - padding[0], max_vals[0] + padding[0])
        ax.set_ylim(max_vals[1] + padding[1], min_vals[1] - padding[1])
        
        # Tạo các đối tượng Line2D rỗng
        lines = []
        for item in ALL_CONNECTIONS:
            (start, end) = item['indices']
            offset = item['offset']
            line = Line2D([], [], color=item['color'], lw=item['lw'], alpha=0.8)
            ax.add_line(line)
            lines.append({'line': line, 'start': start + offset, 'end': end + offset})
            
        # Thêm các điểm (scatter) để thấy rõ các khớp (vẽ tất cả 95 điểm)
        scatter = ax.scatter([], [], s=3, c='black', alpha=0.5)
        lines.append({'scatter': scatter, 'num_points': 95})

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
                idx_start = item['start']
                idx_end = item['end']
                item['line'].set_data(
                    [kps_gt_frame[idx_start, 0], kps_gt_frame[idx_end, 0]],
                    [kps_gt_frame[idx_start, 1], kps_gt_frame[idx_end, 1]]
                )
                all_changed_artists.append(item['line'])
            elif 'scatter' in item:
                num_points = item['num_points']
                item['scatter'].set_offsets(kps_gt_frame[:num_points])
                all_changed_artists.append(item['scatter'])

        # Cập nhật cho ax2 (Recon)
        for item in artists2:
            if 'line' in item:
                idx_start = item['start']
                idx_end = item['end']
                item['line'].set_data(
                    [kps_recon_frame[idx_start, 0], kps_recon_frame[idx_end, 0]],
                    [kps_recon_frame[idx_start, 1], kps_recon_frame[idx_end, 1]]
                )
                all_changed_artists.append(item['line'])
            elif 'scatter' in item:
                num_points = item['num_points']
                item['scatter'].set_offsets(kps_recon_frame[:num_points])
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
    parser.add.argument('--output_video', type=str, default='pose_comparison_skeleton.mp4',
                        help='Tên file video output (mp4)')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.gt_path):
        print(f"Lỗi: Không tìm thấy {args.gt_path}")
    elif not os.path.exists(args.recon_path):
        print(f"Lỗi: Không tìm thấy {args.recon_path}")
    else:
        animate_poses(args.gt_path, args.recon_path, args.output_video)