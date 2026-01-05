import os
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
import pandas as pd

# --- CẤU HÌNH ĐƯỜNG DẪN (Giống file extract của bạn) ---
# Folder chứa dữ liệu vừa trích xuất xong
OUTPUT_DIR = "/Users/kieuvo/Learn/Research/SL/Implement/SignFML/slp-vtk/data/RWTH/PHOENIX-2014-T-release-v3/processed_data/test_1"
SPLIT_TO_CHECK = "train" # Kiểm tra tập dev (hoặc train/test)

# --- 1. TOPOLOGY & STYLE ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]
MOUTH_CONNECTIONS = list(zip(range(0, 19), range(1, 20))) + [(19, 0)]

COLOR_L = "#156551" 
COLOR_R = '#156551'
COLOR_C = '#000000'

ALL_CONNECTIONS = []
# Body 
ALL_CONNECTIONS.append({'indices': (11, 13), 'offset': 0, 'color': COLOR_L, 'lw': 2.5})
ALL_CONNECTIONS.append({'indices': (13, 15), 'offset': 0, 'color': COLOR_L, 'lw': 2.5})
ALL_CONNECTIONS.append({'indices': (12, 14), 'offset': 0, 'color': COLOR_R, 'lw': 2.5})
ALL_CONNECTIONS.append({'indices': (14, 16), 'offset': 0, 'color': COLOR_R, 'lw': 2.5})
for pair in [(11, 12), (11, 23), (12, 24), (23, 24)]:
    ALL_CONNECTIONS.append({'indices': pair, 'offset': 0, 'color': COLOR_C, 'lw': 2})
# Face
FACE_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)]
for pair in FACE_PAIRS:
    ALL_CONNECTIONS.append({'indices': pair, 'offset': 0, 'color': COLOR_C, 'lw': 1.5})
# Hands
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 33, 'color': COLOR_L, 'lw': 1.5} for (s, e) in HAND_CONNECTIONS])
ALL_CONNECTIONS.append({'indices': (15, 0), 'offset': (0, 33), 'color': COLOR_L, 'lw': 2})
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 54, 'color': COLOR_R, 'lw': 1.5} for (s, e) in HAND_CONNECTIONS])
ALL_CONNECTIONS.append({'indices': (16, 0), 'offset': (0, 54), 'color': COLOR_R, 'lw': 2})
# Mouth
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 75, 'color': '#c0392b', 'lw': 1.5} for (s, e) in MOUTH_CONNECTIONS])

JOINTS_L = [13, 15] + [33+i for i in range(21)] 
JOINTS_R = [14, 16] + [54+i for i in range(21)] 
JOINTS_C = [0, 11, 12, 23, 24] 

class DataProcessor:
    def process(self, manual_kps, mouth_kps):
        # manual_kps: (T, 75, 2)
        # mouth_kps: (T, 20, 2)
        T = manual_kps.shape[0]
        
        # Ghép lại thành (T, 95, 2)
        kps = np.concatenate([manual_kps, mouth_kps], axis=1)
        
        # 1. INTERPOLATION (Lấp đầy frame bị xoá do confidence thấp)
        kps_clean = kps.copy()
        for i in range(kps.shape[1]): 
            for c in range(2): 
                signal = kps_clean[:, i, c]
                # Nếu giá trị = 0 (do mình lọc ở bước extract) -> Gán NaN
                signal[np.abs(signal) < 0.001] = np.nan
                
                # Nội suy
                series = pd.Series(signal)
                series = series.interpolate(method='linear', limit_direction='both')
                series = series.fillna(0)
                kps_clean[:, i, c] = series.to_numpy()

        # 2. STABILIZATION (Chống rung)
        shoulder_L = kps_clean[:, 11, :]
        shoulder_R = kps_clean[:, 12, :]
        neck_center = (shoulder_L + shoulder_R) / 2
        global_neck_mean = np.mean(neck_center, axis=0)
        stabilized_kps = kps_clean - neck_center[:, np.newaxis, :] + global_neck_mean

        # 3. SMOOTHING (Savgol)
        final_kps = stabilized_kps.copy()
        window = 7
        if T > window:
            for i in range(final_kps.shape[1]):
                for c in range(2):
                    try:
                        final_kps[:, i, c] = savgol_filter(final_kps[:, i, c], window, 2)
                    except: pass
                    
        return final_kps

def create_animation(kps, vid_id, save_path):
    T = len(kps)
    fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
    ax.set_title(f"Check: {vid_id}", fontsize=12)
    
    # Auto Zoom
    torso = kps[:, [11, 12, 23, 24], :]
    valid = torso[np.sum(np.abs(torso), axis=2) > 0.01]
    if len(valid) > 0:
        min_xy = np.min(valid, axis=0)
        max_xy = np.max(valid, axis=0)
        ctr = (min_xy + max_xy) / 2
        h = max_xy[1] - min_xy[1]
        r = h * 1.5 if h > 0.1 else 0.5
        ax.set_xlim(ctr[0] - r, ctr[0] + r)
        ax.set_ylim(ctr[1] + r, ctr[1] - r) # Đảo ngược trục Y cho đúng ảnh
    else:
        ax.set_xlim(0, 1); ax.set_ylim(1, 0)
        
    ax.axis('off')
    ax.set_aspect('equal')

    lines = [Line2D([], [], color=item['color'], lw=item['lw'], solid_capstyle='round') for item in ALL_CONNECTIONS]
    for line in lines: ax.add_line(line)
    
    scatters = [
        ax.scatter([], [], s=15, c=COLOR_L, zorder=5),
        ax.scatter([], [], s=15, c=COLOR_R, zorder=5),
        ax.scatter([], [], s=20, c=COLOR_C, zorder=5),
        ax.scatter([], [], s=30, c='black', zorder=6) # Eyes
    ]

    def update(frame):
        # Update Lines
        for line, item in zip(lines, ALL_CONNECTIONS):
            (s, e) = item['indices']
            off = item['offset']
            if isinstance(off, tuple): s, e = s+off[0], e+off[1]
            else: s, e = s+off, e+off
            
            p1, p2 = kps[frame, s], kps[frame, e]
            if np.sum(np.abs(p1))>0 and np.sum(np.abs(p2))>0:
                line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
            else:
                line.set_data([], [])

        # Update Scatters
        def get_pts(idxs):
            pts = kps[frame, idxs]
            return pts[np.sum(np.abs(pts), axis=1) > 0.01]
            
        scatters[0].set_offsets(get_pts(JOINTS_L))
        scatters[1].set_offsets(get_pts(JOINTS_R))
        scatters[2].set_offsets(get_pts(JOINTS_C))
        
        eyes = []
        if np.sum(np.abs(kps[frame, 2])) > 0: eyes.append(kps[frame, 2])
        if np.sum(np.abs(kps[frame, 5])) > 0: eyes.append(kps[frame, 5])
        if eyes: scatters[3].set_offsets(eyes)
        else: scatters[3].set_offsets(np.zeros((0, 2)))
        
        return lines + scatters

    ani = animation.FuncAnimation(fig, update, frames=T, blit=True, interval=40)
    ani.save(save_path, writer='pillow', fps=25)
    plt.close()

def main():
    pose_dir = os.path.join(OUTPUT_DIR, SPLIT_TO_CHECK, "poses")
    nmm_dir = os.path.join(OUTPUT_DIR, SPLIT_TO_CHECK, "nmms")
    
    # Tìm file .npz
    npz_files = glob.glob(os.path.join(pose_dir, "*.npz"))
    if not npz_files:
        print(f"❌ Không tìm thấy file .npz nào trong {pose_dir}")
        print("👉 Hãy kiểm tra lại đường dẫn OUTPUT_DIR hoặc chạy file extract trước.")
        return

    # Sắp xếp và lấy 5 file đầu
    npz_files.sort()
    files_to_check = npz_files[:15]
    
    processor = DataProcessor()
    vis_dir = "check_new_extraction_test"
    os.makedirs(vis_dir, exist_ok=True)
    
    print(f"🔍 Đang kiểm tra {len(files_to_check)} mẫu trong folder: {vis_dir}")

    for npz_path in files_to_check:
        vid_name = os.path.splitext(os.path.basename(npz_path))[0]
        pkl_path = os.path.join(nmm_dir, f"{vid_name}.pkl")
        
        # Load Data
        try:
            # 1. Load Pose
            with np.load(npz_path) as data:
                manual_kps = data['keypoints'] # (T, 75, 2)
            
            # 2. Load NMM (Mouth)
            if os.path.exists(pkl_path):
                with open(pkl_path, 'rb') as f:
                    nmm_data = pickle.load(f)
                    mouth_kps = nmm_data['mouth_shape'] # (T, 20, 2)
            else:
                print(f"⚠️ Thiếu file NMM cho {vid_name}, dùng dummy.")
                mouth_kps = np.zeros((len(manual_kps), 20, 2))

            # Đồng bộ độ dài (phòng khi lệch 1-2 frame)
            min_len = min(len(manual_kps), len(mouth_kps))
            manual_kps = manual_kps[:min_len]
            mouth_kps = mouth_kps[:min_len]
            
            # 3. Process & Visualize
            final_kps = processor.process(manual_kps, mouth_kps)
            
            save_path = os.path.join(vis_dir, f"{vid_name}.gif")
            create_animation(final_kps, vid_name, save_path)
            print(f"✅ Đã tạo: {save_path}")
            
        except Exception as e:
            print(f"❌ Lỗi {vid_name}: {e}")

if __name__ == "__main__":
    main()