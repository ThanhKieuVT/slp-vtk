import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
import pandas as pd
import argparse

# --- STYLE DEFINITION ---
COLOR_SIDE = "#156551"   # Dark Teal
COLOR_TRUNK = '#000000'  # Black
COLOR_MOUTH = '#c0392b'  # Dark Red

# Định nghĩa kết nối tay (20 đoạn cho 21 điểm)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]
# Định nghĩa kết nối mặt (Face contour cơ bản)
FACE_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)]

# Định nghĩa kết nối Môi (Nếu có dữ liệu)
MOUTH_CONNECTIONS = list(zip(range(0, 19), range(1, 20))) + [(19, 0)]

# Build Connection List
ALL_CONNECTIONS = []

# 1. Torso & Arms
ALL_CONNECTIONS.append({'indices': (11, 13), 'offset': 0, 'color': COLOR_SIDE, 'lw': 2.5})
ALL_CONNECTIONS.append({'indices': (13, 15), 'offset': 0, 'color': COLOR_SIDE, 'lw': 2.5})
ALL_CONNECTIONS.append({'indices': (12, 14), 'offset': 0, 'color': COLOR_SIDE, 'lw': 2.5})
ALL_CONNECTIONS.append({'indices': (14, 16), 'offset': 0, 'color': COLOR_SIDE, 'lw': 2.5})
# Torso Box
for pair in [(11, 12), (11, 23), (12, 24), (23, 24)]:
    ALL_CONNECTIONS.append({'indices': pair, 'offset': 0, 'color': COLOR_TRUNK, 'lw': 2})

# 2. Face
for pair in FACE_PAIRS:
    ALL_CONNECTIONS.append({'indices': pair, 'offset': 0, 'color': COLOR_TRUNK, 'lw': 1.5})

# 3. Hands 
# Left Hand (Offset 33)
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 33, 'color': COLOR_SIDE, 'lw': 1.5} for (s, e) in HAND_CONNECTIONS])
# Kết nối cổ tay trái (15) vào gốc bàn tay trái (33 -> index 0 của hand)
ALL_CONNECTIONS.append({'indices': (15, 0), 'offset': (0, 33), 'color': COLOR_SIDE, 'lw': 2})

# Right Hand (Offset 54)
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 54, 'color': COLOR_SIDE, 'lw': 1.5} for (s, e) in HAND_CONNECTIONS])
# Kết nối cổ tay phải (16) vào gốc bàn tay phải (54 -> index 0 của hand)
ALL_CONNECTIONS.append({'indices': (16, 0), 'offset': (0, 54), 'color': COLOR_SIDE, 'lw': 2})

# 4. Mouth (Offset 75) - Chỉ vẽ nếu dữ liệu đủ 95 điểm
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 75, 'color': COLOR_MOUTH, 'lw': 1.5} for (s, e) in MOUTH_CONNECTIONS])

# Joints to Plot (FIXED: Use these for scatter)
IDX_TEAL = [13, 15] + [33+i for i in range(21)] + [14, 16] + [54+i for i in range(21)]
IDX_BLACK = [0, 11, 12, 23, 24]
IDX_EYES = [2, 5]  # Correct eye indices for MediaPipe

class DataProcessor:
    def process_sequence(self, kps_seq):
        # kps_seq: [T, N, 2]
        T = kps_seq.shape[0]
        kps_clean = kps_seq.copy()
        
        # 1. Interpolation & NaN Handling
        for i in range(kps_clean.shape[1]):
            for c in range(2):
                signal = kps_clean[:, i, c]
                # Coi các điểm xấp xỉ 0 là bị mất (NaN) để Matplotlib không vẽ
                signal[np.abs(signal) < 0.001] = np.nan
                
                # Nội suy để lấp khoảng trống
                series = pd.Series(signal)
                series = series.interpolate(method='linear', limit_direction='both')
                
                kps_clean[:, i, c] = series.to_numpy()

        # 2. Wrist Gluing (RE-ENABLED)
        # Gán tọa độ gốc bàn tay (Hand Root) bằng đúng tọa độ cổ tay (Wrist)
        kps_clean[:, 33, :] = kps_clean[:, 15, :]  # Left
        kps_clean[:, 54, :] = kps_clean[:, 16, :]  # Right

        # 3. Stabilization (RE-ENABLED)
        shoulder_L = kps_clean[:, 11, :]
        shoulder_R = kps_clean[:, 12, :]
        neck_center = (shoulder_L + shoulder_R) / 2
        stabilized_kps = kps_clean - neck_center[:, np.newaxis, :]
        
        # 4. Smoothing (Savgol)
        final_kps = stabilized_kps.copy()
        window = 15
        poly = 3
        
        if T > window:
            for i in range(final_kps.shape[1]):
                for c in range(2):
                    try:
                        signal = final_kps[:, i, c]
                        mask = ~np.isnan(signal)
                        if np.sum(mask) > window:
                            # Only smooth if continuous segment is long enough
                            final_kps[mask, i, c] = savgol_filter(signal[mask], window, poly)
                    except:
                        pass
                    
        return final_kps


def animate_poses(gt_path, recon_path, output_path):
    # Auto-detect and load both .npy and .npz formats
    try:
        gt_data = np.load(gt_path, allow_pickle=True)
        recon_data = np.load(recon_path, allow_pickle=True)
    except Exception as e:
        print(f"❌ Error loading files: {e}")
        return

    # Helper to standardize to [T, N, 2]
    def to_standard_format(data):
        # 1. Extract array if dict
        if isinstance(data, dict):
            if 'keypoints' in data: kps = data['keypoints']
            elif 'pose' in data: kps = data['pose']
            else: kps = list(data.values())[0]
        elif isinstance(data, np.lib.npyio.NpzFile):
            kps = data['keypoints']
        else:
            kps = data
            
        # 2. Check dimensions
        if kps.ndim == 2:
            T, D = kps.shape
            
            if D == 214:
                manual_150 = kps[:, :150]
                manual_kps = manual_150.reshape(T, 75, 2)
                mouth_40 = kps[:, 174:]
                mouth_kps = mouth_40.reshape(T, 20, 2)
                kps = np.concatenate([manual_kps, mouth_kps], axis=1)
                
            elif D == 150:
                 kps = kps.reshape(T, 75, 2)
                 
            elif D % 2 == 0:
                 kps = kps.reshape(T, D//2, 2)
                 
        return kps

    gt_kps = to_standard_format(gt_data)
    recon_kps = to_standard_format(recon_data)
    
    print(f"Loaded GT shape: {gt_kps.shape}")
    print(f"Loaded Recon shape: {recon_kps.shape}")
    
    has_mouth = gt_kps.shape[1] >= 95
    if not has_mouth:
        print("⚠️ Data only has 75 points. Mouth visualization will be skipped.")

    processor = DataProcessor()
    gt_kps = processor.process_sequence(gt_kps)
    recon_kps = processor.process_sequence(recon_kps)
    
    T = min(len(gt_kps), len(recon_kps))
    gt_kps = gt_kps[:T]
    recon_kps = recon_kps[:T]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
    
    def setup_ax(ax, title):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
        
        lines = []
        for item in ALL_CONNECTIONS:
            if item['offset'] == 75 and gt_kps.shape[1] < 90:
                continue
                
            line = Line2D([], [], color=item['color'], lw=item['lw'], solid_capstyle='round')
            ax.add_line(line)
            lines.append(line)
            
        s_teal = ax.scatter([], [], s=15, c=COLOR_SIDE, zorder=5)
        s_black = ax.scatter([], [], s=20, c=COLOR_TRUNK, zorder=5)
        s_eyes = ax.scatter([], [], s=40, c=COLOR_TRUNK, zorder=6)
        s_mouth = ax.scatter([], [], s=10, c=COLOR_MOUTH, zorder=6)
        
        return lines, [s_teal, s_black, s_eyes, s_mouth]

    lines1, scatters1 = setup_ax(ax1, "GROUND TRUTH")
    lines2, scatters2 = setup_ax(ax2, "RECONSTRUCTED")
    
    # Auto Zoom Logic
    valid_frame_idx = 0
    for i in range(len(gt_kps)):
        if not np.isnan(gt_kps[i, 11, :]).any():
            valid_frame_idx = i
            break
            
    torso_pts = gt_kps[valid_frame_idx, [11, 12, 23, 24], :]
    if not np.isnan(torso_pts).any():
        min_xy = np.nanmin(torso_pts, axis=0)
        max_xy = np.nanmax(torso_pts, axis=0)
        ctr = (min_xy + max_xy) / 2
        h = max_xy[1] - min_xy[1]
        r = max(h * 2.5 if h > 0.05 else 0.8, 0.4)
        
        for ax in [ax1, ax2]:
            ax.set_xlim(ctr[0] - r, ctr[0] + r)
            ax.set_ylim(ctr[1] + r, ctr[1] - r)
    else:
        for ax in [ax1, ax2]:
            ax.set_xlim(-0.8, 0.8)
            ax.set_ylim(0.8, -0.8)

    def update(frame):
        def update_plot(kps, lines, scatters):
            # Lines
            line_idx = 0
            for item in ALL_CONNECTIONS:
                if item['offset'] == 75 and kps.shape[1] < 90:
                    continue
                
                s, e = item['indices']
                off = item['offset']
                if isinstance(off, tuple):
                    s, e = s+off[0], e+off[1]
                else:
                    s, e = s+off, e+off
                
                if s < kps.shape[1] and e < kps.shape[1]:
                    p1, p2 = kps[frame, s], kps[frame, e]
                    if not (np.isnan(p1).any() or np.isnan(p2).any()):
                        lines[line_idx].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                    else:
                        lines[line_idx].set_data([], [])
                line_idx += 1
            
            # FIXED: Use proper index lists
            frame_kps = kps[frame]
            valid = ~np.isnan(frame_kps).any(axis=1)
            indices = np.arange(len(frame_kps))
            
            # Teal (use IDX_TEAL)
            mask_teal = valid & np.isin(indices, IDX_TEAL)
            scatters[0].set_offsets(frame_kps[mask_teal])
            
            # Black (use IDX_BLACK)
            mask_black = valid & np.isin(indices, IDX_BLACK)
            scatters[1].set_offsets(frame_kps[mask_black])
            
            # Eyes (use IDX_EYES)
            mask_eyes = valid & np.isin(indices, IDX_EYES)
            scatters[2].set_offsets(frame_kps[mask_eyes])
            
            # Mouth (indices >= 75)
            mask_mouth = valid & (indices >= 75)
            if has_mouth:
                scatters[3].set_offsets(frame_kps[mask_mouth])
            else:
                scatters[3].set_offsets(np.zeros((0, 2)))

        update_plot(gt_kps, lines1, scatters1)
        update_plot(recon_kps, lines2, scatters2)
        fig.suptitle(f'Frame {frame} / {T}', fontsize=12)
        return lines1 + scatters1 + lines2 + scatters2

    print(f"🎬 Generating Animation ({T} frames)...")
    ani = animation.FuncAnimation(fig, update, frames=T, blit=True, interval=40)
    try:
        ani.save(output_path, writer='ffmpeg', fps=25, dpi=100)
        print(f"✅ Video saved: {output_path}")
    except Exception as e:
        print(f"⚠️ Error: {e}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', required=True)
    parser.add_argument('--recon_path', required=True)
    parser.add_argument('--output', default='comparison_viz.mp4')
    args = parser.parse_args()
    
    animate_poses(args.gt_path, args.recon_path, args.output)

if __name__ == "__main__":
    main()