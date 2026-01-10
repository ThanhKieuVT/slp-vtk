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
# Nếu dữ liệu chỉ có 75 điểm (body+hands), phần này sẽ được check khi vẽ để tránh lỗi
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 75, 'color': COLOR_MOUTH, 'lw': 1.5} for (s, e) in MOUTH_CONNECTIONS])

# Joints to Plot
IDX_TEAL = [13, 15] + [33+i for i in range(21)] + [14, 16] + [54+i for i in range(21)]
IDX_BLACK = [0, 11, 12, 23, 24]
IDX_EYES = [2, 5] 

class DataProcessor:
    def process_sequence(self, kps_seq):
        # kps_seq: [T, N, 2]
        T = kps_seq.shape[0]
        kps_clean = kps_seq.copy()
        
        # 1. Interpolation & NaN Handling (RESTORED FOR SMOOTHNESS)
        for i in range(kps_clean.shape[1]):
            for c in range(2):
                signal = kps_clean[:, i, c]
                # Coi các điểm xấp xỉ 0 là bị mất (NaN) để Matplotlib không vẽ
                # Tăng ngưỡng lên 0.1 để lọc noise tốt hơn (khớp với visualize_pose.py)
                signal[np.abs(signal) < 0.1] = np.nan
                
                # Nội suy để lấp khoảng trống (Linear)
                series = pd.Series(signal)
                series = series.interpolate(method='linear', limit_direction='both')
                
                kps_clean[:, i, c] = series.to_numpy()

        # 2. "Hàn" khớp cổ tay (Wrist Gluing) - DISABLED (User feedback: weird joints)
        # kps_clean[:, 33, :] = kps_clean[:, 15, :] 
        # kps_clean[:, 54, :] = kps_clean[:, 16, :]

        # 3. Stabilization (Center Neck at 0,0) - DISABLED (Match visualize_pose.py)
        # shoulder_L = kps_clean[:, 11, :]
        
        final_kps = kps_clean.copy() # Skip stabilization
        window = 15
        poly = 3
        
        if T > window:
            for i in range(final_kps.shape[1]):
                for c in range(2):
                    # Chỉ smooth nếu không có quá nhiều NaN
                    try:
                        mask = ~np.isnan(final_kps[:, i, c])
                        if np.sum(mask) > window:
                            final_kps[mask, i, c] = savgol_filter(final_kps[mask, i, c], window, poly)
                    except: pass
                    
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
            else: kps = list(data.values())[0] # Try first value
        elif isinstance(data, np.lib.npyio.NpzFile):
            kps = data['keypoints']
        else:
            kps = data # Raw array
            
        # 2. Check dimensions
        if kps.ndim == 2:
            T, D = kps.shape
            
            # Case A: 214 dim -> Specific format from 'visualize_pose.py'
            # 0-150: Body + Hands (75 points)
            # 150-174: Discarded
            # 174-214: Mouth (20 points)
            if D == 214:
                manual_150 = kps[:, :150]
                manual_kps = manual_150.reshape(T, 75, 2)
                
                mouth_40 = kps[:, 174:]
                mouth_kps = mouth_40.reshape(T, 20, 2)
                
                # Combine to 95 points
                kps = np.concatenate([manual_kps, mouth_kps], axis=1)
                
            # Case B: 150 dim (Body + Hands only) -> 75 points
            elif D == 150:
                 kps = kps.reshape(T, 75, 2)
                 
            # Case C: Other dims -> Unknown, try reshape to [T, N, 2]
            elif D % 2 == 0:
                 kps = kps.reshape(T, D//2, 2)
                 
        return kps

    gt_kps = to_standard_format(gt_data)
    recon_kps = to_standard_format(recon_data)
    
    # Debug info
    print(f"Loaded GT shape: {gt_kps.shape}")
    print(f"Loaded Recon shape: {recon_kps.shape}")

    
    # Kiểm tra xem dữ liệu có đủ 95 điểm (có môi) hay không
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
    
    # --- PLOTTING LOGIC FROM visualize_pose.py ---
    
    # Define Connections (Exact Match)
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

    ALL_CONN = []
    # Make Body lines BLACK and THICKER (was gray/2)
    ALL_CONN.extend([{'indices': (s, e), 'offset': 0, 'color': 'black', 'lw': 3} for (s, e) in POSE_CONNECTIONS_UPPER_BODY])
    # Keep Hands Blue/Green but slightly thicker
    ALL_CONN.extend([{'indices': (s, e), 'offset': 33, 'color': 'teal', 'lw': 2} for (s, e) in HAND_CONNECTIONS])
    ALL_CONN.append({'indices': (15, 0), 'offset': (0, 33), 'color': 'teal', 'lw': 2.5}) 
    ALL_CONN.extend([{'indices': (s, e), 'offset': 54, 'color': 'darkgreen', 'lw': 2} for (s, e) in HAND_CONNECTIONS])
    ALL_CONN.append({'indices': (16, 0), 'offset': (0, 54), 'color': 'darkgreen', 'lw': 2.5}) 
    # Mouth
    ALL_CONN.extend([{'indices': (s, e), 'offset': 75, 'color': 'black', 'lw': 1.5} for (s, e) in MOUTH_CONNECTIONS_20])

    def setup_ax(ax, title):
        ax.set_title(title, fontsize=16, fontweight='bold', pad=12)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_linewidth(2)
            spine.set_color('black')
        
        lines = []
        for item in ALL_CONN:
             # Skip mouth if missing
            if item['offset'] == 75 and not has_mouth:
                continue
            line = Line2D([], [], color=item['color'], lw=item['lw'], solid_capstyle='round')
            ax.add_line(line)
            lines.append(line)
            
        # Increase Scatter Sizes
        s_teal = ax.scatter([], [], s=25, c=COLOR_SIDE, zorder=5) # Hands
        s_black = ax.scatter([], [], s=35, c='black', zorder=5) # Body (Force black color)
        s_mouth = ax.scatter([], [], s=15, c=COLOR_MOUTH, zorder=6)
        
        return lines, [s_teal, s_black, s_mouth]

    lines1, scatters1 = setup_ax(ax1, "GROUND TRUTH")
    lines2, scatters2 = setup_ax(ax2, "RECONSTRUCTED")
    
    # Zoom logic (TIGHTER)
    torso_pts = gt_kps[0, [11, 12, 23, 24], :] # Use first frame
    if not np.isnan(torso_pts).any():
        min_xy = np.nanmin(torso_pts, axis=0)
        max_xy = np.nanmax(torso_pts, axis=0)
        ctr = (min_xy + max_xy) / 2
        h = max_xy[1] - min_xy[1]
        
        # TIGHTER ZOOM: 2.0x height instead of 3.5x
        raw_r = h * 2.0 if h > 0.05 else 0.6 
        r = max(raw_r, 0.35) 
        
        for ax in [ax1, ax2]:
            ax.set_xlim(ctr[0] - r, ctr[0] + r)
            ax.set_ylim(ctr[1] + r, ctr[1] - r)
    else:
        for ax in [ax1, ax2]: ax.set_xlim(-0.6, 0.6); ax.set_ylim(0.6, -0.6)

    def update(frame):
        def update_plot(kps, lines, scatters):
            # Lines
            line_idx = 0
            for item in ALL_CONN:
                if item['offset'] == 75 and kps.shape[1] < 90:
                    continue
                
                s, e = item['indices']
                off = item['offset']
                if isinstance(off, tuple): s, e = s+off[0], e+off[1]
                else: s, e = s+off, e+off
                
                if s < kps.shape[1] and e < kps.shape[1]:
                    p1, p2 = kps[frame, s], kps[frame, e]
                    if not (np.isnan(p1).any() or np.isnan(p2).any()):
                        lines[line_idx].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                    else:
                         lines[line_idx].set_data([], [])
                line_idx += 1
            
            # Scatters
            frame_kps = kps[frame]
            valid = ~np.isnan(frame_kps).any(axis=1)
            indices = np.arange(len(frame_kps))
            
            # Hands: 33-75
            mask_hands = valid & (indices >= 33) & (indices < 75)
            scatters[0].set_offsets(frame_kps[mask_hands])
            
            # Body: 0-23
            mask_body = valid & (indices < 23)
            scatters[1].set_offsets(frame_kps[mask_body])
            
            # Mouth: >= 75
            if has_mouth:
                mask_mouth = valid & (indices >= 75)
                scatters[2].set_offsets(frame_kps[mask_mouth])

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