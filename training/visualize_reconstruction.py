"""
Simple 214D Visualization - Matching Original Video Style
Keep it simple like the original working version
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
import pandas as pd
import argparse

# Style from original
COLOR_SIDE = "#156551"   # Dark Teal
COLOR_TRUNK = '#000000'  # Black
COLOR_MOUTH = '#c0392b'  # Dark Red

# Hand connections (21 landmarks)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

# Face connections (simple)
FACE_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)]

# Build connection list
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

# 3. Left Hand (offset 33)
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 33, 'color': COLOR_SIDE, 'lw': 1.5} for (s, e) in HAND_CONNECTIONS])
ALL_CONNECTIONS.append({'indices': (15, 0), 'offset': (0, 33), 'color': COLOR_SIDE, 'lw': 2})

# 4. Right Hand (offset 54)
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 54, 'color': COLOR_SIDE, 'lw': 1.5} for (s, e) in HAND_CONNECTIONS])
ALL_CONNECTIONS.append({'indices': (16, 0), 'offset': (0, 54), 'color': COLOR_SIDE, 'lw': 2})

# Keypoint groups
IDX_TEAL = [13, 15] + list(range(33, 54)) + [14, 16] + list(range(54, 75))
IDX_BLACK = [0, 11, 12, 23, 24]
IDX_EYES = [2, 5]


class DataProcessor:
    def process_sequence(self, pose_214):
        """
        Process 214D pose data
        Args:
            pose_214: [T, 214] where:
                - 0:150 = manual keypoints (75 points × 2)
                - 150:214 = NMM features (not used for visualization)
        """
        # Extract only manual keypoints [T, 150] → [T, 75, 2]
        T = pose_214.shape[0]
        kps = pose_214[:, :150].reshape(T, 75, 2)
        
        # 1. Clean NaN
        for i in range(75):
            for c in range(2):
                signal = kps[:, i, c]
                signal[np.abs(signal) < 0.001] = np.nan
                
                series = pd.Series(signal)
                series = series.interpolate(method='linear', limit_direction='both')
                kps[:, i, c] = series.to_numpy()
        
        # 2. Glue wrists to hands
        kps[:, 33, :] = kps[:, 15, :]  # Left hand root = left wrist
        kps[:, 54, :] = kps[:, 16, :]  # Right hand root = right wrist
        
        # 3. Stabilize at neck center
        shoulder_L = kps[:, 11, :]
        shoulder_R = kps[:, 12, :]
        neck_center = (shoulder_L + shoulder_R) / 2
        kps = kps - neck_center[:, np.newaxis, :]
        
        # 4. Smooth
        window = 15
        poly = 3
        if T > window:
            for i in range(75):
                for c in range(2):
                    try:
                        mask = ~np.isnan(kps[:, i, c])
                        if np.sum(mask) > window:
                            kps[mask, i, c] = savgol_filter(kps[mask, i, c], window, poly)
                    except:
                        pass
        
        return kps


def load_data(path):
    """Load and extract 214D data"""
    data = np.load(path, allow_pickle=True)
    
    # Extract array
    if isinstance(data, dict):
        if 'keypoints' in data:
            arr = data['keypoints']
        elif 'pose' in data:
            arr = data['pose']
        else:
            arr = list(data.values())[0]
    elif isinstance(data, np.lib.npyio.NpzFile):
        arr = data['keypoints']
    else:
        arr = data
    
    # Convert to [T, 214]
    if arr.ndim == 2 and arr.shape[1] == 214:
        return arr
    elif arr.ndim == 2 and arr.shape[1] == 150:
        # Pad with zeros for NMM part
        T = arr.shape[0]
        padded = np.zeros((T, 214), dtype=arr.dtype)
        padded[:, :150] = arr
        return padded
    elif arr.ndim == 3 and arr.shape[1] == 75:
        # Reshape and pad
        T = arr.shape[0]
        manual = arr.reshape(T, 150)
        padded = np.zeros((T, 214), dtype=arr.dtype)
        padded[:, :150] = manual
        return padded
    
    raise ValueError(f"Unsupported shape: {arr.shape}")


def animate_poses(gt_path, recon_path, output_path):
    print("Loading data...")
    gt_214 = load_data(gt_path)
    recon_214 = load_data(recon_path)
    
    print(f"GT: {gt_214.shape}, Recon: {recon_214.shape}")
    
    processor = DataProcessor()
    gt_kps = processor.process_sequence(gt_214)
    recon_kps = processor.process_sequence(recon_214)
    
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
            line = Line2D([], [], color=item['color'], lw=item['lw'], solid_capstyle='round')
            ax.add_line(line)
            lines.append(line)
        
        s_teal = ax.scatter([], [], s=15, c=COLOR_SIDE, zorder=5)
        s_black = ax.scatter([], [], s=20, c=COLOR_TRUNK, zorder=5)
        s_eyes = ax.scatter([], [], s=40, c=COLOR_TRUNK, zorder=6)
        
        return lines, [s_teal, s_black, s_eyes]
    
    lines1, scatters1 = setup_ax(ax1, "GROUND TRUTH")
    lines2, scatters2 = setup_ax(ax2, "RECONSTRUCTED")
    
    # Fixed zoom - MUCH LARGER to fill frame
    # After stabilization at neck, pose is centered at (0,0)
    # Use fixed view that fills the frame nicely
    r = 0.35  # Smaller radius = BIGGER pose
    
    for ax in [ax1, ax2]:
        ax.set_xlim(-r, r)
        ax.set_ylim(r, -r)  # Inverted Y
    
    def update(frame):
        def update_plot(kps, lines, scatters):
            # Draw lines
            line_idx = 0
            for item in ALL_CONNECTIONS:
                s, e = item['indices']
                off = item['offset']
                
                if isinstance(off, tuple):
                    s_idx, e_idx = s + off[0], e + off[1]
                else:
                    s_idx, e_idx = s + off, e + off
                
                if s_idx < 75 and e_idx < 75:
                    p1, p2 = kps[frame, s_idx], kps[frame, e_idx]
                    if not (np.isnan(p1).any() or np.isnan(p2).any()):
                        lines[line_idx].set_data([p1[0], p2[0]], [p1[1], p2[1]])
                    else:
                        lines[line_idx].set_data([], [])
                line_idx += 1
            
            # Draw keypoints
            frame_kps = kps[frame]
            valid = ~np.isnan(frame_kps).any(axis=1)
            indices = np.arange(75)
            
            # Teal (hands + arms)
            mask_teal = valid & np.isin(indices, IDX_TEAL)
            scatters[0].set_offsets(frame_kps[mask_teal])
            
            # Black (body)
            mask_black = valid & np.isin(indices, IDX_BLACK)
            scatters[1].set_offsets(frame_kps[mask_black])
            
            # Eyes
            mask_eyes = valid & np.isin(indices, IDX_EYES)
            scatters[2].set_offsets(frame_kps[mask_eyes])
        
        update_plot(gt_kps, lines1, scatters1)
        update_plot(recon_kps, lines2, scatters2)
        fig.suptitle(f'Frame {frame} / {T}', fontsize=12)
        return lines1 + scatters1 + lines2 + scatters2
    
    print(f"Rendering {T} frames...")
    ani = animation.FuncAnimation(fig, update, frames=T, blit=True, interval=40)
    
    try:
        ani.save(output_path, writer='ffmpeg', fps=25, dpi=100)
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True)
    parser.add_argument('--recon', required=True)
    parser.add_argument('--output', default='comparison.mp4')
    args = parser.parse_args()
    
    animate_poses(args.gt, args.recon, args.output)