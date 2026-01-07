import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
import pandas as pd
import argparse
import os

# --- STYLE DEFINITION (Teal/Black/Red logic from check_extraction_result.py) ---
COLOR_SIDE = "#156551"   # Dark Teal
COLOR_TRUNK = '#000000'  # Black
COLOR_MOUTH = '#c0392b'  # Dark Red

# Topology Indices (95-point format)
# 0-10: Face
# 11-12-23-24: Torso Box
# 11-13-15: Left Arm, 12-14-16: Right Arm
# 33-53: Left Hand, 54-74: Right Hand
# 75-94: Mouth

HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]
MOUTH_CONNECTIONS = list(zip(range(0, 19), range(1, 20))) + [(19, 0)]
FACE_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)]

# Build Connection List for Matplotlib
ALL_CONNECTIONS = []

# 1. Torso & Arms (Teal Sides, Black Trunk)
# Left Arm
ALL_CONNECTIONS.append({'indices': (11, 13), 'offset': 0, 'color': COLOR_SIDE, 'lw': 2.5})
ALL_CONNECTIONS.append({'indices': (13, 15), 'offset': 0, 'color': COLOR_SIDE, 'lw': 2.5})
# Right Arm
ALL_CONNECTIONS.append({'indices': (12, 14), 'offset': 0, 'color': COLOR_SIDE, 'lw': 2.5})
ALL_CONNECTIONS.append({'indices': (14, 16), 'offset': 0, 'color': COLOR_SIDE, 'lw': 2.5})
# Torso Box (Black)
for pair in [(11, 12), (11, 23), (12, 24), (23, 24)]:
    ALL_CONNECTIONS.append({'indices': pair, 'offset': 0, 'color': COLOR_TRUNK, 'lw': 2})

# 2. Face (Black)
for pair in FACE_PAIRS:
    ALL_CONNECTIONS.append({'indices': pair, 'offset': 0, 'color': COLOR_TRUNK, 'lw': 1.5})

# 3. Hands (Teal)
# Left Hand
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 33, 'color': COLOR_SIDE, 'lw': 1.5} for (s, e) in HAND_CONNECTIONS])
# Connect L_Wrist(15) -> L_Hand_Root(33)
ALL_CONNECTIONS.append({'indices': (15, 0), 'offset': (0, 33), 'color': COLOR_SIDE, 'lw': 2})

# Right Hand
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 54, 'color': COLOR_SIDE, 'lw': 1.5} for (s, e) in HAND_CONNECTIONS])
# Connect R_Wrist(16) -> R_Hand_Root(54)
ALL_CONNECTIONS.append({'indices': (16, 0), 'offset': (0, 54), 'color': COLOR_SIDE, 'lw': 2})

# 4. Mouth (Red)
ALL_CONNECTIONS.extend([{'indices': (s, e), 'offset': 75, 'color': COLOR_MOUTH, 'lw': 1.5} for (s, e) in MOUTH_CONNECTIONS])

# Joints to Plot (Scatter)
IDX_TEAL = [13, 15] + [33+i for i in range(21)] + [14, 16] + [54+i for i in range(21)]
IDX_BLACK = [0, 11, 12, 23, 24]
# Eyes: Based on check_extraction_result.py, we use indices 2 and 5 (Face)
IDX_EYES = [2, 5] 


class DataProcessor:
    """ Implements Interpolation, Stabilization, and Smoothing from check_extraction_result.py """
    def process(self, pose_214):
        # Convert 214 -> 95x2
        manual_150 = pose_214[:150]
        manual_kps = manual_150.reshape(75, 2)
        mouth_40 = pose_214[174:214]
        mouth_kps = mouth_40.reshape(20, 2)
        kps = np.concatenate([manual_kps, mouth_kps], axis=0) # [95, 2]
        return kps

    def process_sequence(self, kps_seq):
        # kps_seq: [T, 95, 2]
        T = kps_seq.shape[0]
        kps_clean = kps_seq.copy()
        
        # 1. Interpolation (Fill NaNs/Zeros)
        for i in range(kps_clean.shape[1]):
            for c in range(2):
                signal = kps_clean[:, i, c]
                # Treat near-zero as missing
                signal[np.abs(signal) < 0.001] = np.nan
                series = pd.Series(signal)
                series = series.interpolate(method='linear', limit_direction='both')
                series = series.fillna(0)
                kps_clean[:, i, c] = series.to_numpy()
                
        # 2. Stabilization (Center Neck)
        # Neck is average of Shoulders (11, 12)
        shoulder_L = kps_clean[:, 11, :]
        shoulder_R = kps_clean[:, 12, :]
        neck_center = (shoulder_L + shoulder_R) / 2 # [T, 2]
        global_neck_mean = np.mean(neck_center, axis=0)
        
        # Shift all points so neck is stable at global mean
        stabilized_kps = kps_clean - neck_center[:, np.newaxis, :] + global_neck_mean
        
        # 3. Smoothing (Savgol)
        final_kps = stabilized_kps.copy()
        window = 7
        if T > window:
            for i in range(final_kps.shape[1]):
                for c in range(2):
                    try:
                        final_kps[:, i, c] = savgol_filter(final_kps[:, i, c], window, 2)
                    except: pass
                    
        return final_kps


def animate_poses(gt_path, recon_path, output_path):
    # Load
    gt_raw = np.load(gt_path)      # [T, 214]
    recon_raw = np.load(recon_path) # [T, 214]
    
    if gt_raw.ndim == 1: gt_raw = gt_raw[None, :]
    if recon_raw.ndim == 1: recon_raw = recon_raw[None, :]
    
    # Process
    processor = DataProcessor()
    
    # Convert to 95pts
    gt_kps = np.array([processor.process(f) for f in gt_raw])
    recon_kps = np.array([processor.process(f) for f in recon_raw])
    
    # Apply Smoothing/Stabilization
    gt_kps = processor.process_sequence(gt_kps)
    recon_kps = processor.process_sequence(recon_kps)
    
    # Sync length
    T = min(len(gt_kps), len(recon_kps))
    gt_kps = gt_kps[:T]
    recon_kps = recon_kps[:T]
    
    # Setup Figure (Matplotlib)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), dpi=120)
    # fig.patch.set_facecolor('white')
    
    # Prepare Axes
    def setup_ax(ax, title):
        ax.set_title(title, fontsize=14, fontweight='bold', pad=10)
        ax.invert_yaxis() # Image coord system
        ax.set_aspect('equal')
        # ax.axis('off') # HIDDEN: User wants the box!
        
        # Show box, hide ticks
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(1)
        
        # Setup Lines
        lines = []
        for item in ALL_CONNECTIONS:
            line = Line2D([], [], color=item['color'], lw=item['lw'], solid_capstyle='round')
            ax.add_line(line)
            lines.append(line)
            
        # Setup Scatters
        # 0: Teal Small, 1: Black Med, 2: Eyes Large
        s_teal = ax.scatter([], [], s=15, c=COLOR_SIDE, zorder=5)
        s_black = ax.scatter([], [], s=20, c=COLOR_TRUNK, zorder=5)
        s_eyes = ax.scatter([], [], s=40, c=COLOR_TRUNK, zorder=6)
        
        return lines, [s_teal, s_black, s_eyes]

    lines1, scatters1 = setup_ax(ax1, "GROUND TRUTH")
    lines2, scatters2 = setup_ax(ax2, "RECONSTRUCTED")
    
    # Global Zoom (Based on GT Torso)
    # Use first few frames to determine zoom to match checking script logic
    # Or use average of entire sequence
    torso_pts = gt_kps[:, [11, 12, 23, 24], :]
    valid_torso = torso_pts[np.sum(np.abs(torso_pts), axis=2) > 0.01]
    if len(valid_torso) > 0:
        min_xy = np.min(valid_torso, axis=0)
        max_xy = np.max(valid_torso, axis=0)
        ctr = (min_xy + max_xy) / 2
        h = max_xy[1] - min_xy[1]
        r = h * 1.5 if h > 0.1 else 0.5
        
        # Apply to both
        for ax in [ax1, ax2]:
            ax.set_xlim(ctr[0] - r, ctr[0] + r)
            ax.set_ylim(ctr[1] + r, ctr[1] - r) # Y inverted
    else:
        # Fallback
        for ax in [ax1, ax2]:
            ax.set_xlim(-1, 1); ax.set_ylim(1, -1)

    def update(frame):
        # Helper to update one subplot
        def update_plot(kps, lines, scatters):
            # Lines
            for line, item in zip(lines, ALL_CONNECTIONS):
                s, e = item['indices']
                off = item['offset']
                if isinstance(off, tuple): s, e = s+off[0], e+off[1]
                else: s, e = s+off, e+off
                
                if s >= len(kps) or e >= len(kps): continue
                
                p1, p2 = kps[s], kps[e]
                # Simple valid check
                if np.sum(np.abs(p1)) > 0.01 and np.sum(np.abs(p2)) > 0.01:
                    line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                else:
                    line.set_data([], [])
            
            # Scatters
            # Teal
            pts_teal = kps[IDX_TEAL]
            scatters[0].set_offsets(pts_teal[np.sum(np.abs(pts_teal), axis=1) > 0.01])
            # Black
            pts_black = kps[IDX_BLACK]
            scatters[1].set_offsets(pts_black[np.sum(np.abs(pts_black), axis=1) > 0.01])
            # Eyes
            pts_eyes = kps[IDX_EYES]
            scatters[2].set_offsets(pts_eyes[np.sum(np.abs(pts_eyes), axis=1) > 0.01])
            
            return lines + scatters

        artists = []
        artists += update_plot(gt_kps[frame], lines1, scatters1)
        artists += update_plot(recon_kps[frame], lines2, scatters2)
        
        fig.suptitle(f'Frame {frame} / {T}', fontsize=12)
        return artists

    print(f"🎬 Generating Animation ({T} frames)...")
    ani = animation.FuncAnimation(fig, update, frames=T, blit=True, interval=40)
    
    # Save
    # Try using ffmpeg, fall back to pillow if needed, but MP4 requested
    try:
        ani.save(output_path, writer='ffmpeg', fps=25, dpi=100)
    except Exception as e:
        print(f"⚠️ FFmpeg error: {e}. Trying Pillow (GIF)...")
        gif_path = output_path.replace('.mp4', '.gif')
        ani.save(gif_path, writer='pillow', fps=25)
        print(f"Saved as GIF instead: {gif_path}")
        return

    print(f"✅ Video saved: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', required=True)
    parser.add_argument('--recon_path', required=True)
    parser.add_argument('--output', default='comparison_viz.mp4')
    args = parser.parse_args()
    
    animate_poses(args.gt_path, args.recon_path, args.output)

if __name__ == "__main__":
    main()