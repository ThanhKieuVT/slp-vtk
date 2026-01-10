"""
COMPLETE 214D Visualization with LEGS + PROPER MOUTH
[T, 214] = 150 (manual) + 64 (NMM: 17 AU + 3 Head + 4 Gaze + 40 Mouth)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
import pandas as pd
import argparse

# ===== COLORS =====
COLOR_BODY = "#156551"   # Teal
COLOR_TRUNK = '#000000'  # Black
COLOR_MOUTH = '#c0392b'  # Red

# ===== MediaPipe POSE (33 landmarks) =====
POSE_CONNECTIONS = [
    # Face contour
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10),
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Arms
    (11, 13), (13, 15), (12, 14), (14, 16),
    # Legs - CRITICAL!
    (23, 25), (25, 27), (27, 29), (29, 31),  # Left leg
    (24, 26), (26, 28), (28, 30), (30, 32),  # Right leg
]

# ===== HAND (21 landmarks) =====
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (5, 9), (9, 13), (13, 17), (0, 5), (5, 9), (9, 13), (13, 17)
]

# ===== MOUTH (20 points) =====
MOUTH_CONNECTIONS = [(i, (i+1)%20) for i in range(20)]


class PoseProcessor:
    def extract_keypoints(self, pose_214):
        """Extract from 214D"""
        T = pose_214.shape[0]
        manual = pose_214[:, :150].reshape(T, 75, 2)
        
        body = manual[:, :33, :]
        left_hand = manual[:, 33:54, :]
        right_hand = manual[:, 54:75, :]
        
        # FIXED: Extract mouth from correct location
        mouth_data = pose_214[:, 174:214]
        mouth = mouth_data.reshape(T, 20, 2)
        
        return {
            'body': body,
            'left_hand': left_hand,
            'right_hand': right_hand,
            'mouth': mouth
        }
    
    def clean_and_smooth(self, keypoints_dict):
        def process_component(kps):
            T, N, _ = kps.shape
            kps_clean = kps.copy()
            
            # Mark near-zero as NaN
            mask = (np.abs(kps_clean).sum(axis=-1) < 0.001)
            kps_clean[mask] = np.nan
            
            # Interpolate
            for i in range(N):
                for c in range(2):
                    series = pd.Series(kps_clean[:, i, c])
                    series = series.interpolate(method='linear', limit_direction='both')
                    kps_clean[:, i, c] = series.values
            
            # Smooth
            if T > 15:
                for i in range(N):
                    for c in range(2):
                        try:
                            signal = kps_clean[:, i, c]
                            valid_mask = ~np.isnan(signal)
                            if valid_mask.sum() > 15:
                                kps_clean[valid_mask, i, c] = savgol_filter(
                                    signal[valid_mask], 15, 3
                                )
                        except:
                            pass
            
            return kps_clean
        
        return {k: process_component(v) for k, v in keypoints_dict.items()}
    
    def stabilize(self, keypoints_dict):
        """Center at mid-hip instead of neck for better stability"""
        body = keypoints_dict['body']
        
        # Use hip center (23, 24) for more stable anchor
        left_hip = body[:, 23, :]
        right_hip = body[:, 24, :]
        hip_center = (left_hip + right_hip) / 2
        
        # Subtract from all
        stabilized = {}
        for key, kps in keypoints_dict.items():
            stabilized[key] = kps - hip_center[:, np.newaxis, :]
        
        return stabilized
    
    def glue_wrists(self, keypoints_dict):
        """Attach hands to wrists"""
        body = keypoints_dict['body']
        keypoints_dict['left_hand'][:, 0, :] = body[:, 15, :]
        keypoints_dict['right_hand'][:, 0, :] = body[:, 16, :]
        return keypoints_dict
    
    def process(self, pose_214):
        kps = self.extract_keypoints(pose_214)
        kps = self.clean_and_smooth(kps)
        kps = self.stabilize(kps)
        kps = self.glue_wrists(kps)
        return kps


def load_214d_data(path):
    data = np.load(path, allow_pickle=True)
    
    if isinstance(data, dict):
        arr = data.get('keypoints', data.get('pose', list(data.values())[0]))
    elif isinstance(data, np.lib.npyio.NpzFile):
        arr = data['keypoints']
    else:
        arr = data
    
    # Handle different formats
    if arr.ndim == 2 and arr.shape[1] == 214:
        return arr
    elif arr.ndim == 2 and arr.shape[1] == 150:
        T = arr.shape[0]
        padded = np.zeros((T, 214))
        padded[:, :150] = arr
        return padded
    elif arr.ndim == 3 and arr.shape[1] == 75:
        T = arr.shape[0]
        manual = arr.reshape(T, 150)
        padded = np.zeros((T, 214))
        padded[:, :150] = manual
        return padded
    
    raise ValueError(f"Unknown shape: {arr.shape}")


def visualize_comparison(gt_path, recon_path, output_path, scale=3.0):
    print("Loading...")
    gt_214 = load_214d_data(gt_path)
    recon_214 = load_214d_data(recon_path)
    
    print(f"GT: {gt_214.shape}, Recon: {recon_214.shape}")
    
    processor = PoseProcessor()
    gt_kps = processor.process(gt_214)
    recon_kps = processor.process(recon_214)
    
    T = min(len(gt_214), len(recon_214))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 9), dpi=120)
    
    def setup_ax(ax, title):
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Larger view to include legs
        r = 0.8 * scale
        ax.set_xlim(-r, r)
        ax.set_ylim(r, -r)
        
        return [], []
    
    setup_ax(ax1, "GROUND TRUTH")
    setup_ax(ax2, "RECONSTRUCTED")
    
    def draw_skeleton(ax, body, color=COLOR_TRUNK, lw=2.5):
        """Draw body skeleton"""
        for (i, j) in POSE_CONNECTIONS:
            if i < len(body) and j < len(body):
                p1, p2 = body[i], body[j]
                if not (np.isnan(p1).any() or np.isnan(p2).any()):
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color=color, lw=lw, solid_capstyle='round', zorder=5)
    
    def draw_hand(ax, hand, wrist, color=COLOR_BODY, lw=1.8):
        """Draw hand skeleton"""
        if np.isnan(wrist).any():
            return
        
        for (i, j) in HAND_CONNECTIONS:
            if i < len(hand) and j < len(hand):
                p1, p2 = hand[i], hand[j]
                if not (np.isnan(p1).any() or np.isnan(p2).any()):
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color=color, lw=lw, solid_capstyle='round', zorder=6)
        
        # Connect wrist to hand root
        if not np.isnan(hand[0]).any():
            ax.plot([wrist[0], hand[0, 0]], [wrist[1], hand[0, 1]],
                   color=color, lw=2.0, solid_capstyle='round', zorder=6)
    
    def draw_mouth(ax, mouth, nose):
        """Draw mouth relative to nose"""
        if np.isnan(nose).any() or np.isnan(mouth).all():
            return
        
        # FIXED: Position mouth correctly relative to face
        # Mouth should be below nose
        mouth_center_y = nose[1] + 0.08  # Offset down from nose
        mouth_scaled = mouth * 0.15  # Scale down
        mouth_scaled[:, 1] += mouth_center_y  # Move to correct Y
        mouth_scaled[:, 0] += nose[0]  # Center on nose X
        
        for (i, j) in MOUTH_CONNECTIONS:
            if i < len(mouth_scaled) and j < len(mouth_scaled):
                p1, p2 = mouth_scaled[i], mouth_scaled[j]
                if not (np.isnan(p1).any() or np.isnan(p2).any()):
                    ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                           color=COLOR_MOUTH, lw=1.8, solid_capstyle='round', zorder=7)
    
    def draw_keypoints(ax, pts, color, size, zorder=10):
        """Draw keypoints"""
        valid = ~np.isnan(pts).any(axis=1)
        if valid.any():
            ax.scatter(pts[valid, 0], pts[valid, 1], 
                      c=color, s=size, zorder=zorder, edgecolors='white', linewidths=0.5)
    
    def render_frame(ax, kps_dict):
        """Render one frame"""
        body = kps_dict['body']
        left_hand = kps_dict['left_hand']
        right_hand = kps_dict['right_hand']
        mouth = kps_dict['mouth']
        
        # Draw skeleton
        draw_skeleton(ax, body)
        
        # Draw hands
        draw_hand(ax, left_hand, body[15])
        draw_hand(ax, right_hand, body[16])
        
        # Draw mouth
        draw_mouth(ax, mouth, body[0])
        
        # Draw keypoints
        draw_keypoints(ax, body, COLOR_TRUNK, size=50, zorder=10)
        draw_keypoints(ax, left_hand, COLOR_BODY, size=25, zorder=11)
        draw_keypoints(ax, right_hand, COLOR_BODY, size=25, zorder=11)
        
        # Highlight eyes
        eyes = body[[2, 5]]
        if not np.isnan(eyes).any():
            ax.scatter(eyes[:, 0], eyes[:, 1], c='black', s=80, zorder=15)
    
    def update(frame):
        # Clear
        for ax in [ax1, ax2]:
            for artist in ax.lines + ax.collections:
                artist.remove()
        
        # Render
        render_frame(ax1, {k: v[frame] for k, v in gt_kps.items()})
        render_frame(ax2, {k: v[frame] for k, v in recon_kps.items()})
        
        fig.suptitle(f'Frame {frame+1}/{T}', fontsize=16, y=0.96)
        
        return []
    
    print(f"Rendering {T} frames...")
    ani = animation.FuncAnimation(fig, update, frames=T, interval=40, blit=False)
    
    try:
        ani.save(output_path, writer='ffmpeg', fps=25, dpi=120)
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error: {e}")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True)
    parser.add_argument('--recon', required=True)
    parser.add_argument('--output', default='comparison_full.mp4')
    parser.add_argument('--scale', type=float, default=3.0)
    args = parser.parse_args()
    
    visualize_comparison(args.gt, args.recon, args.output, args.scale)