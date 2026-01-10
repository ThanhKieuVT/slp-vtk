"""
FIXED Visualization for 214D Format
[T, 214] = 150 (manual keypoints) + 64 (NMM features)
  - 0:150   → 75 points × 2 (Body + Hands)
  - 150:214 → 64 NMM (17 AU + 3 Head + 4 Gaze + 40 Mouth)
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from scipy.signal import savgol_filter
import pandas as pd
import argparse

# ===== STYLE =====
COLOR_BODY = "#156551"   # Teal for limbs
COLOR_TRUNK = '#000000'  # Black for torso
COLOR_MOUTH = '#c0392b'  # Red for mouth

# ===== MediaPipe POSE CONNECTIONS (33 landmarks) =====
POSE_CONNECTIONS = [
    # Face
    (0, 1), (1, 2), (2, 3), (3, 7),
    (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10),  # Mouth
    # Torso
    (11, 12), (11, 23), (12, 24), (23, 24),
    # Arms
    (11, 13), (13, 15), (12, 14), (14, 16),
    # Legs (if present)
    (23, 25), (25, 27), (27, 29), (29, 31),
    (24, 26), (26, 28), (28, 30), (30, 32),
    (27, 31), (28, 32)
]

# ===== HAND CONNECTIONS (21 landmarks each) =====
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (5, 9), (9, 13), (13, 17)            # Palm
]

# ===== MOUTH OUTLINE (20 points) =====
MOUTH_CONNECTIONS = [(i, (i+1)%20) for i in range(20)]


class PoseProcessor:
    """Process 214D pose data"""
    
    def extract_keypoints(self, pose_214):
        """
        Extract keypoints from 214D format
        
        Args:
            pose_214: [T, 214]
        
        Returns:
            dict with:
                'body': [T, 33, 2]
                'left_hand': [T, 21, 2]
                'right_hand': [T, 21, 2]
                'mouth': [T, 20, 2]
        """
        T = pose_214.shape[0]
        
        # Extract manual keypoints [0:150] → reshape to [T, 75, 2]
        manual = pose_214[:, :150].reshape(T, 75, 2)
        
        # Split into components (MediaPipe standard)
        body = manual[:, :33, :]        # Pose landmarks
        left_hand = manual[:, 33:54, :]  # Left hand (21 points)
        right_hand = manual[:, 54:75, :] # Right hand (21 points)
        
        # Extract mouth from NMM [174:214] → 40 dims = 20 points × 2
        mouth_flat = pose_214[:, 174:214]
        mouth = mouth_flat.reshape(T, 20, 2)
        
        return {
            'body': body,
            'left_hand': left_hand,
            'right_hand': right_hand,
            'mouth': mouth
        }
    
    def clean_and_smooth(self, keypoints_dict):
        """Clean NaN, interpolate, smooth"""
        
        def process_component(kps):
            """Process [T, N, 2] component"""
            T, N, _ = kps.shape
            kps_clean = kps.copy()
            
            # 1. Mark near-zero as NaN
            mask = np.abs(kps_clean).sum(axis=-1) < 0.001
            kps_clean[mask] = np.nan
            
            # 2. Interpolate
            for i in range(N):
                for c in range(2):
                    series = pd.Series(kps_clean[:, i, c])
                    series = series.interpolate(method='linear', limit_direction='both')
                    kps_clean[:, i, c] = series.values
            
            # 3. Smooth
            if T > 15:
                for i in range(N):
                    for c in range(2):
                        try:
                            signal = kps_clean[:, i, c]
                            mask = ~np.isnan(signal)
                            if mask.sum() > 15:
                                kps_clean[mask, i, c] = savgol_filter(
                                    signal[mask], 15, 3
                                )
                        except:
                            pass
            
            return kps_clean
        
        return {
            key: process_component(val) 
            for key, val in keypoints_dict.items()
        }
    
    def stabilize(self, keypoints_dict):
        """Center at neck (shoulders midpoint)"""
        body = keypoints_dict['body']
        
        # Get shoulder centers
        left_shoulder = body[:, 11, :]
        right_shoulder = body[:, 12, :]
        neck_center = (left_shoulder + right_shoulder) / 2  # [T, 2]
        
        # Subtract from all components
        stabilized = {}
        for key, kps in keypoints_dict.items():
            stabilized[key] = kps - neck_center[:, np.newaxis, :]
        
        return stabilized
    
    def glue_wrists(self, keypoints_dict):
        """Attach hands to wrists"""
        body = keypoints_dict['body']
        
        # Left wrist (15) → Left hand root (0)
        keypoints_dict['left_hand'][:, 0, :] = body[:, 15, :]
        
        # Right wrist (16) → Right hand root (0)
        keypoints_dict['right_hand'][:, 0, :] = body[:, 16, :]
        
        return keypoints_dict
    
    def process(self, pose_214):
        """Full pipeline"""
        kps = self.extract_keypoints(pose_214)
        kps = self.clean_and_smooth(kps)
        kps = self.stabilize(kps)
        kps = self.glue_wrists(kps)
        return kps


def load_214d_data(path):
    """Load and convert to [T, 214] format"""
    data = np.load(path, allow_pickle=True)
    
    # Extract array
    if isinstance(data, dict):
        if 'keypoints' in data: arr = data['keypoints']
        elif 'pose' in data: arr = data['pose']
        else: arr = list(data.values())[0]
    elif isinstance(data, np.lib.npyio.NpzFile):
        arr = data['keypoints']
    else:
        arr = data
    
    # Check if already [T, 214]
    if arr.ndim == 2 and arr.shape[1] == 214:
        return arr
    
    # If [T, 150] → pad with zeros for NMM
    if arr.ndim == 2 and arr.shape[1] == 150:
        T = arr.shape[0]
        padded = np.zeros((T, 214), dtype=arr.dtype)
        padded[:, :150] = arr
        return padded
    
    # If [T, 75, 2] → reshape to [T, 150] and pad
    if arr.ndim == 3 and arr.shape[1] == 75:
        T = arr.shape[0]
        manual = arr.reshape(T, 150)
        padded = np.zeros((T, 214), dtype=arr.dtype)
        padded[:, :150] = manual
        return padded
    
    raise ValueError(f"Unknown format: {arr.shape}")


def visualize_comparison(gt_path, recon_path, output_path, scale=2.0):
    """
    Main visualization function
    
    Args:
        scale: Zoom factor (default 2.0 for larger pose)
    """
    print(f"Loading data...")
    gt_214 = load_214d_data(gt_path)
    recon_214 = load_214d_data(recon_path)
    
    print(f"GT shape: {gt_214.shape}")
    print(f"Recon shape: {recon_214.shape}")
    
    # Process
    processor = PoseProcessor()
    gt_kps = processor.process(gt_214)
    recon_kps = processor.process(recon_214)
    
    T = min(len(gt_214), len(recon_214))
    
    # Setup figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 7), dpi=120)
    
    def setup_ax(ax, title):
        ax.set_title(title, fontsize=16, fontweight='bold', pad=15)
        ax.invert_yaxis()
        ax.set_aspect('equal')
        ax.axis('off')
        
        # Set view limits (scaled)
        r = 0.6 * scale  # Scale up the view
        ax.set_xlim(-r, r)
        ax.set_ylim(r, -r)
        
        lines = []
        scatters = []
        return lines, scatters
    
    lines1, scatters1 = setup_ax(ax1, "GROUND TRUTH")
    lines2, scatters2 = setup_ax(ax2, "RECONSTRUCTED")
    
    def draw_connections(ax, kps, connections, offset=np.array([0, 0]), 
                        color=COLOR_BODY, lw=2):
        """Draw skeleton connections"""
        lines = []
        for (i, j) in connections:
            if i < len(kps) and j < len(kps):
                p1, p2 = kps[i], kps[j]
                if not (np.isnan(p1).any() or np.isnan(p2).any()):
                    p1, p2 = p1 + offset, p2 + offset
                    line, = ax.plot([p1[0], p2[0]], [p1[1], p2[1]], 
                                   color=color, lw=lw, solid_capstyle='round')
                    lines.append(line)
        return lines
    
    def draw_keypoints(ax, kps, color=COLOR_BODY, size=30):
        """Draw keypoints as scatter"""
        valid = ~np.isnan(kps).any(axis=1)
        return ax.scatter(kps[valid, 0], kps[valid, 1], 
                         c=color, s=size, zorder=10)
    
    def update(frame):
        """Update frame"""
        # Clear previous
        for ax in [ax1, ax2]:
            for artist in ax.lines + ax.collections:
                artist.remove()
        
        def render(ax, kps_dict):
            """Render one frame"""
            body = kps_dict['body'][frame]
            left_hand = kps_dict['left_hand'][frame]
            right_hand = kps_dict['right_hand'][frame]
            mouth = kps_dict['mouth'][frame]
            
            # Draw body skeleton
            draw_connections(ax, body, POSE_CONNECTIONS, 
                           color=COLOR_TRUNK, lw=2.5)
            
            # Draw hands
            left_wrist = body[15]
            right_wrist = body[16]
            
            if not np.isnan(left_wrist).any():
                draw_connections(ax, left_hand, HAND_CONNECTIONS,
                               offset=np.zeros(2), color=COLOR_BODY, lw=1.5)
                # Connect wrist to hand root
                ax.plot([left_wrist[0], left_hand[0, 0]], 
                       [left_wrist[1], left_hand[0, 1]],
                       color=COLOR_BODY, lw=2, solid_capstyle='round')
            
            if not np.isnan(right_wrist).any():
                draw_connections(ax, right_hand, HAND_CONNECTIONS,
                               offset=np.zeros(2), color=COLOR_BODY, lw=1.5)
                ax.plot([right_wrist[0], right_hand[0, 0]], 
                       [right_wrist[1], right_hand[0, 1]],
                       color=COLOR_BODY, lw=2, solid_capstyle='round')
            
            # Draw mouth (relative to face)
            nose = body[0]
            if not np.isnan(nose).any() and not np.isnan(mouth).all():
                # Position mouth below nose
                mouth_offset = nose + np.array([0, 0.05])
                mouth_scaled = mouth * 0.3 + mouth_offset  # Scale down mouth
                draw_connections(ax, mouth_scaled, MOUTH_CONNECTIONS,
                               color=COLOR_MOUTH, lw=1.5)
            
            # Draw keypoints
            draw_keypoints(ax, body, COLOR_TRUNK, size=40)
            draw_keypoints(ax, left_hand, COLOR_BODY, size=20)
            draw_keypoints(ax, right_hand, COLOR_BODY, size=20)
            
            # Eyes
            if not np.isnan(body[[2, 5]]).any():
                ax.scatter(body[[2, 5], 0], body[[2, 5], 1],
                          c='black', s=60, zorder=15)
        
        render(ax1, gt_kps)
        render(ax2, recon_kps)
        
        fig.suptitle(f'Frame {frame+1}/{T}', fontsize=14, y=0.98)
        
        return ax1.lines + ax1.collections + ax2.lines + ax2.collections
    
    print(f"Rendering animation ({T} frames)...")
    ani = animation.FuncAnimation(fig, update, frames=T, 
                                 interval=40, blit=False)
    
    try:
        ani.save(output_path, writer='ffmpeg', fps=25, dpi=120,
                metadata={'title': 'Pose Comparison'})
        print(f"✅ Saved: {output_path}")
    except Exception as e:
        print(f"❌ Error saving: {e}")
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt', required=True, help='Ground truth path')
    parser.add_argument('--recon', required=True, help='Reconstructed path')
    parser.add_argument('--output', default='comparison_fixed.mp4')
    parser.add_argument('--scale', type=float, default=2.0, 
                       help='Zoom scale (default 2.0)')
    args = parser.parse_args()
    
    visualize_comparison(args.gt, args.recon, args.output, args.scale)