import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm

# --- 1. TOPOLOGY & STYLE (MATCHING check_extraction_result.py) ---
# Colors (Hex -> BGR)
# COLOR_L = "#156551" (RGB: 21, 101, 81) -> BGR: (81, 101, 21)
# COLOR_R = '#156551' -> Same
# COLOR_C = '#000000' -> (0, 0, 0)
# Mouth = '#c0392b' (RGB: 192, 57, 43) -> BGR: (43, 57, 192)

C_SIDE = (81, 101, 21)   # Dark Teal
C_TRUNK = (0, 0, 0)      # Black
C_MOUTH = (43, 57, 192)  # Dark Red

DRAW_RULES = []

# --- Body (Torso & Arms) ---
# Left Arm (11->13->15)
DRAW_RULES.append({'idx': (11, 13), 'off': 0, 'color': C_SIDE, 'lw': 3})
DRAW_RULES.append({'idx': (13, 15), 'off': 0, 'color': C_SIDE, 'lw': 3})
# Right Arm (12->14->16)
DRAW_RULES.append({'idx': (12, 14), 'off': 0, 'color': C_SIDE, 'lw': 3})
DRAW_RULES.append({'idx': (14, 16), 'off': 0, 'color': C_SIDE, 'lw': 3})

# Torso Box (11-12-24-23-11)
for pair in [(11, 12), (11, 23), (12, 24), (23, 24)]:
    DRAW_RULES.append({'idx': pair, 'off': 0, 'color': C_TRUNK, 'lw': 2})

# Face (0..10)
FACE_PAIRS = [(0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), (9, 10)]
for pair in FACE_PAIRS:
    DRAW_RULES.append({'idx': pair, 'off': 0, 'color': C_TRUNK, 'lw': 2})

# --- Hands ---
HAND_CONNECTIONS = [
     (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
     (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
     (0, 17), (17, 18), (18, 19), (19, 20)
]
# Left Hand (33+)
DRAW_RULES.extend([{'idx': (s, e), 'off': 33, 'color': C_SIDE, 'lw': 2} for (s, e) in HAND_CONNECTIONS])
# Connect Body L_Wrist(15) -> L_Hand_Root(0+33)
DRAW_RULES.append({'idx': (15, 0), 'off': (0, 33), 'color': C_SIDE, 'lw': 3})

# Right Hand (54+)
DRAW_RULES.extend([{'idx': (s, e), 'off': 54, 'color': C_SIDE, 'lw': 2} for (s, e) in HAND_CONNECTIONS])
# Connect Body R_Wrist(16) -> R_Hand_Root(0+54)
DRAW_RULES.append({'idx': (16, 0), 'off': (0, 54), 'color': C_SIDE, 'lw': 3})

# --- Mouth ---
MOUTH_CONNECTIONS = list(zip(range(0, 19), range(1, 20))) + [(19, 0)]
DRAW_RULES.extend([{'idx': (s, e), 'off': 75, 'color': C_MOUTH, 'lw': 2} for (s, e) in MOUTH_CONNECTIONS])

# Joints to scatter (Dots)
JOINTS_SIDE = [13, 15] + [33+i for i in range(21)] + [14, 16] + [54+i for i in range(21)]
JOINTS_TRUNK = [0, 11, 12, 23, 24] # Head + Shoulders + Hips

def load_and_prepare_pose_95(pose_214):
    manual_150 = pose_214[:150]
    manual_kps = manual_150.reshape(75, 2)
    mouth_40 = pose_214[174:214]
    mouth_kps = mouth_40.reshape(20, 2)
    return np.concatenate([manual_kps, mouth_kps], axis=0)

def get_torso_zoom_params(kps_seq):
    """
    Smart Zoom focusing on Torso (indices 11, 12, 23, 24) to keep frame stable.
    """
    # Torso indices in 95-point array: 11, 12, 23, 24
    torso_indices = [11, 12, 23, 24]
    
    # Collect valid torso points across sequence
    # Shape: [T, 4, 2]
    torso_pts = kps_seq[:, torso_indices, :]
    
    # Filter valid
    valid = torso_pts[np.sum(np.abs(torso_pts), axis=2) > 0.01]
    
    if len(valid) == 0:
        # Fallback to all points
        valid = kps_seq[np.sum(np.abs(kps_seq), axis=2) > 0.01]
        if len(valid) == 0: return 0.5, 0.5, 0.5 # Default
    
    min_xy = np.min(valid, axis=0)
    max_xy = np.max(valid, axis=0)
    
    ctr_x = (min_xy[0] + max_xy[0]) / 2
    ctr_y = (min_xy[1] + max_xy[1]) / 2
    
    # Height of torso
    h = max_xy[1] - min_xy[1]
    
    # Radius = 1.6 * Height (Slightly zoomed out to include hands)
    if h < 0.1: h = 0.5
    radius = h * 1.5 
    
    return ctr_x, ctr_y, radius

def draw_skeleton_cv2(canvas, kps_95, ctr_x, ctr_y, radius):
    H, W, _ = canvas.shape
    
    # Scale to fit 'radius' in canvas half-height
    # View window: [ctr-r, ctr+r]
    # Scale factor: (H/2) / r ? No.
    # We want 2*r to cover the Canvas Height (approx)
    
    # Let's say H maps to 2*radius (+ padding)
    scale = H / (2 * radius * 1.2) # 1.2 for extra margin
    
    def transform(pt):
        # Center pt relative to ctr
        # x' = (pt.x - ctr.x) * scale + W/2
        # y' = (pt.y - ctr.y) * scale + H/2
        px = int((pt[0] - ctr_x) * scale + W/2)
        py = int((pt[1] - ctr_y) * scale + H/2)
        return (px, py)

    VALID_THRESH = 0.01
    MAX_LEN_SQ = (0.5 * min(W, H))**2 # Artifact filter

    # Draw Lines
    for rule in DRAW_RULES:
        s, e = rule['idx']
        off = rule['off']
        color = rule['color']
        th = rule['lw']
        
        real_s = s + (off[0] if isinstance(off, tuple) else off)
        real_e = e + (off[1] if isinstance(off, tuple) else off)
        
        if real_s >= len(kps_95) or real_e >= len(kps_95): continue
        p1, p2 = kps_95[real_s], kps_95[real_e]
        
        if np.sum(np.abs(p1)) < VALID_THRESH or np.sum(np.abs(p2)) < VALID_THRESH: continue
        t1, t2 = transform(p1), transform(p2)
        
        if (t1[0]-t2[0])**2 + (t1[1]-t2[1])**2 > MAX_LEN_SQ: continue
        
        cv2.line(canvas, t1, t2, color, th, cv2.LINE_AA)

    # Draw Dots (Replicating check_extraction_result.py style)
    # Side Joints (Teal)
    for i in JOINTS_SIDE:
        if i < len(kps_95) and np.sum(np.abs(kps_95[i])) > VALID_THRESH:
            cv2.circle(canvas, transform(kps_95[i]), 3, C_SIDE, -1, cv2.LINE_AA)
            
    # Trunk Joints (Black) - Slightly larger
    for i in JOINTS_TRUNK:
        if i < len(kps_95) and np.sum(np.abs(kps_95[i])) > VALID_THRESH:
            cv2.circle(canvas, transform(kps_95[i]), 4, C_TRUNK, -1, cv2.LINE_AA)
            
    # Eyes (Using Face indices 2 and 5? In 75-point, eyes are 15,16?)
    # check_extraction_result uses manual indices? 
    # It says: scatters[3] -> eyes. Logic: kps[frame, 2] and kps[frame, 5]?
    # Wait, check_extraction_result.py lines 150-151: kps[2] & kps[5].
    # In my 75-point mapping, 2 is R_Shoulder? No.
    # Op 25: 0=Nose, 1=Neck, 2=RShoulder.
    # Wait, check_extraction_result.py FACE_PAIRS uses (0,1), (1,2)... 
    # This implies indices 0,1,2,3,4,5,6,7,8,9,10 are FACE?
    # IF check_extraction_result assumes first 11 points are Face, then my 75-point load is wrong?
    # NO. `load_and_prepare_pose` in check_extraction_result simply reshapes 75->(75,2).
    # IF indices 2 and 5 are eyes, then the topology is NOT OpenPose 25.
    # It might be the "Sign Language Dataset" customized topology.
    # BUT, I must follow what `check_extraction_result.py` does visually.
    # It draws circles at indices 2 and 5 (Black, Large).
    
    # Let's map 2 and 5 as Eyes based on that file.
    for i in [2, 5]:
         if i < len(kps_95) and np.sum(np.abs(kps_95[i])) > VALID_THRESH:
            cv2.circle(canvas, transform(kps_95[i]), 5, (0,0,0), -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', required=True)
    parser.add_argument('--recon_path', required=True)
    parser.add_argument('--output', default='comparison_viz.mp4')
    args = parser.parse_args()

    gt_raw = np.load(args.gt_path)
    recon_raw = np.load(args.recon_path)
    if gt_raw.ndim == 1: gt_raw = gt_raw[None, :]
    if recon_raw.ndim == 1: recon_raw = recon_raw[None, :]

    L = min(len(gt_raw), len(recon_raw))
    print(f"🎬 Processing {L} frames...")

    gt_kps = [load_and_prepare_pose_95(f) for f in gt_raw[:L]]
    rc_kps = [load_and_prepare_pose_95(f) for f in recon_raw[:L]]
    
    # Calculate Center and Radius based on GT Torso
    ctr_x, ctr_y, radius = get_torso_zoom_params(np.array(gt_kps))
    
    H, W = 512, 512
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 25, (W*2, H))

    for t in tqdm(range(L)):
        f_gt = np.ones((H, W, 3), dtype=np.uint8) * 255
        f_rc = np.ones((H, W, 3), dtype=np.uint8) * 255
        
        draw_skeleton_cv2(f_gt, gt_kps[t], ctr_x, ctr_y, radius)
        draw_skeleton_cv2(f_rc, rc_kps[t], ctr_x, ctr_y, radius)
        
        cv2.putText(f_gt, "GROUND TRUTH", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(f_rc, "RECONSTRUCTED", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        writer.write(np.hstack((f_gt, f_rc)))

    writer.release()
    print(f"✅ Video: {args.output}")

if __name__ == "__main__":
    main()