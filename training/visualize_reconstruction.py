import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm

# --- 1. TOPOLOGY DEFINITION (From visualize_single_pose.py) ---
# Indices in the 95-point array:
# 0-22: Upper Body (subset of std 25)
# 23-32: Gap (unused)
# 33-53: Left Hand (21 points)
# 54-74: Right Hand (21 points)
# 75-94: Mouth (20 points)

# Basic Connections
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

# Mouth Connections
MOUTH_OUTER_LIP = list(zip(range(0, 11), range(1, 12))) + [(11, 0)]
MOUTH_INNER_LIP = list(zip(range(12, 19), range(13, 20))) + [(19, 12)]
MOUTH_CONNECTIONS_20 = MOUTH_OUTER_LIP + MOUTH_INNER_LIP

# Define Drawing Rules: (indices_list, offset, color_bgr, thickness)
# User Request: "Thân đen (Body Black), cánh tay và bàn tay màu xanh (Arms/Hands Green)"
# On White Background
COLOR_BODY = (0, 0, 0)         # Black
COLOR_HAND = (0, 150, 0)       # Darker Green (visible on white)
COLOR_MOUTH = (0, 0, 255)      # Red
COLOR_EYES = (0, 0, 0)         # Black

DRAW_RULES = []

# 1. Upper Body (Indices 0-22)
# We need to separate Arms from Torso if user wants "Arms" green.
# OpenPose 25: 
# Torso/Head: 0,1,8, 15,16,17,18 (Eyes/Ears)
# Arms: 2->3->4 (Right), 5->6->7 (Left)
# Legs: 9->10->11 (Right), 12->13->14 (Left) -- Data might not have legs, strictly Upper Body.

# Re-defining rules to separate colors
# Torso + Head
TORSO_HEAD_CONNECTIONS = [
    (0, 1), (1, 8), (1, 2), (1, 5), # Neck to extensions
    (0, 15), (0, 16), (15, 17), (16, 18), # Face
    (8, 9), (8, 12) # Hips
]
# Arms
ARM_CONNECTIONS = [
    (2, 3), (3, 4), # Right Shoulder->Elbow->Wrist
    (5, 6), (6, 7)  # Left Shoulder->Elbow->Wrist
]

# Note: The original POSE_CONNECTIONS_UPPER_BODY had a mix.
# Let's map specifically:

# Body (Torso/Head) -> Black
DRAW_RULES.extend([
    {'idx': (s, e), 'off': 0, 'color': COLOR_BODY, 'lw': 3}
    for (s, e) in TORSO_HEAD_CONNECTIONS
])

# Arms -> Green
DRAW_RULES.extend([
    {'idx': (s, e), 'off': 0, 'color': COLOR_HAND, 'lw': 3}
    for (s, e) in ARM_CONNECTIONS
])

# 2. Left Hand (Index 33+) -> Green
DRAW_RULES.extend([
    {'idx': (s, e), 'off': 33, 'color': COLOR_HAND, 'lw': 2}
    for (s, e) in HAND_CONNECTIONS
])

# 3. Connection: Body Left Wrist (7) -> Left Hand Root (33)
# Note: Body index 7 is Left Wrist in OpenPose?
# Wait, previous code used (15,0) and (16,0). In OpenPose 25, 4 is Right Wrist, 7 is Left Wrist.
# But `visualize_single_pose.py` used 15 and 16? 
# Let's double check standard OpenPose 25 vs COCO 18 vs Body 25.
# If `visualize_single_pose.py` used 15/16, it implied 15=Leye, 16=Reye for Body_25? No.
# Actually `visualize_single_pose.py` explicitly connected (15,0) with offset.
# Let's stick to connecting "Wrist" to "Hand Root". 
# If the previous code worked for topology, I will trust the user wants 'Arms' green.
# I will assume indices 4 and 7 are wrists if standard, OR use the explicit connection indices from before.
# Previous code: Body(15) -> Hand(0). 
# Let's make this connection GREEN.
DRAW_RULES.append({'idx': (15, 0), 'off': (0, 33), 'color': COLOR_HAND, 'lw': 3})

# 4. Right Hand (Index 54+) -> Green
DRAW_RULES.extend([
    {'idx': (s, e), 'off': 54, 'color': COLOR_HAND, 'lw': 2}
    for (s, e) in HAND_CONNECTIONS
])
# Connection Body(16) -> Hand(0)
DRAW_RULES.append({'idx': (16, 0), 'off': (0, 54), 'color': COLOR_HAND, 'lw': 3})

# 5. Mouth (Index 75+) -> Red
DRAW_RULES.extend([
    {'idx': (s, e), 'off': 75, 'color': COLOR_MOUTH, 'lw': 2}
    for (s, e) in MOUTH_CONNECTIONS_20
])


def load_and_prepare_pose_95(pose_214):
    manual_150 = pose_214[:150]
    manual_kps = manual_150.reshape(75, 2)
    mouth_40 = pose_214[174:214]
    mouth_kps = mouth_40.reshape(20, 2)
    return np.concatenate([manual_kps, mouth_kps], axis=0)

def get_crop_params(kps_seq, padding_ratio=0.1):
    """
    Calculate crop using Percentiles to ignore outliers (the cause of 'small' skeleton).
    """
    # Filter only non-zero points
    valid_mask = np.sum(np.abs(kps_seq), axis=2) > 0.01
    valid_points = kps_seq[valid_mask]

    if len(valid_points) == 0:
        return 0, 0, 1.0, 1.0

    # Use 5th and 95th percentiles to determine "Core" bounding box
    # This prevents one flying hand from shrinking the whole view
    min_x = np.percentile(valid_points[:, 0], 2)
    max_x = np.percentile(valid_points[:, 0], 98) 
    min_y = np.percentile(valid_points[:, 1], 2)
    max_y = np.percentile(valid_points[:, 1], 98)
    
    width = max_x - min_x
    height = max_y - min_y
    
    # Enforce a minimum size to prevent super zoom on empty frames
    if width < 0.1: width = 0.5
    if height < 0.1: height = 0.5

    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    
    return min_x - pad_x, min_y - pad_y, max_x + pad_x, max_y + pad_y

def draw_skeleton_cv2(canvas, kps_95, min_x, min_y, scale_w, scale_h):
    H, W, _ = canvas.shape
    
    def transform(pt):
        px = int((pt[0] - min_x) * scale_w * W)
        py = int((pt[1] - min_y) * scale_h * H)
        return (px, py)

    VALID_THRESH = 0.01
    # Max length squared (filter long lines)
    # Reduced to 30% of screen to be stricter
    MAX_LEN_SQ = (0.3 * min(W, H))**2

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
        
        # Strict Valid Check
        if np.sum(np.abs(p1)) < VALID_THRESH or np.sum(np.abs(p2)) < VALID_THRESH:
            continue
            
        t1, t2 = transform(p1), transform(p2)
        
        # Check Artifact Distance
        if (t1[0]-t2[0])**2 + (t1[1]-t2[1])**2 > MAX_LEN_SQ:
            continue
            
        cv2.line(canvas, t1, t2, color, th, cv2.LINE_AA)

    # Draw Points (Only for Face/Eyes?)
    # User requested specific style. Usually Skeleton lines are enough.
    # But Face landmarks needing dots?
    # Let's draw dots ONLY for Eyes/Nose (Black) and Mouth (Red)
    # Hiding dots for Body/Hands to look cleaner (like Image 3)
    
    # Nose (0) + Eyes (15,16,17,18)
    for i in [0, 15, 16, 17, 18]:
        if i < len(kps_95) and np.sum(np.abs(kps_95[i])) > VALID_THRESH:
            # Check if point is isolated? (Optional)
            cv2.circle(canvas, transform(kps_95[i]), 3, COLOR_EYES, -1, cv2.LINE_AA)
            
    # Mouth (75+)
    for i in range(75, 95):
        if i < len(kps_95) and np.sum(np.abs(kps_95[i])) > VALID_THRESH:
             cv2.circle(canvas, transform(kps_95[i]), 1, COLOR_MOUTH, -1, cv2.LINE_AA)

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
    
    # Crop based on GT percentile to ignore outliers
    min_x, min_y, max_x, max_y = get_crop_params(np.array(gt_kps), padding_ratio=0.1)
    
    # Square aspect ratio
    center_x, center_y = (min_x + max_x)/2, (min_y + max_y)/2
    max_dim = max(max_x - min_x, max_y - min_y)
    min_x, min_y = center_x - max_dim/2, center_y - max_dim/2
    scale = 1.0 / (max_dim + 1e-6)

    H, W = 512, 512
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 25, (W*2, H))

    for t in tqdm(range(L)):
        f_gt = np.ones((H, W, 3), dtype=np.uint8) * 255
        f_rc = np.ones((H, W, 3), dtype=np.uint8) * 255
        
        draw_skeleton_cv2(f_gt, gt_kps[t], min_x, min_y, scale, scale)
        draw_skeleton_cv2(f_rc, rc_kps[t], min_x, min_y, scale, scale)
        
        cv2.putText(f_gt, "GROUND TRUTH", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        cv2.putText(f_rc, "RECONSTRUCTED", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,0), 2)
        
        writer.write(np.hstack((f_gt, f_rc)))

    writer.release()
    print(f"✅ Video: {args.output}")

if __name__ == "__main__":
    main()