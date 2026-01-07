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
# Gray, Blue, Green, Red
COLOR_BODY = (128, 128, 128)   # Gray
COLOR_LHAND = (255, 0, 0)      # Blue (OpenCV uses BGR) -> This is actually Blue in BGR
COLOR_RHAND = (0, 255, 0)      # Green
COLOR_MOUTH = (0, 0, 255)      # Red

DRAW_RULES = []

# 1. Upper Body
DRAW_RULES.extend([
    {'idx': (s, e), 'off': 0, 'color': COLOR_BODY, 'lw': 2}
    for (s, e) in POSE_CONNECTIONS_UPPER_BODY
])

# 2. Left Hand (Index 33+)
DRAW_RULES.extend([
    {'idx': (s, e), 'off': 33, 'color': COLOR_LHAND, 'lw': 1}
    for (s, e) in HAND_CONNECTIONS
])

# 3. Connection: Body Left Wrist (15) -> Left Hand Root (33)
# Note: In 95-point array, Body index 15 is at 15. Hand Root is at 0+33=33.
DRAW_RULES.append({'idx': (15, 0), 'off': (0, 33), 'color': COLOR_LHAND, 'lw': 2})

# 4. Right Hand (Index 54+)
DRAW_RULES.extend([
    {'idx': (s, e), 'off': 54, 'color': COLOR_RHAND, 'lw': 1}
    for (s, e) in HAND_CONNECTIONS
])

# 5. Connection: Body Right Wrist (16) -> Right Hand Root (54)
DRAW_RULES.append({'idx': (16, 0), 'off': (0, 54), 'color': COLOR_RHAND, 'lw': 2})

# 6. Mouth (Index 75+)
DRAW_RULES.extend([
    {'idx': (s, e), 'off': 75, 'color': COLOR_MOUTH, 'lw': 1}
    for (s, e) in MOUTH_CONNECTIONS_20
])


def load_and_prepare_pose_95(pose_214):
    """
    Transforms 214D vector into 95x2 matrix matching `visualize_single_pose.py` logic.
    """
    # 1. Manual/Body parts (0-150) -> 75x2
    # The reference script reshapes 150 -> (75,2)
    manual_150 = pose_214[:150]
    manual_kps = manual_150.reshape(75, 2)
    
    # 2. Mouth parts (174-214) -> 20x2 (Total 40 values)
    mouth_40 = pose_214[174:214]
    mouth_kps = mouth_40.reshape(20, 2)
    
    # Concatenate -> (95, 2)
    all_kps = np.concatenate([manual_kps, mouth_kps], axis=0)
    return all_kps

def get_crop_params(kps_seq, padding_ratio=0.2):
    """
    Calculate a STABLE crop/scale based on valid keypoints across the ENTIRE sequence or just the first frame.
    To avoid jitter, we compute global min/max of the GT sequence.
    """
    # Filter invalid points (approx 0)
    # We use all frames to find the bounding box of the actor
    mask = np.sum(np.abs(kps_seq), axis=2) > 0.01 # [T, 95]
    valid_points = kps_seq[mask]

    if len(valid_points) == 0:
        return 0, 0, 1.0 # Default

    min_x, min_y = np.min(valid_points, axis=0)
    max_x, max_y = np.max(valid_points, axis=0)
    
    width = max_x - min_x
    height = max_y - min_y
    
    pad_x = width * padding_ratio
    pad_y = height * padding_ratio
    
    # Apply padding
    min_x -= pad_x
    max_x += pad_x
    min_y -= pad_y
    max_y += pad_y
    
    return min_x, min_y, max_x, max_y

def draw_skeleton_cv2(canvas, kps_95, min_x, min_y, scale_w, scale_h):
    """
    Draw using OpenCV.
    Canvas size is assumed WxH.
    """
    H, W, _ = canvas.shape
    
    # Transform function
    def transform(pt):
        # Normalize to [0,1] then map to [0, W/H]
        # x' = (x - min_x) * scale_w
        px = int((pt[0] - min_x) * scale_w * W)
        py = int((pt[1] - min_y) * scale_h * H)
        return (px, py)

    # Valid check threshold
    VALID_THRESH = 0.01

    # Draw Lines
    for rule in DRAW_RULES:
        s, e = rule['idx']
        off = rule['off']
        color = rule['color']
        thickness = rule['lw']
        
        if isinstance(off, (tuple, list)):
            off_s, off_e = off
        else:
            off_s, off_e = off, off
            
        real_s = s + off_s
        real_e = e + off_e
        
        # Check indices bounds
        if real_s >= len(kps_95) or real_e >= len(kps_95): continue
        
        p1 = kps_95[real_s]
        p2 = kps_95[real_e]
        
        # Check valid
        if np.sum(np.abs(p1)) < VALID_THRESH or np.sum(np.abs(p2)) < VALID_THRESH:
            continue
            
        pt1 = transform(p1)
        pt2 = transform(p2)
        
        cv2.line(canvas, pt1, pt2, color, thickness, cv2.LINE_AA)

    # Draw Points (Optional: add dots for joints)
    # Body (Gray)
    for i in range(23):
        if np.sum(np.abs(kps_95[i])) > VALID_THRESH:
            cv2.circle(canvas, transform(kps_95[i]), 3, COLOR_BODY, -1, cv2.LINE_AA)
            
    # Hands & Mouth? (Maybe too crowded, but let's add small dots)
    # Left Hand
    for i in range(33, 54):
        if np.sum(np.abs(kps_95[i])) > VALID_THRESH:
            cv2.circle(canvas, transform(kps_95[i]), 2, COLOR_LHAND, -1, cv2.LINE_AA)
    # Right Hand
    for i in range(54, 75):
        if np.sum(np.abs(kps_95[i])) > VALID_THRESH:
            cv2.circle(canvas, transform(kps_95[i]), 2, COLOR_RHAND, -1, cv2.LINE_AA)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--recon_path', type=str, required=True)
    parser.add_argument('--output', type=str, default='comparison_viz.mp4')
    args = parser.parse_args()

    # Load Data
    gt_raw = np.load(args.gt_path)      # [T, 214]
    recon_raw = np.load(args.recon_path) # [T, 214]

    # Handle 1 frame case
    if gt_raw.ndim == 1: gt_raw = gt_raw[None, :]
    if recon_raw.ndim == 1: recon_raw = recon_raw[None, :]

    length = min(len(gt_raw), len(recon_raw))
    print(f"🎬 Processing {length} frames...")

    # Convert to 95-point format
    gt_kps_list = [load_and_prepare_pose_95(f) for f in gt_raw[:length]]
    recon_kps_list = [load_and_prepare_pose_95(f) for f in recon_raw[:length]]
    
    # Calculate Global Crop Parameter derived from GT (to ensure stability)
    min_x, min_y, max_x, max_y = get_crop_params(np.array(gt_kps_list))
    
    # Dimension check to avoid div/0
    box_w = max_x - min_x
    box_h = max_y - min_y
    if box_w < 1e-6: box_w = 1.0
    if box_h < 1e-6: box_h = 1.0

    # We want to fit this box into a square canvas (maintaining aspect ratio)
    # But usually we map normalized 0..1 to W,H. 
    # Here we map [min_x, max_x] -> [0, 1] relative to the box?
    # Simpler: Map [min_x, max_x] to the Canvas Width with padding.
    # To keep aspect ratio, we use the larger dimension as the 'scale' base.
    
    # Let's define the scaling factor to map the "box" to "canvas"
    # Ideally, we center the box.
    # New logic: transform returns normalized 0..1 coordinates relative to Box
    # But we need Square Box to keep Aspect Ratio.
    
    center_x = (min_x + max_x) / 2
    center_y = (min_y + max_y) / 2
    max_dim = max(box_w, box_h)
    
    # Update min/max to represent a square box
    min_x = center_x - max_dim / 2
    min_y = center_y - max_dim / 2
    # max_x = center_x + max_dim / 2
    # max_y = center_y + max_dim / 2
    
    # Inverse scale factor
    scale_factor = 1.0 / max_dim

    # Video Setup
    H, W = 512, 512
    # Black background is standard for these visualizations
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 25, (W*2, H))

    for t in tqdm(range(length)):
        # Canvas (Black)
        frame_gt = np.zeros((H, W, 3), dtype=np.uint8)
        frame_recon = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Draw GT
        draw_skeleton_cv2(frame_gt, gt_kps_list[t], min_x, min_y, scale_factor, scale_factor)
        
        # Draw Recon (Using SAME scaling params as GT for fair comparison and stability)
        draw_skeleton_cv2(frame_recon, recon_kps_list[t], min_x, min_y, scale_factor, scale_factor)
        
        # Add Text
        cv2.putText(frame_gt, "GROUND TRUTH", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(frame_recon, "RECONSTRUCTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Combine
        combined = np.hstack((frame_gt, frame_recon))
        writer.write(combined)

    writer.release()
    print(f"✅ Video saved: {args.output}")

if __name__ == "__main__":
    main()