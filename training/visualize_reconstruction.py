import numpy as np
import cv2
import argparse
from tqdm import tqdm

# --- 1. ĐỊNH NGHĨA KẾT NỐI XƯƠNG (TOPOLOGY) ---
# Cấu trúc 75 điểm: Body (25) + Left Hand (21) + Right Hand (21) + Face (8)
# Indices: 0-24 (Body), 25-45 (Left Hand), 46-66 (Right Hand), 67-74 (Face landmarks)

# Body connections (OpenPose 25-point body model)
BODY_CONNECTIONS = [
    # Spine and neck
    (0, 1),    # Nose -> Neck
    (1, 2),    # Neck -> Right Shoulder
    (1, 5),    # Neck -> Left Shoulder
    (1, 8),    # Neck -> Mid Hip
    
    # Right arm
    (2, 3),    # Right Shoulder -> Right Elbow
    (3, 4),    # Right Elbow -> Right Wrist
    
    # Left arm
    (5, 6),    # Left Shoulder -> Left Elbow
    (6, 7),    # Left Elbow -> Left Wrist
    
    # Right leg
    (8, 9),    # Mid Hip -> Right Hip
    (9, 10),   # Right Hip -> Right Knee
    (10, 11),  # Right Knee -> Right Ankle
    
    # Left leg
    (8, 12),   # Mid Hip -> Left Hip
    (12, 13),  # Left Hip -> Left Knee
    (13, 14),  # Left Knee -> Left Ankle
    
    # Face
    (0, 15),   # Nose -> Right Eye
    (0, 16),   # Nose -> Left Eye
    (15, 17),  # Right Eye -> Right Ear
    (16, 18),  # Left Eye -> Left Ear
]

# Hand connections (21 points per hand - MediaPipe/OpenPose hand model)
# Each finger: Wrist(0) -> MCP -> PIP -> DIP -> TIP
def get_hand_connections(offset=0):
    """Generate hand skeleton connections with given offset"""
    connections = []
    # Wrist to palm base
    connections.extend([
        (0+offset, 1+offset), (0+offset, 5+offset), (0+offset, 9+offset),
        (0+offset, 13+offset), (0+offset, 17+offset)
    ])
    # Thumb (4 points)
    connections.extend([(1+offset, 2+offset), (2+offset, 3+offset), (3+offset, 4+offset)])
    # Index finger (4 points)
    connections.extend([(5+offset, 6+offset), (6+offset, 7+offset), (7+offset, 8+offset)])
    # Middle finger (4 points)
    connections.extend([(9+offset, 10+offset), (10+offset, 11+offset), (11+offset, 12+offset)])
    # Ring finger (4 points)
    connections.extend([(13+offset, 14+offset), (14+offset, 15+offset), (15+offset, 16+offset)])
    # Pinky (4 points)
    connections.extend([(17+offset, 18+offset), (18+offset, 19+offset), (19+offset, 20+offset)])
    return connections

# Left hand: indices 25-45 (21 points)
LEFT_HAND_CONNECTIONS = get_hand_connections(offset=25)

# Right hand: indices 46-66 (21 points)
RIGHT_HAND_CONNECTIONS = get_hand_connections(offset=46)

def get_skeleton_topology():
    """Return all skeleton connections"""
    edges = list(BODY_CONNECTIONS)
    edges.extend(LEFT_HAND_CONNECTIONS)
    edges.extend(RIGHT_HAND_CONNECTIONS)
    return edges

# --- 2. HÀM XỬ LÝ DỮ LIỆU ---

def split_pose(pose_vector):
    """
    Tách vector 214D -> Body (75,2) và Mouth (20,2)
    """
    # Body: 0-150 -> 75 điểm x 2
    body = pose_vector[:150].reshape(-1, 2)
    # Mouth: 174-214 -> 20 điểm x 2
    mouth = pose_vector[174:214].reshape(-1, 2)
    return body, mouth

def robust_normalize_to_canvas(points, W=512, H=512, padding=50):
    """
    Kỹ thuật AUTO-ZOOM: Bất chấp input là số nhỏ (normalized) hay số to,
    nó sẽ scale về khung hình 512x512.
    """
    # Lọc bỏ các điểm (0,0) hoặc gần 0 để không bị nhiễu scale
    valid_mask = np.sum(np.abs(points), axis=1) > 0.01
    valid_points = points[valid_mask]

    # Nếu không có điểm nào valid (frame đen), trả về gốc
    if len(valid_points) == 0:
        return points

    # Tìm Min/Max của dữ liệu thật
    min_x, min_y = np.min(valid_points, axis=0)
    max_x, max_y = np.max(valid_points, axis=0)

    # Tính width/height của skeleton
    skel_w = max_x - min_x
    skel_h = max_y - min_y

    # Tránh chia cho 0
    if skel_w < 1e-6: skel_w = 1
    if skel_h < 1e-6: skel_h = 1

    # Tính tỷ lệ scale để fit vào khung hình (trừ padding)
    scale_x = (W - 2 * padding) / skel_w
    scale_y = (H - 2 * padding) / skel_h
    scale = min(scale_x, scale_y) # Giữ aspect ratio

    # Công thức: (Point - Min) * Scale + Padding + Center_Offset
    # Center offset để hình nằm giữa khung
    final_w = skel_w * scale
    final_h = skel_h * scale
    offset_x = padding + (W - 2*padding - final_w) / 2
    offset_y = padding + (H - 2*padding - final_h) / 2

    # Apply transform
    points_scaled = np.copy(points)
    points_scaled[valid_mask] = (valid_points - [min_x, min_y]) * scale + [offset_x, offset_y]
    
    # Những điểm invalid (0,0) gán về -1 để không vẽ
    points_scaled[~valid_mask] = -1 
    
    return points_scaled

# --- 3. HÀM VẼ (STYLE OPENPOSE) ---

def draw_pose(canvas, body, mouth, is_gt=True):
    """Draw pose skeleton with proper topology"""
    # Màu sắc (BGR) - Softer colors for better visual
    if is_gt:
        c_body = (100, 255, 100)     # Light Green
        c_hand = (255, 100, 100)     # Light Blue
        c_face = (200, 200, 200)     # Light Gray
        label = "GROUND TRUTH"
    else:
        c_body = (100, 255, 255)     # Light Yellow
        c_hand = (255, 100, 255)     # Light Purple
        c_face = (150, 200, 255)     # Light Orange
        label = "RECONSTRUCTION"

    # Get all skeleton connections
    skeleton_edges = get_skeleton_topology()
    
    # 1. Draw skeleton lines first (so points appear on top)
    for (i, j) in skeleton_edges:
        # Check if both points are valid
        if i >= len(body) or j >= len(body):
            continue
        
        pt1 = body[i]
        pt2 = body[j]
        
        # Skip if either point is invalid
        if pt1[0] < 0 or pt2[0] < 0:
            continue
        
        # Determine color based on connection type
        if i >= 25 or j >= 25:  # Hand connections
            color = c_hand
            thickness = 1
        else:  # Body connections
            color = c_body
            thickness = 2
        
        # Draw line
        cv2.line(canvas, 
                (int(pt1[0]), int(pt1[1])), 
                (int(pt2[0]), int(pt2[1])), 
                color, thickness, cv2.LINE_AA)
    
    # 2. Draw keypoints on top of lines
    for i, pt in enumerate(body):
        if pt[0] < 0: continue  # Skip invalid points
        
        # Determine color and size based on point type
        if i >= 25:  # Hand points
            color = c_hand
            radius = 2
        else:  # Body points
            color = c_body
            radius = 3
        
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), radius, color, -1, cv2.LINE_AA)
    
    # 3. Draw mouth points
    for pt in mouth:
        if pt[0] < 0: continue
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 1, c_face, -1, cv2.LINE_AA)
    
    # 4. Text Label (smaller and cleaner)
    cv2.putText(canvas, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, c_body, 1, cv2.LINE_AA)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gt_path", required=True)
    parser.add_argument("--recon_path", required=True)
    parser.add_argument("--output", default="comparison_fixed.mp4")
    args = parser.parse_args()

    # Load Data
    gt_seq = np.load(args.gt_path)
    recon_seq = np.load(args.recon_path)

    # Fix shape 1 frame
    if gt_seq.ndim == 1: gt_seq = gt_seq[None, :]
    if recon_seq.ndim == 1: recon_seq = recon_seq[None, :]

    length = min(len(gt_seq), len(recon_seq))
    
    # Init Video Writer
    H, W = 512, 512
    writer = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 25, (W*2, H))

    print(f"🎬 Processing {length} frames...")

    for t in tqdm(range(length)):
        # Tạo canvas đen
        frame_gt = np.zeros((H, W, 3), dtype=np.uint8)
        frame_recon = np.zeros((H, W, 3), dtype=np.uint8)

        # 1. Tách Features
        b_gt, m_gt = split_pose(gt_seq[t])
        b_rc, m_rc = split_pose(recon_seq[t])

        # 2. AUTO-ZOOM (Critical Step)
        # Scale GT
        b_gt_s = robust_normalize_to_canvas(b_gt, W, H)
        m_gt_s = robust_normalize_to_canvas(m_gt, W, H) # Lưu ý: Mouth nên scale theo Body để đúng tỉ lệ
        
        # Scale Recon
        # Ta dùng parameter scale của GT để áp dụng cho Recon -> Để so sánh công bằng vị trí
        # Tuy nhiên, nếu recon nát quá thì scale riêng. Ở đây scale riêng cho chắc ăn hiển thị.
        b_rc_s = robust_normalize_to_canvas(b_rc, W, H)
        m_rc_s = robust_normalize_to_canvas(m_rc, W, H)

        # 3. Vẽ
        draw_pose(frame_gt, b_gt_s, m_gt_s, is_gt=True)
        draw_pose(frame_recon, b_rc_s, m_rc_s, is_gt=False)

        # 4. Gộp và Lưu
        combined = np.hstack((frame_gt, frame_recon))
        writer.write(combined)

    writer.release()
    print(f"✅ Đã lưu video tại: {args.output}")

if __name__ == "__main__":
    main()