import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm

# --- 1. CẤU HÌNH MAPPING (Quan trọng để không bị dị dạng) ---
OFFSET_BODY = 0
OFFSET_LHAND = 33
OFFSET_RHAND = 54

# Định nghĩa các đường nối (Topology)
BODY_EDGES = [
    (1,0),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(8,12),
    (9,10),(10,11),(12,13),(13,14),(0,15),(0,16),(15,17),(16,18)
]
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
    (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)
]

def auto_scale_pose(pose, W=512, H=512, padding=60):
    """
    Tự động scale pose vào giữa khung hình
    """
    # Lấy các điểm thực sự (bỏ qua khoảng gap 23-32)
    valid_indices = list(range(0,23)) + list(range(33,54)) + list(range(54,75))
    valid_points = pose[valid_indices]
    
    # Lọc điểm rác (0,0)
    mask = np.sum(np.abs(valid_points), axis=1) > 0.001
    valid_points = valid_points[mask]

    if len(valid_points) == 0: return pose

    min_x, min_y = np.min(valid_points, axis=0)
    max_x, max_y = np.max(valid_points, axis=0)
    
    # Tính tỉ lệ scale giữ aspect ratio
    scale = min((W - 2*padding) / (max_x - min_x + 1e-6), 
                (H - 2*padding) / (max_y - min_y + 1e-6))
    
    # Scale & Center
    pose_scaled = (pose - [min_x, min_y]) * scale
    
    # Căn giữa
    off_x = (W - (max_x - min_x) * scale) / 2
    off_y = (H - (max_y - min_y) * scale) / 2
    
    return pose_scaled + [off_x, off_y]

def draw_skeleton(canvas, keypoints, is_gt=True):
    """Vẽ skeleton với màu sắc phân biệt trái/phải"""
    
    # Cấu hình màu (BGR)
    # GT: Body Xanh lá, Tay Trái Đỏ, Tay Phải Xanh Dương (Chuẩn OpenPose)
    # REC: Body Vàng, Tay Trái Hồng, Tay Phải Cyan (Để dễ phân biệt)
    if is_gt:
        c_body = (0, 255, 0)      # Green
        c_lhand = (0, 0, 255)     # Red
        c_rhand = (255, 0, 0)     # Blue
    else:
        c_body = (0, 255, 255)    # Yellow
        c_lhand = (255, 0, 255)   # Magenta
        c_rhand = (255, 255, 0)   # Cyan

    def draw_part(edges, offset, color, thick):
        for u, v in edges:
            idx_u, idx_v = u + offset, v + offset
            if idx_u < len(keypoints) and idx_v < len(keypoints):
                pt1 = tuple(keypoints[idx_u].astype(int))
                pt2 = tuple(keypoints[idx_v].astype(int))
                if pt1 != (0,0) and pt2 != (0,0):
                    cv2.line(canvas, pt1, pt2, color, thick)
                    # Vẽ khớp
                    cv2.circle(canvas, pt1, 3, color, -1)

    draw_part(BODY_EDGES, OFFSET_BODY, c_body, 2)
    draw_part(HAND_EDGES, OFFSET_LHAND, c_lhand, 2)
    draw_part(HAND_EDGES, OFFSET_RHAND, c_rhand, 2)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--recon_path', type=str, required=True)
    parser.add_argument('--video_id', type=str, default="Unknown ID", help="ID video để hiển thị")
    parser.add_argument('--output', type=str, default='comparison.mp4')
    args = parser.parse_args()
    
    # Load Data
    gt = np.load(args.gt_path)
    rec = np.load(args.recon_path)
    min_len = min(len(gt), len(rec))
    
    # Config Video
    H, W = 512, 512
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 25, (W*2, H))
    
    print(f"🎬 Rendering Video ID: {args.video_id}")
    
    for t in tqdm(range(min_len)):
        # Reshape & Scale
        p_gt = auto_scale_pose(gt[t, :150].reshape(-1, 2), W, H)
        p_rec = auto_scale_pose(rec[t, :150].reshape(-1, 2), W, H)
        
        # Canvas
        f_gt = np.zeros((H, W, 3), dtype=np.uint8)
        f_rec = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Draw
        draw_skeleton(f_gt, p_gt, is_gt=True)
        draw_skeleton(f_rec, p_rec, is_gt=False)
        
        # Text Info
        # 1. Tiêu đề
        cv2.putText(f_gt, "ORIGINAL (GT)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(f_rec, "RECONSTRUCTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        
        # 2. Video ID (Quan trọng)
        cv2.putText(f_gt, f"ID: {args.video_id}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 1)
        
        # 3. Frame Number
        cv2.putText(f_gt, f"Frame: {t}/{min_len}", (20, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Merge & Save
        out.write(np.hstack([f_gt, f_rec]))
        
    out.release()
    print(f"✅ Xong! Video lưu tại: {args.output}")

if __name__ == '__main__':
    main()