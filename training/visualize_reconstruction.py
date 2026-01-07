import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm

# --- CẤU HÌNH ---
# Định nghĩa các đường nối (Topology) cho Body & Hand
BODY_EDGES = [
    (1,0),(1,2),(1,5),(2,3),(3,4),(5,6),(6,7),(1,8),(8,9),(8,12),
    (9,10),(10,11),(12,13),(13,14),(0,15),(0,16),(15,17),(16,18)
]
HAND_EDGES = [
    (0,1),(1,2),(2,3),(3,4),(0,5),(5,6),(6,7),(7,8),(0,9),(9,10),
    (10,11),(11,12),(0,13),(13,14),(14,15),(15,16),(0,17),(17,18),(18,19),(19,20)
]

def split_pose_features(pose_frame):
    """
    Tách 1 frame [214] thành Body [75,2] và Mouth [20,2]
    """
    # 1. Body/Hands: 0-150
    body_flat = pose_frame[:150]
    body_kps = body_flat.reshape(75, 2)
    
    # 2. Mouth: 174-214
    mouth_flat = pose_frame[174:214]
    mouth_kps = mouth_flat.reshape(20, 2)
    
    return body_kps, mouth_kps

def get_scale_params(body_points, W=512, H=512, padding=50):
    """
    Tính toán scale dựa trên BODY để giữ tỉ lệ chuẩn cho cả mặt
    """
    # Lấy các điểm thực sự (bỏ điểm 0,0 và các khoảng gap)
    valid_indices = list(range(0,23)) + list(range(33,54)) + list(range(54,75))
    valid_points = body_points[valid_indices]
    
    mask = np.sum(np.abs(valid_points), axis=1) > 0.001
    valid_points = valid_points[mask]

    if len(valid_points) == 0: return 1.0, 0, 0

    min_x, min_y = np.min(valid_points, axis=0)
    max_x, max_y = np.max(valid_points, axis=0)
    
    # Tính scale
    scale_w = (W - 2*padding) / (max_x - min_x + 1e-6)
    scale_h = (H - 2*padding) / (max_y - min_y + 1e-6)
    scale = min(scale_w, scale_h)
    
    # Tính offset để center
    off_x = (W - (max_x - min_x) * scale) / 2 - min_x * scale
    off_y = (H - (max_y - min_y) * scale) / 2 - min_y * scale
    
    return scale, off_x, off_y

def transform_points(points, scale, off_x, off_y):
    """Áp dụng scale và offset cho các điểm"""
    return points * scale + [off_x, off_y]

def draw_skeleton_and_face(canvas, body_kps, mouth_kps, is_gt=True):
    """Vẽ cả người và miệng"""
    
    # --- MÀU SẮC ---
    if is_gt:
        c_body = (0, 255, 0)      # Green
        c_lhand = (0, 0, 255)     # Red
        c_rhand = (255, 0, 0)     # Blue
        c_face  = (255, 255, 255) # White
    else:
        c_body = (0, 255, 255)    # Yellow
        c_lhand = (255, 0, 255)   # Magenta
        c_rhand = (255, 255, 0)   # Cyan
        c_face  = (0, 165, 255)   # Orange
    
    # 1. Vẽ Body & Hands (Lines)
    def draw_part(edges, offset, color):
        for u, v in edges:
            idx_u, idx_v = u + offset, v + offset
            pt1 = tuple(body_kps[idx_u].astype(int))
            pt2 = tuple(body_kps[idx_v].astype(int))
            # Check rác (0,0)
            if pt1[0] > 1 and pt2[0] > 1:
                cv2.line(canvas, pt1, pt2, color, 2)
                cv2.circle(canvas, pt1, 3, color, -1)

    draw_part(BODY_EDGES, 0, c_body)      # Body
    draw_part(HAND_EDGES, 33, c_lhand)    # Left Hand (offset 33)
    draw_part(HAND_EDGES, 54, c_rhand)    # Right Hand (offset 54)
    
    # 2. Vẽ Mouth (Dots)
    for pt in mouth_kps:
        pt_coord = tuple(pt.astype(int))
        if pt_coord[0] > 1: # Check rác
            cv2.circle(canvas, pt_coord, 2, c_face, -1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--recon_path', type=str, required=True)
    parser.add_argument('--output', type=str, default='comparison_final.mp4')
    args = parser.parse_args()
    
    # Load Data
    gt = np.load(args.gt_path)
    rec = np.load(args.recon_path)
    
    # Xử lý trường hợp 1 frame
    if gt.ndim == 1: gt = gt[np.newaxis, :]
    if rec.ndim == 1: rec = rec[np.newaxis, :]
        
    min_len = min(len(gt), len(rec))
    
    # Config Video
    H, W = 512, 512
    out = cv2.VideoWriter(args.output, cv2.VideoWriter_fourcc(*'mp4v'), 25, (W*2, H))
    
    print(f"🎬 Rendering {min_len} frames to {args.output}")
    
    for t in tqdm(range(min_len)):
        # 1. Parse Data
        gt_body, gt_mouth = split_pose_features(gt[t])
        rec_body, rec_mouth = split_pose_features(rec[t])
        
        # 2. Tính toán Scale dựa trên GT Body (để khung hình ổn định theo GT)
        # (Nếu dùng rec để tính scale thì hình rec sẽ bị giật nếu model lỗi)
        scale, off_x, off_y = get_scale_params(gt_body, W, H)
        
        # 3. Apply Scale cho tất cả
        gt_body_draw = transform_points(gt_body, scale, off_x, off_y)
        gt_mouth_draw = transform_points(gt_mouth, scale, off_x, off_y)
        
        rec_body_draw = transform_points(rec_body, scale, off_x, off_y)
        rec_mouth_draw = transform_points(rec_mouth, scale, off_x, off_y)
        
        # 4. Vẽ
        f_gt = np.zeros((H, W, 3), dtype=np.uint8)
        f_rec = np.zeros((H, W, 3), dtype=np.uint8)
        
        draw_skeleton_and_face(f_gt, gt_body_draw, gt_mouth_draw, is_gt=True)
        draw_skeleton_and_face(f_rec, rec_body_draw, rec_mouth_draw, is_gt=False)
        
        # Text Info
        cv2.putText(f_gt, "GROUND TRUTH", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(f_rec, "RECONSTRUCTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        cv2.putText(f_gt, f"Frame: {t}", (20, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)
        
        # Merge & Write
        out.write(np.hstack([f_gt, f_rec]))
        
    out.release()
    print("✅ Done!")

if __name__ == '__main__':
    main()