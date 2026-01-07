import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm

# --- 1. CẤU HÌNH INDEX DỮ LIỆU (QUAN TRỌNG) ---
# Dựa trên code cũ của chị: Body(0-23), LHand(33-54), RHand(54-75)
IDXS = {
    'body': slice(0, 23),       # Lấy 23 điểm đầu
    'l_hand': slice(33, 54),    # Bắt đầu từ 33 (21 điểm)
    'r_hand': slice(54, 75)     # Bắt đầu từ 54 (21 điểm)
}

# Bản đồ nối dây (Edges)
BODY_EDGES = [
    (1, 0), (1, 2), (1, 5),   # Cổ->Mũi, Vai
    (2, 3), (3, 4),           # Tay phải
    (5, 6), (6, 7),           # Tay trái
    (1, 8), (8, 9), (8, 12),  # Thân
    (9, 10), (10, 11),        # Chân phải
    (12, 13), (13, 14),       # Chân trái
    (0, 15), (0, 16),         # Mắt
    (15, 17), (16, 18)        # Tai
]

HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Ngón cái
    (0, 5), (5, 6), (6, 7), (7, 8),      # Ngón trỏ
    (0, 9), (9, 10), (10, 11), (11, 12), # Ngón giữa
    (0, 13), (13, 14), (14, 15), (15, 16), # Ngón áp út
    (0, 17), (17, 18), (18, 19), (19, 20)  # Ngón út
]

def auto_scale_pose(pose, W=512, H=512, padding=50):
    """Tự động scale pose vào giữa khung hình"""
    # Lấy các điểm quan trọng để tính scale (tránh lấy điểm 0,0)
    # Gom tất cả điểm body và hand lại để tính bounding box chung
    body = pose[IDXS['body']]
    l_hand = pose[IDXS['l_hand']]
    r_hand = pose[IDXS['r_hand']]
    all_points = np.concatenate([body, l_hand, r_hand], axis=0)
    
    valid_mask = np.sum(np.abs(all_points), axis=1) > 0.001
    valid_points = all_points[valid_mask]
    
    if len(valid_points) == 0: return pose

    min_x, min_y = np.min(valid_points, axis=0)
    max_x, max_y = np.max(valid_points, axis=0)
    
    scale = min((W - 2*padding) / (max_x - min_x + 1e-6), 
                (H - 2*padding) / (max_y - min_y + 1e-6))
    
    # Scale & Center
    pose_scaled = (pose - [min_x, min_y]) * scale
    
    # Tính offset để đặt vào giữa
    cur_w = (max_x - min_x) * scale
    cur_h = (max_y - min_y) * scale
    off_x = (W - cur_w) / 2
    off_y = (H - cur_h) / 2
    
    return pose_scaled + [off_x, off_y]

def draw_pose_on_canvas(canvas, keypoints, is_gt=True):
    """Vẽ lên nền đen dùng đúng Index Map"""
    # Cắt dữ liệu theo đúng index của chị
    body = keypoints[IDXS['body']]
    l_hand = keypoints[IDXS['l_hand']]
    r_hand = keypoints[IDXS['r_hand']]
    
    # Màu sắc: GT=Xanh lá, Rec=Đỏ (Style ảnh mẫu)
    c_body = (0, 255, 0) if is_gt else (0, 0, 255)
    c_hand_l = (0, 200, 200) if is_gt else (0, 128, 255) # Vàng chanh / Cam
    c_hand_r = (200, 200, 0) if is_gt else (255, 0, 255) # Xanh lơ / Tím

    def draw_part(points, edges, color):
        for u, v in edges:
            if u < len(points) and v < len(points):
                pt1 = tuple(points[u].astype(int))
                pt2 = tuple(points[v].astype(int))
                if pt1 != (0,0) and pt2 != (0,0):
                    cv2.line(canvas, pt1, pt2, color, 2 if len(points)>22 else 1)
                    if len(points) > 22: # Chỉ vẽ khớp cho Body
                        cv2.circle(canvas, pt1, 3, color, -1)

    draw_part(body, BODY_EDGES, c_body)
    draw_part(l_hand, HAND_EDGES, c_hand_l)
    draw_part(r_hand, HAND_EDGES, c_hand_r)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True)
    parser.add_argument('--recon_path', type=str, required=True)
    parser.add_argument('--output_video', type=str, default='comparison_fixed.mp4')
    args = parser.parse_args()
    
    gt_data = np.load(args.gt_path)
    recon_data = np.load(args.recon_path)
    
    min_len = min(len(gt_data), len(recon_data))
    H, W = 512, 512
    out = cv2.VideoWriter(args.output_video, cv2.VideoWriter_fourcc(*'mp4v'), 25, (W*2, H))
    
    print(f"🎬 Rendering {min_len} frames (Fixed Index Mapping)...")
    
    for t in tqdm(range(min_len)):
        # Reshape về [75, 2]
        p_gt = gt_data[t, :150].reshape(-1, 2)
        p_rec = recon_data[t, :150].reshape(-1, 2)
        
        # Scale
        p_gt = auto_scale_pose(p_gt, W, H)
        p_rec = auto_scale_pose(p_rec, W, H)
        
        # Draw
        frame_gt = np.zeros((H, W, 3), dtype=np.uint8)
        frame_rec = np.zeros((H, W, 3), dtype=np.uint8)
        
        draw_pose_on_canvas(frame_gt, p_gt, is_gt=True)
        draw_pose_on_canvas(frame_rec, p_rec, is_gt=False)
        
        # Text
        cv2.putText(frame_gt, "ORIGINAL (GT)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_rec, "RECONSTRUCTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        out.write(np.hstack([frame_gt, frame_rec]))
        
    out.release()
    print(f"✅ Xong! Video: {args.output_video}")

if __name__ == '__main__':
    main()