import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm

# --- 1. ĐỊNH NGHĨA KHỚP NỐI (SKELETON TOPOLOGY) ---
# Dựa trên chuẩn OpenPose thường dùng cho Phoenix/How2Sign (75 points)
# 0-24: Body, 25-45: Left Hand, 46-66: Right Hand

# Body (OpenPose 25)
BODY_EDGES = [
    (1, 0), (1, 2), (1, 5),   # Neck -> Nose, R-Shoulder, L-Shoulder
    (2, 3), (3, 4),           # R-Arm
    (5, 6), (6, 7),           # L-Arm
    (1, 8), (8, 9), (8, 12),  # Neck->MidHip, MidHip->R-Hip, MidHip->L-Hip
    (9, 10), (10, 11),        # R-Leg
    (12, 13), (13, 14),       # L-Leg
    (0, 15), (0, 16),         # Nose -> Eyes
    (15, 17), (16, 18),       # Ears
    (11, 24), (11, 22), (22, 23), # R-Foot details
    (14, 21), (14, 19), (19, 20)  # L-Foot details
]

# Hand (21 points: Wrist=0, Thumb=1-4, Index=5-8...)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

def scale_to_canvas(pose, W=512, H=512, padding=50):
    """
    Tự động scale pose để vừa khít khung hình 512x512
    pose: [75, 2]
    """
    # Lọc các điểm (0,0) (điểm ẩn/nhiễu) ra khỏi việc tính min/max
    valid_points = pose[np.sum(pose, axis=1) != 0]
    
    if len(valid_points) == 0:
        return pose # Không có điểm nào valid
        
    min_x, min_y = np.min(valid_points, axis=0)
    max_x, max_y = np.max(valid_points, axis=0)
    
    # Kích thước thật của pose
    pose_w = max_x - min_x
    pose_h = max_y - min_y
    
    # Tỉ lệ scale
    scale_x = (W - 2*padding) / (pose_w + 1e-6)
    scale_y = (H - 2*padding) / (pose_h + 1e-6)
    scale = min(scale_x, scale_y)
    
    # Scale và Center
    pose_scaled = (pose - [min_x, min_y]) * scale
    
    # Căn giữa
    offset_x = (W - pose_w * scale) / 2
    offset_y = (H - pose_h * scale) / 2
    
    return pose_scaled + [offset_x, offset_y]

def draw_skeleton(frame, keypoints, color_body=(0, 255, 0), color_lhand=(0, 0, 255), color_rhand=(255, 0, 0)):
    """
    Vẽ khung xương lên frame
    keypoints: [75, 2]
    """
    # 1. Tách bộ phận
    body = keypoints[0:25]
    l_hand = keypoints[25:46]
    r_hand = keypoints[46:67]
    
    # Hàm phụ vẽ line
    def draw_connections(points, edges, color, thickness=2):
        for u, v in edges:
            if u < len(points) and v < len(points):
                pt1 = tuple(points[u].astype(int))
                pt2 = tuple(points[v].astype(int))
                # Chỉ vẽ nếu không phải điểm (0,0) hoặc điểm ngoài khung
                if pt1 != (0,0) and pt2 != (0,0):
                     cv2.line(frame, pt1, pt2, color, thickness)
                     cv2.circle(frame, pt1, 2, color, -1)

    # 2. Vẽ Body (Xanh Lá)
    draw_connections(body, BODY_EDGES, color_body, thickness=2)
    
    # 3. Vẽ Tay Trái (Đỏ - Red)
    draw_connections(l_hand, HAND_EDGES, color_lhand, thickness=1)

    # 4. Vẽ Tay Phải (Xanh Dương - Blue)
    draw_connections(r_hand, HAND_EDGES, color_rhand, thickness=1)

    return frame

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str, required=True)
    parser.add_argument('--reconstructed', type=str, required=True)
    parser.add_argument('--output', type=str, default='comparison_result.mp4')
    args = parser.parse_args()
    
    # Load data [T, 214]
    gt = np.load(args.original)
    rec = np.load(args.reconstructed)
    
    T = len(gt)
    H, W = 512, 512
    
    # Reshape về [T, 75, 2] (Chỉ lấy 150 chiều đầu Manual Pose)
    gt_pose = gt[:, :150].reshape(-1, 75, 2)
    rec_pose = rec[:, :150].reshape(-1, 75, 2)
    
    # --- AUTO-SCALE ---
    # Kiểm tra xem dữ liệu là pixel (0-256) hay norm (-2.0 đến 2.0)
    # Nếu là norm, ta cần denormalize "giả" để nó hiện lên hình được
    is_normalized = np.max(np.abs(gt_pose)) < 10.0
    print(f"📊 Data Type detected: {'Normalized' if is_normalized else 'Pixel Coordinates'}")

    # Init Video Writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, 25, (W*2, H))
    
    print("🎬 Rendering video...")
    for t in tqdm(range(T)):
        # Tạo canvas đen
        canvas_gt = np.zeros((H, W, 3), dtype=np.uint8)
        canvas_rec = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Lấy frame t
        p_gt = gt_pose[t].copy()
        p_rec = rec_pose[t].copy()
        
        # Nếu data normalized -> Scale theo min/max của chính frame đó (hoặc global) để fit vào 512x512
        # Ở đây dùng hàm scale_to_canvas để luôn đảm bảo hình nằm giữa
        p_gt = scale_to_canvas(p_gt, W, H)
        p_rec = scale_to_canvas(p_rec, W, H)
        
        # Vẽ
        # GT: Body Xanh, Tay Đỏ/Xanh Dương
        draw_skeleton(canvas_gt, p_gt)
        # Rec: Body Vàng, Tay Đỏ/Xanh Dương (để phân biệt)
        draw_skeleton(canvas_rec, p_rec, color_body=(0, 255, 255)) 
        
        # Thêm Text
        cv2.putText(canvas_gt, "Original (GT)", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(canvas_rec, "Reconstructed", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Ghép
        frame = np.hstack([canvas_gt, canvas_rec])
        out.write(frame)
        
    out.release()
    print(f"✅ Xong! Video lưu tại: {args.output}")

if __name__ == '__main__':
    main()