import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm

# --- 1. BẢN ĐỒ XƯƠNG CHUẨN (PHOENIX-14T / OPENPOSE) ---
# Body: 25 điểm (0-24)
BODY_EDGES = [
    (1, 0), (1, 2), (1, 5),   # Cổ->Mũi, Vai Phải, Vai Trái
    (2, 3), (3, 4),           # Cánh tay phải
    (5, 6), (6, 7),           # Cánh tay trái
    (1, 8), (8, 9), (8, 12),  # Thân trên -> Hông
    (9, 10), (10, 11),        # Chân phải
    (12, 13), (13, 14),       # Chân trái
    (0, 15), (0, 16),         # Mắt
    (15, 17), (16, 18)        # Tai
]

# Hand: 21 điểm (Gốc=0, Ngón cái=1-4, Trỏ=5-8...)
HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

def auto_scale_pose(pose, W=512, H=512, padding=50):
    """
    Tự động phóng to/thu nhỏ pose để vừa khít khung hình 512x512
    """
    # Lọc bỏ các điểm (0,0) (điểm bị che khuất/không có)
    valid_mask = np.sum(pose, axis=1) != 0
    valid_points = pose[valid_mask]
    
    if len(valid_points) == 0:
        return pose # Không có điểm nào để vẽ

    # Tìm hộp bao (Bounding Box)
    min_x, min_y = np.min(valid_points, axis=0)
    max_x, max_y = np.max(valid_points, axis=0)
    
    pose_w = max_x - min_x
    pose_h = max_y - min_y
    
    # Tính tỉ lệ scale để fit vào khung hình (trừ padding)
    scale_x = (W - 2*padding) / (pose_w + 1e-6)
    scale_y = (H - 2*padding) / (pose_h + 1e-6)
    scale = min(scale_x, scale_y) # Giữ tỉ lệ khung hình (aspect ratio)
    
    # Scale và dịch chuyển về giữa
    pose_scaled = (pose - [min_x, min_y]) * scale
    
    # Căn giữa
    new_w = pose_w * scale
    new_h = pose_h * scale
    offset_x = (W - new_w) / 2
    offset_y = (H - new_h) / 2
    
    return pose_scaled + [offset_x, offset_y]

def draw_pose_on_canvas(canvas, keypoints, is_gt=True):
    """
    Vẽ khung xương lên nền đen
    keypoints: [75, 2]
    """
    # Tách bộ phận (Dựa trên cấu trúc 75 điểm: 25 Body + 21 LHand + 21 RHand)
    # Lưu ý: Index có thể thay đổi tùy bộ data, nhưng đây là cấu trúc phổ biến nhất
    body = keypoints[0:25]
    l_hand = keypoints[25:46]
    r_hand = keypoints[46:67]
    
    # Màu sắc: GT (Xanh lá), Recon (Đỏ/Cam)
    color_body = (0, 255, 0) if is_gt else (0, 0, 255)       # Body
    color_lhand = (0, 200, 200) if is_gt else (0, 165, 255) # Tay Trái (Vàng/Cam)
    color_rhand = (200, 200, 0) if is_gt else (255, 0, 255) # Tay Phải (Xanh lơ/Tím)

    # Hàm vẽ đường nối
    def draw_lines(points, edges, color, thick=2):
        for u, v in edges:
            if u < len(points) and v < len(points):
                pt1 = tuple(points[u].astype(int))
                pt2 = tuple(points[v].astype(int))
                # Không vẽ nếu điểm là (0,0) hoặc bay ra ngoài khung
                if pt1 != (0,0) and pt2 != (0,0):
                    cv2.line(canvas, pt1, pt2, color, thick)
                    # Vẽ khớp tròn nhỏ
                    cv2.circle(canvas, pt1, 2, color, -1)

    draw_lines(body, BODY_EDGES, color_body, 2)
    draw_lines(l_hand, HAND_EDGES, color_lhand, 1)
    draw_lines(r_hand, HAND_EDGES, color_rhand, 1)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gt_path', type=str, required=True, help='Path file gốc .npy')
    parser.add_argument('--recon_path', type=str, required=True, help='Path file tái tạo .npy')
    parser.add_argument('--output_video', type=str, default='comparison_video.mp4')
    args = parser.parse_args()
    
    # 1. Load Data
    print(f"📂 Loading: {args.gt_path}")
    gt_data = np.load(args.gt_path)
    print(f"📂 Loading: {args.recon_path}")
    recon_data = np.load(args.recon_path)
    
    # 2. Xử lý độ dài lệch nhau (Cắt theo cái ngắn nhất)
    len_gt = len(gt_data)
    len_recon = len(recon_data)
    min_len = min(len_gt, len_recon)
    
    if len_gt != len_recon:
        print(f"⚠️ Cảnh báo: Độ dài lệch nhau (GT={len_gt}, Rec={len_recon}). Sẽ cắt về {min_len} frames.")
    
    # 3. Chuẩn bị Video Writer
    H, W = 512, 512 # Kích thước mỗi khung hình
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # Video đầu ra sẽ rộng gấp đôi (Side-by-side)
    out = cv2.VideoWriter(args.output_video, fourcc, 25, (W * 2, H))
    
    print(f"🎬 Đang render video ({min_len} frames)...")
    
    # 4. Vòng lặp vẽ
    for t in tqdm(range(min_len)):
        # Lấy data frame t & reshape về [75, 2] (Chỉ lấy 150 chiều đầu)
        pose_gt = gt_data[t, :150].reshape(-1, 2)
        pose_rec = recon_data[t, :150].reshape(-1, 2)
        
        # Auto Scale để fit vào khung hình 512x512
        pose_gt_scaled = auto_scale_pose(pose_gt, W, H)
        pose_rec_scaled = auto_scale_pose(pose_rec, W, H)
        
        # Tạo canvas đen
        canvas_gt = np.zeros((H, W, 3), dtype=np.uint8)
        canvas_rec = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Vẽ
        draw_pose_on_canvas(canvas_gt, pose_gt_scaled, is_gt=True)
        draw_pose_on_canvas(canvas_rec, pose_rec_scaled, is_gt=False)
        
        # Thêm nhãn
        cv2.putText(canvas_gt, "GROUND TRUTH", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(canvas_rec, "RECONSTRUCTED", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        cv2.putText(canvas_gt, f"Frame: {t}/{min_len}", (20, H-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Ghép 2 khung hình
        final_frame = np.hstack([canvas_gt, canvas_rec])
        out.write(final_frame)
        
    out.release()
    print(f"\n✅ Xong! Video đã lưu tại: {args.output_video}")

if __name__ == '__main__':
    main()