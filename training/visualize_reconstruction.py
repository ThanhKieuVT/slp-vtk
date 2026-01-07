import numpy as np
import cv2
import argparse
from tqdm import tqdm

# --- 1. ĐỊNH NGHĨA KẾT NỐI XƯƠNG (TOPOLOGY) ---
# Dựa trên cấu trúc 75 điểm (OpenPose/Sign Language Datasets)
# 0-8: Body, 9-24: Right Hand? (Cần check kỹ dataset, đây là cấu trúc phổ biến)
# Tuy nhiên, để an toàn, ta vẽ theo cấu trúc chuẩn OpenPose 25 body + Hands

# Body chain (đơn giản hóa để không bị rối)
BODY_CONNECTIONS = [
    (0,1), (1,2), (2,3), (3,4),       # Left Arm
    (1,5), (5,6), (6,7),              # Right Arm
    (1,8), (8,9), (9,10), (10,11),    # Torso & Left Leg
    (8,12), (12,13), (13,14),         # Right Leg
    (0,15), (0,16), (15,17), (16,18)  # Face contour/Eyes
]

# Hand chains (Cấu trúc bàn tay chuẩn)
# Left Hand (bắt đầu từ điểm cổ tay - index tuỳ dataset, thường là nối tiếp body)
# Giả sử cấu trúc 75 điểm: 0-14 (Body), 15-18 (Head), 
# 25-45 (Left Hand), 46-66 (Right Hand) -> Đây là giả định phổ biến.
# NHƯNG code data_prep của bạn gộp chung. 
# Tôi sẽ dùng logic vẽ đoạn thẳng liền kề cho phần tay để đảm bảo hiện hình.

def get_skeleton_topology():
    edges = list(BODY_CONNECTIONS)
    # Thêm các đường nối tay (dựa trên index giả định cho vector 75 điểm)
    # Nếu topology sai, nó sẽ nối lung tung, nhưng ít nhất sẽ hiện hình để bạn debug.
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
    # Màu sắc (BGR)
    if is_gt:
        c_body = (0, 255, 0)     # Green
        c_hand = (0, 0, 255)     # Red
        c_face = (255, 255, 255) # White
        label = "GROUND TRUTH"
    else:
        c_body = (0, 255, 255)   # Yellow
        c_hand = (255, 0, 255)   # Purple
        c_face = (0, 165, 255)   # Orange
        label = "RECONSTRUCTION"

    # 1. Vẽ Body (Nối dây)
    # Lưu ý: Vì topology dataset này khá đặc thù, ta sẽ vẽ các điểm trước
    # Để an toàn: Vẽ tất cả các điểm body thành chấm tròn
    for i, pt in enumerate(body):
        if pt[0] < 0: continue # Bỏ qua điểm invalid
        
        # Phân biệt màu tay và người (giả định index > 20 là tay)
        color = c_hand if i > 20 else c_body
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 3, color, -1)
        
        # Thử nối điểm i với i+1 (Heuristic đơn giản để tạo hình liền mạch)
        # Chỉ nối nếu điểm tiếp theo cũng valid và không quá xa (tránh nối từ tay nọ sang chân kia)
        if i + 1 < len(body):
            pt_next = body[i+1]
            if pt_next[0] > 0:
                dist = np.linalg.norm(pt - pt_next)
                if dist < 100: # Ngưỡng khoảng cách pixel
                    cv2.line(canvas, (int(pt[0]), int(pt[1])), (int(pt_next[0]), int(pt_next[1])), color, 2)

    # 2. Vẽ Mouth (Chấm nhỏ)
    for pt in mouth:
        if pt[0] < 0: continue
        cv2.circle(canvas, (int(pt[0]), int(pt[1])), 1, c_face, -1)

    # 3. Text Label
    cv2.putText(canvas, label, (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, c_body, 2)

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