import numpy as np
import cv2
import argparse
import os
from tqdm import tqdm

# --- 1. ĐỊNH NGHĨA KHUNG XƯƠNG (PHOENIX-14T) ---
# Body (0-24) nhưng chỉ vẽ các khớp chính
BODY_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Spine
    (1, 5), (5, 6), (6, 7),              # Left Arm
    (1, 8), (8, 9), (9, 10),             # Right Arm
    (0, 11), (11, 12), (12, 13),         # Left Leg
    (0, 14), (14, 15), (15, 16),         # Right Leg
    (0, 17), (17, 18), (18, 19), (19, 20) # Head
]

# Hands (21 points each)
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),      # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),      # Index
    (0, 9), (9, 10), (10, 11), (11, 12), # Middle
    (0, 13), (13, 14), (14, 15), (15, 16), # Ring
    (0, 17), (17, 18), (18, 19), (19, 20)  # Pinky
]

def smart_scale_coords(coords, width, height):
    """
    Tự động scale tọa độ:
    - Nếu max < 1.5 -> Dữ liệu đang Norm (0-1) -> Nhân với W, H
    - Nếu max > 1.5 -> Dữ liệu đang Pixel -> Giữ nguyên
    """
    coords_scaled = coords.copy()
    max_val = np.max(np.abs(coords))
    
    if max_val <= 1.5:
        # Data đang ở dạng [0, 1], cần scale lên
        coords_scaled[:, 0] = coords[:, 0] * width
        coords_scaled[:, 1] = coords[:, 1] * height
    else:
        # Data có vẻ đã là pixel, giữ nguyên (hoặc resize nếu cần)
        pass
        
    return coords_scaled.astype(np.int32)

def draw_skeleton(frame, keypoints, color=(0, 255, 0), thickness=2):
    """Vẽ skeleton lên frame đen"""
    # Tách các bộ phận (Dựa trên 75 keypoints chuẩn)
    # [0-24]: Body, [25-45]: Left Hand, [46-66]: Right Hand
    body = keypoints[:25]
    left_hand = keypoints[25:46]
    right_hand = keypoints[46:67]
    
    # 1. Draw Body
    for i, j in BODY_CONNECTIONS:
        if i < len(body) and j < len(body):
            pt1, pt2 = body[i], body[j]
            # Chỉ vẽ nếu điểm không phải (0,0)
            if np.sum(pt1) > 0 and np.sum(pt2) > 0:
                cv2.line(frame, tuple(pt1), tuple(pt2), color, thickness)
                
    # 2. Draw Hands (Mảnh hơn chút cho đẹp)
    hand_thickness = max(1, thickness - 1)
    
    # Left Hand
    for i, j in HAND_CONNECTIONS:
        if i < len(left_hand) and j < len(left_hand):
            pt1, pt2 = left_hand[i], left_hand[j]
            if np.sum(pt1) > 0 and np.sum(pt2) > 0:
                cv2.line(frame, tuple(pt1), tuple(pt2), color, hand_thickness)

    # Right Hand
    for i, j in HAND_CONNECTIONS:
        if i < len(right_hand) and j < len(right_hand):
            pt1, pt2 = right_hand[i], right_hand[j]
            if np.sum(pt1) > 0 and np.sum(pt2) > 0:
                cv2.line(frame, tuple(pt1), tuple(pt2), color, hand_thickness)

    return frame

def create_comparison_video(original_path, reconstructed_path, output_video, fps=25):
    print("📂 Đang load dữ liệu...")
    original = np.load(original_path)       # [T, 214]
    reconstructed = np.load(reconstructed_path) # [T, 214]
    
    T = len(original)
    
    # Chỉ lấy 150 chiều đầu (Pose 75x2), bỏ qua NMM (Facial) lúc vẽ xương
    # Reshape về [T, 75, 2]
    original_kp = original[:, :150].reshape(T, 75, 2)
    recon_kp = reconstructed[:, :150].reshape(T, 75, 2)
    
    # Cấu hình Video
    H, W = 512, 512 # Độ phân giải hiển thị
    frame_size = (W * 2, H) # Side-by-side
    
    # Codec cho MP4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_video, fourcc, fps, frame_size)
    
    print(f"🎬 Đang render video {T} frames...")
    print(f"   Mode: {'Normalized [0-1]' if np.max(original_kp) <= 1.5 else 'Pixel Coords'}")

    for t in tqdm(range(T)):
        # Tạo nền đen
        frame_orig = np.zeros((H, W, 3), dtype=np.uint8)
        frame_recon = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Scale tọa độ cho khớp với khung hình H, W
        kp_orig = smart_scale_coords(original_kp[t], W, H)
        kp_recon = smart_scale_coords(recon_kp[t], W, H)
        
        # Vẽ Skeleton
        # Gốc: Màu Xanh Lá (Green)
        draw_skeleton(frame_orig, kp_orig, color=(0, 255, 0))
        # Tái tạo: Màu Đỏ (Red)
        draw_skeleton(frame_recon, kp_recon, color=(0, 0, 255))
        
        # Thêm nhãn (Labels)
        cv2.putText(frame_orig, "ORIGINAL (GT)", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(frame_recon, "RECONSTRUCTED", (20, 40), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Số Frame
        cv2.putText(frame_orig, f"Frame: {t}/{T}", (20, H-20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        # Ghép 2 ảnh lại (Trái - Phải)
        combined = np.hstack((frame_orig, frame_recon))
        out.write(combined)
        
    out.release()
    print(f"\n✅ Xong! Video lưu tại: {output_video}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--original', type=str, required=True, help='File .npy gốc')
    parser.add_argument('--reconstructed', type=str, required=True, help='File .npy tái tạo')
    parser.add_argument('--output_video', type=str, default='comparison.mp4')
    parser.add_argument('--fps', type=int, default=25)
    
    args = parser.parse_args()
    
    if not os.path.exists(args.original):
        print(f"❌ Không tìm thấy file gốc: {args.original}")
        return
        
    create_comparison_video(args.original, args.reconstructed, args.output_video, args.fps)

if __name__ == '__main__':
    main()