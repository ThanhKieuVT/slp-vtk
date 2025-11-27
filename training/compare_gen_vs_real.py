"""
Script so sánh Real vs Generated Pose (Phiên bản Chuẩn Xác)
Chức năng:
1. Load pose thật (Real) từ ID và pose giả (Gen) từ file .npy
2. Tách bỏ các thành phần phi tọa độ (AUs, Head, Gaze) trong vector 214 chiều.
3. Vẽ video so sánh dạng Point Cloud (Chấm điểm).
"""
import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Thêm đường dẫn hiện tại để import module
sys.path.append(os.getcwd())

try:
    # Cố gắng import hàm load chuẩn từ file của chị
    from data_preparation import load_sample
except ImportError:
    print("❌ Lỗi: Không tìm thấy file 'data_preparation.py'.")
    print("💡 Hãy đảm bảo chị chạy lệnh từ thư mục gốc của project.")
    sys.exit(1)

# === 1. HÀM TÁCH TỌA ĐỘ (CỐT LÕI) ===
def extract_visual_coordinates(pose_214):
    """
    Input: [T, 214] vector chứa lung tung cả tọa độ lẫn số đo.
    Output: [T, N_points, 2] chỉ chứa tọa độ X,Y để vẽ.
    
    Cấu trúc vector 214 chiều (dựa trên data_preparation.py):
    - 0   -> 150: Body + Hands (75 điểm x 2 chiều) -> LẤY
    - 150 -> 167: Facial AUs (17 số)               -> BỎ
    - 167 -> 170: Head Pose (3 số)                 -> BỎ
    - 170 -> 174: Eye Gaze (4 số)                  -> BỎ
    - 174 -> 214: Mouth (20 điểm x 2 chiều)        -> LẤY
    """
    # 1. Lấy phần Body + Hands
    body_hands_flat = pose_214[:, :150] # [T, 150]
    
    # 2. Lấy phần Mouth
    mouth_flat = pose_214[:, 174:214]   # [T, 40]
    
    # 3. Gộp lại
    visual_flat = np.concatenate([body_hands_flat, mouth_flat], axis=1) # [T, 190]
    
    # 4. Reshape thành tọa độ (X, Y)
    # Tổng số điểm = 190 / 2 = 95 điểm
    visual_points = visual_flat.reshape(len(pose_214), -1, 2) # [T, 95, 2]
    
    return visual_points

# === 2. HÀM VẼ VIDEO SO SÁNH ===
def create_comparison_video(real_pose_raw, gen_pose_raw, output_path, video_id):
    print(f"🔄 Đang xử lý data cho ID: {video_id}")
    
    # 1. Cắt độ dài cho bằng nhau
    min_len = min(len(real_pose_raw), len(gen_pose_raw))
    real_raw = real_pose_raw[:min_len]
    gen_raw = gen_pose_raw[:min_len]
    
    # 2. Trích xuất tọa độ sạch
    real_data = extract_visual_coordinates(real_raw)
    gen_data = extract_visual_coordinates(gen_raw)
    
    print(f"🎬 Đang render video ({min_len} frames)...")
    
    # 3. Setup khung hình (2 bên)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Tính giới hạn khung hình (Scale) dựa trên dữ liệu thật
    all_x = real_data[:, :, 0].flatten()
    all_y = real_data[:, :, 1].flatten()
    
    # Lọc bỏ điểm 0 (padding) để tính scale chuẩn
    valid_mask = (all_x > 0.01) & (all_y > 0.01)
    
    if valid_mask.sum() > 0:
        x_min, x_max = all_x[valid_mask].min(), all_x[valid_mask].max()
        y_min, y_max = all_y[valid_mask].min(), all_y[valid_mask].max()
    else:
        # Fallback nếu data lỗi
        x_min, x_max, y_min, y_max = 0, 1, 0, 1
        
    # Nới rộng viền tí cho đẹp
    margin_w = (x_max - x_min) * 0.1
    margin_h = (y_max - y_min) * 0.1
    
    for ax in [ax1, ax2]:
        ax.set_xlim(x_min - margin_w, x_max + margin_w)
        ax.set_ylim(y_max + margin_h, y_min - margin_h) # Đảo trục Y (ảnh gốc gốc toạ độ ở trên cùng)
        ax.axis('off') # Tắt trục số
        
    ax1.set_title("REAL (Ground Truth)", color='darkred', fontweight='bold')
    ax2.set_title("GENERATED (AI)", color='darkblue', fontweight='bold')
    fig.suptitle(f"Video ID: {video_id}", fontsize=10)
    
    # 4. Init Artists (Các chấm điểm)
    # Real = Đỏ, Gen = Xanh
    scat_real = ax1.scatter([], [], s=15, c='red', alpha=0.7, edgecolors='none')
    scat_gen = ax2.scatter([], [], s=15, c='blue', alpha=0.7, edgecolors='none')
    
    def update(frame):
        # Lấy frame t
        p_real = real_data[frame]
        p_gen = gen_data[frame]
        
        # Lọc bỏ các điểm rác (tọa độ 0,0 do padding hoặc missing)
        # Điểm hợp lệ là điểm có tổng trị tuyệt đối > 0
        mask_r = (np.abs(p_real).sum(axis=1) > 1e-3)
        mask_g = (np.abs(p_gen).sum(axis=1) > 1e-3)
        
        scat_real.set_offsets(p_real[mask_r])
        scat_gen.set_offsets(p_gen[mask_g])
        
        return scat_real, scat_gen

    # Render
    ani = animation.FuncAnimation(fig, update, frames=min_len, blit=True, interval=50)
    ani.save(output_path, writer='ffmpeg', fps=20)
    print(f"✅ HOÀN TẤT! Video lưu tại: {output_path}")
    plt.close()

# === 3. MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', type=str, required=True, help="Đường dẫn file .npy sinh ra từ model")
    parser.add_argument('--data_dir', type=str, required=True, help="Thư mục chứa data gốc (processed_data/data)")
    parser.add_argument('--video_id', type=str, required=True, help="Tên ID của video gốc để so sánh")
    parser.add_argument('--split', type=str, default='train', help="Split chứa video gốc (train/dev/test)")
    parser.add_argument('--output_video', type=str, default='compare_final.mp4', help="Tên file video đầu ra")
    
    args = parser.parse_args()
    
    # 1. Load Gen
    if not os.path.exists(args.gen_path):
        print(f"❌ Không tìm thấy file Gen: {args.gen_path}")
        return
    gen_pose = np.load(args.gen_path)
    print(f"📂 Loaded Gen Pose: {gen_pose.shape}")
    
    # 2. Load Real
    split_dir = os.path.join(args.data_dir, args.split)
    if not os.path.exists(split_dir):
        print(f"❌ Không tìm thấy thư mục split: {split_dir}")
        return
        
    print(f"📂 Loading Real ID: {args.video_id}...")
    real_pose, T = load_sample(args.video_id, split_dir)
    
    if real_pose is None:
        print("❌ Không load được Pose gốc! Kiểm tra lại ID hoặc đường dẫn Data.")
        return
    
    print(f"📊 Real Pose Shape: {real_pose.shape}")
    
    # 3. Tạo Video
    create_comparison_video(real_pose, gen_pose, args.output_video, args.video_id)

if __name__ == '__main__':
    main()