import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sys

# Thêm đường dẫn để import module của chị
sys.path.append(os.getcwd())

try:
    from data_preparation import load_sample
    # Import cấu trúc xương để vẽ
    from visualize_single_pose import HAND_CONNECTIONS, POSE_CONNECTIONS_UPPER_BODY, FACE_CONNECTIONS
except ImportError:
    print("⚠️ Cảnh báo: Không import được cấu trúc xương. Vẽ point cloud đơn giản.")
    HAND_CONNECTIONS = []
    POSE_CONNECTIONS_UPPER_BODY = []
    FACE_CONNECTIONS = []

def visualize_comparison(real_pose, gen_pose, output_path, title_text=""):
    """
    Tạo video so sánh: Bên trái (Real) - Bên phải (Generated)
    """
    # Cắt ngắn về độ dài chung nhỏ nhất để so sánh
    min_len = min(len(real_pose), len(gen_pose))
    real_pose = real_pose[:min_len]
    gen_pose = gen_pose[:min_len]
    
    print(f"🎬 Đang tạo video so sánh ({min_len} frames)...")
    
    # Setup Plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Config trục (Giả sử pose đã denormalize về pixel 256x256 hoặc tương tự)
    # Chị có thể cần chỉnh limit này tùy theo scale data của chị
    for ax in [ax1, ax2]:
        ax.set_xlim(0, 260) # Chiều rộng ảnh gốc Phoenix
        ax.set_ylim(210, 0) # Chiều cao ảnh gốc (đảo ngược trục y)
        ax.axis('off')

    ax1.set_title("Ground Truth (Real)")
    ax2.set_title("Generated (AI)")
    fig.suptitle(f"Compare: {title_text}", fontsize=10)

    # Init artists
    lines_real = []
    lines_gen = []
    scats_real = []
    scats_gen = []

    # Hàm vẽ helper
    def init_skeleton(ax, collection_lines, collection_scats):
        # Vẽ body (Line)
        for _ in range(len(POSE_CONNECTIONS_UPPER_BODY) + len(HAND_CONNECTIONS)):
            line, = ax.plot([], [], 'k-', linewidth=1) # Đen
            collection_lines.append(line)
        # Vẽ face (Scatter cho nhẹ)
        scat = ax.scatter([], [], s=2, c='r') # Đỏ
        collection_scats.append(scat)

    init_skeleton(ax1, lines_real, scats_real)
    init_skeleton(ax2, lines_gen, scats_gen)

    def update(frame):
        # Lấy frame hiện tại
        pose_r = real_pose[frame].reshape(-1, 2) # [214/2, 2]
        pose_g = gen_pose[frame].reshape(-1, 2)
        
        # Update function cho 1 bên
        def update_ax(pose_data, lines, scats):
            # Tách các phần
            # Indices (Hardcode theo Mediapipe Holistic rút gọn của chị)
            # Coarse: 0-33, Face: ...
            # Để đơn giản, vẽ hết các kết nối có sẵn
            
            line_idx = 0
            
            # 1. Vẽ Body & Hands (Dùng Line)
            # Cần map lại index từ 214 vector sang index của skeleton map
            # Giả sử pose_data đã đúng thứ tự extraction
            
            # Note: Chị cần đảm bảo index trong VISUALIZE_SINGLE_POSE khớp với data 214
            # Ở đây em vẽ point cloud nếu map không khớp, hoặc thử vẽ line cơ bản
            
            # Vẽ Line Body
            for i, (start, end) in enumerate(POSE_CONNECTIONS_UPPER_BODY):
                if start < len(pose_data) and end < len(pose_data):
                    if pose_data[start].sum() != 0 and pose_data[end].sum() != 0:
                        lines[line_idx].set_data([pose_data[start, 0], pose_data[end, 0]],
                                                 [pose_data[start, 1], pose_data[end, 1]])
                    else:
                        lines[line_idx].set_data([], [])
                    line_idx += 1
            
            # Vẽ Hands
            # (Cần offset index nếu tay nằm sau body trong mảng 214)
            # Tạm thời vẽ scatter toàn bộ cho chắc ăn nếu index loạn
            scats[0].set_offsets(pose_data) # Vẽ tất cả điểm dạng chấm đỏ

        update_ax(pose_r, lines_real, scats_real)
        update_ax(pose_g, lines_gen, scats_gen)
        
        return lines_real + scats_real + lines_gen + scats_gen

    ani = animation.FuncAnimation(fig, update, frames=min_len, blit=True, interval=40)
    ani.save(output_path, writer='ffmpeg', fps=25)
    print(f"✅ Xong! Video lưu tại: {output_path}")
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gen_path', type=str, required=True, help="File .npy sinh ra từ inference")
    parser.add_argument('--data_dir', type=str, required=True, help="Thư mục chứa data gốc (processed_data/data)")
    parser.add_argument('--video_id', type=str, required=True, help="ID của video gốc (VD: 27January...)")
    parser.add_argument('--split', type=str, default='train', help="Video gốc nằm ở split nào (train/dev/test)")
    parser.add_argument('--output_video', type=str, default='compare.mp4')
    args = parser.parse_args()

    # 1. Load Generated Pose
    print(f"📂 Loading Gen: {args.gen_path}")
    gen_pose = np.load(args.gen_path)
    
    # 2. Load Real Pose
    split_dir = os.path.join(args.data_dir, args.split)
    print(f"📂 Loading Real ID: {args.video_id} từ {split_dir}")
    
    # Dùng hàm load_sample có sẵn của chị để load chuẩn
    real_pose, T = load_sample(args.video_id, split_dir)
    
    if real_pose is None:
        print("❌ Không tìm thấy file pose gốc! Kiểm tra lại ID hoặc Split.")
        sys.exit(1)
        
    # 3. So sánh
    print(f"📊 Stats:")
    print(f"   - Gen Shape: {gen_pose.shape}")
    print(f"   - Real Shape: {real_pose.shape}")
    
    # Tạo video
    visualize_comparison(real_pose, gen_pose, args.output_video, args.video_id)

if __name__ == '__main__':
    main()