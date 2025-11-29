import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.collections import LineCollection
from torch.utils.data import DataLoader

# Thêm đường dẫn project
sys.path.append(os.getcwd())

try:
    from dataset import SignLanguageDataset, collate_fn
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher
except ImportError:
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from dataset import SignLanguageDataset, collate_fn
    from models.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher

def denormalize(pose, mean, std):
    """Đưa pose về giá trị gốc"""
    return pose * std + mean

class SkeletonVisualizer:
    def __init__(self, keypoint_dim=214):
        # Định nghĩa các kết nối (Indices nối với nhau). 
        # LƯU Ý: Đây là giả định cấu trúc 214 điểm thường gặp (Body + Face + Hands).
        # Nếu bộ dữ liệu của bạn khác, bạn có thể cần điều chỉnh các số này.
        
        # Giả định: Body (0-50?), Hand L, Hand R...
        # Để an toàn và đẹp, ta sẽ vẽ các đường nối cho Bàn Tay (thường chuẩn giống nhau)
        # và Thân mình cơ bản.
        
        # Cấu trúc bàn tay chuẩn (21 điểm mỗi tay):
        # Wrist -> Thumb(1-4), Index(5-8), Middle(9-12), Ring(13-16), Pinky(17-20)
        self.hand_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
            (0, 5), (5, 6), (6, 7), (7, 8),         # Index
            (0, 9), (9, 10), (10, 11), (11, 12),    # Middle
            (0, 13), (13, 14), (14, 15), (15, 16),  # Ring
            (0, 17), (17, 18), (18, 19), (19, 20)   # Pinky
        ]
        
        # Cấu trúc thân mình cơ bản (Body) - Giả định OpenPose/SMPL
        # Đây là ví dụ, có thể cần sửa tùy dataset
        self.body_connections = [
            (0, 1), (1, 2), (2, 3), (3, 4), # Spine/Neck?
            (2, 5), (5, 6), (6, 7),         # R Arm
            (2, 8), (8, 9), (9, 10)         # L Arm
        ]

    def create_animation(self, real_pose, gen_pose, text, save_path):
        """
        real_pose, gen_pose: [T, N_points, 2] (Dữ liệu đã reshape về 2D)
        """
        frames = len(real_pose)
        
        # Setup Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Prompt: {text[:60]}...", fontsize=12)
        
        ax1.set_title("Ground Truth")
        ax2.set_title("Generated (Flow)")
        
        # Tự động tính giới hạn khung hình để zoom vào người
        all_data = np.concatenate([real_pose, gen_pose], axis=0) # [2T, N, 2]
        # Loại bỏ điểm 0 (padding)
        valid_mask = (all_data.sum(axis=-1) != 0)
        if valid_mask.sum() > 0:
            valid_data = all_data[valid_mask]
            min_x, max_x = valid_data[:, 0].min(), valid_data[:, 0].max()
            min_y, max_y = valid_data[:, 1].min(), valid_data[:, 1].max()
            
            # Thêm padding
            pad = 0.1
            ax1.set_xlim(min_x - pad, max_x + pad)
            ax1.set_ylim(max_y + pad, min_y - pad) # Đảo ngược trục Y cho ảnh pose
            ax2.set_xlim(min_x - pad, max_x + pad)
            ax2.set_ylim(max_y + pad, min_y - pad)
        else:
            ax1.set_xlim(-1, 1); ax1.set_ylim(1, -1)
            ax2.set_xlim(-1, 1); ax2.set_ylim(1, -1)

        # Plot elements
        # Scatter cho các khớp
        scat1 = ax1.scatter([], [], s=10, c='black', zorder=2)
        scat2 = ax2.scatter([], [], s=10, c='black', zorder=2)
        
        # Lines cho xương (dùng LineCollection cho nhanh)
        lines1_body = LineCollection([], colors='gray', linewidths=2)
        lines1_lhand = LineCollection([], colors='green', linewidths=1.5) # Left Hand Green (giống ảnh mẫu)
        lines1_rhand = LineCollection([], colors='blue', linewidths=1.5)  # Right Hand Blue
        
        lines2_body = LineCollection([], colors='gray', linewidths=2)
        lines2_lhand = LineCollection([], colors='green', linewidths=1.5)
        lines2_rhand = LineCollection([], colors='blue', linewidths=1.5)
        
        # Add collections
        for ax, lines in zip([ax1, ax2], 
                             [[lines1_body, lines1_lhand, lines1_rhand], 
                              [lines2_body, lines2_lhand, lines2_rhand]]):
            for l in lines: ax.add_collection(l)

        def get_lines(pose_frame):
            # Hàm này cần logic cắt index cụ thể của dataset bạn
            # Giả định format phổ biến: 
            # Body (0-50?), Hand_L (range A), Hand_R (range B)
            # Vì mình không biết index chính xác, mình sẽ thử đoán dựa trên 214 điểm
            # Nếu 214 = 70 (body+face) + 21 (L) + 21 (R) + ...?
            
            # TẠM THỜI: Vẽ scatter màu mè phân biệt tay/người trước nếu chưa biết nối
            # Nhưng để thử nối tay (thường nằm ở cuối hoặc cụm 21 điểm)
            
            # Giả sử 21 điểm cuối là Tay Phải, 21 điểm sát cuối là Tay Trái
            # Cần chỉnh lại offset này nếu dataset khác
            n_points = pose_frame.shape[0]
            
            # Hand Indices Assumption (Thay đổi số này nếu cần)
            idx_r_hand_start = n_points - 21
            idx_l_hand_start = n_points - 42 
            
            segments_l = []
            segments_r = []
            
            # Create Hand Segments
            for (start, end) in self.hand_connections:
                # Left Hand
                p1_l = pose_frame[idx_l_hand_start + start]
                p2_l = pose_frame[idx_l_hand_start + end]
                if np.sum(p1_l) != 0 and np.sum(p2_l) != 0: # Bỏ qua điểm 0
                    segments_l.append([p1_l, p2_l])
                
                # Right Hand
                p1_r = pose_frame[idx_r_hand_start + start]
                p2_r = pose_frame[idx_r_hand_start + end]
                if np.sum(p1_r) != 0 and np.sum(p2_r) != 0:
                    segments_r.append([p1_r, p2_r])

            return [], segments_l, segments_r # Body để trống nếu chưa biết index

        def update(frame):
            # REAL
            p1 = real_pose[frame]
            scat1.set_offsets(p1)
            _, l_segs1, r_segs1 = get_lines(p1)
            lines1_lhand.set_segments(l_segs1)
            lines1_rhand.set_segments(r_segs1)
            
            # GEN
            if frame < len(gen_pose):
                p2 = gen_pose[frame]
                scat2.set_offsets(p2)
                _, l_segs2, r_segs2 = get_lines(p2)
                lines2_lhand.set_segments(l_segs2)
                lines2_rhand.set_segments(r_segs2)
            
            return scat1, scat2, lines1_lhand, lines1_rhand, lines2_lhand, lines2_rhand

        ani = animation.FuncAnimation(fig, update, frames=frames, interval=40, blit=True)
        ani.save(save_path, fps=25, extra_args=['-vcodec', 'libx264'])
        plt.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--text_file", type=str, required=True)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--flow_ckpt", type=str, required=True)
    parser.add_argument("--ae_ckpt", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=5)
    parser.add_argument("--output_dir", type=str, default="eval_results")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--ae_hidden_dim", type=int, default=512)
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Evaluation started...")

    # Load Stats
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        mean, std = 0, 1
    else:
        stats = np.load(stats_path)
        mean = stats['mean']
        std = stats['std']
    
    dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        split=args.split,
        text_file=args.text_file,
        max_seq_len=200
    )
    
    # Load Models
    ae = UnifiedPoseAutoencoder(latent_dim=args.latent_dim, hidden_dim=args.ae_hidden_dim).to(device)
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device)['model_state_dict'])
    ae.eval()
    
    flow_matcher = LatentFlowMatcher(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    flow_ckpt = torch.load(args.flow_ckpt, map_location=device)
    flow_matcher.load_state_dict(flow_ckpt['model_state_dict'], strict=False)
    flow_matcher.eval()
    
    latent_scale = float(flow_ckpt.get("latent_scale_factor", 1.0))
    print(f"📏 Scale: {latent_scale:.4f}")

    # Visualizer
    visualizer = SkeletonVisualizer()

    # Get random indices
    indices = np.random.choice(len(dataset), size=min(len(dataset), args.num_samples), replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, collate_fn=collate_fn)
    
    print(f"🎬 Generating {len(subset)} samples...")
    
    for i, batch in enumerate(loader):
        video_id = batch['video_ids'][0]
        text = batch['texts'][0]
        pose_gt = batch['poses'][0].cpu().numpy()
        
        text_tokens = batch['text_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # Encode Text
        text_features, text_mask = flow_matcher.encode_text(text_tokens, attention_mask)
        
        # Inference (Fix lỗi batch=None trước đó)
        # Sử dụng hàm _inference_forward đã sửa có enable_grad
        gen_latent = flow_matcher._inference_forward(
            batch=None, 
            text_features=text_features, 
            text_mask=text_mask,
            num_steps=50
        )
        
        gen_latent_scaled = gen_latent * latent_scale
        gen_pose = ae.decode(gen_latent_scaled)
        gen_pose = gen_pose.squeeze(0).detach().cpu().numpy()
        
        # Denormalize
        real_pose_denorm = denormalize(pose_gt, mean, std)
        gen_pose_denorm = denormalize(gen_pose, mean, std)
        
        # RESHAPE TỪ [T, 214] -> [T, 107, 2] (Giả định 2D)
        # Nếu data 214 điểm là (X,Y) phẳng
        try:
            real_pose_reshaped = real_pose_denorm.reshape(real_pose_denorm.shape[0], -1, 2)
            gen_pose_reshaped = gen_pose_denorm.reshape(gen_pose_denorm.shape[0], -1, 2)
        except:
            print(f"⚠️ Cannot reshape pose {real_pose_denorm.shape}. Skipping visualization.")
            continue

        # Save
        save_path_base = os.path.join(args.output_dir, f"sample_{i}_{video_id}")
        
        try:
            visualizer.create_animation(real_pose_reshaped, gen_pose_reshaped, text, f"{save_path_base}.mp4")
            print(f"   [{i+1}] 🎥 Video: {save_path_base}.mp4")
        except Exception as e:
            print(f"   ⚠️ Render error: {e}")

    print("✅ Done!")

if __name__ == "__main__":
    main()