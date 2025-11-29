import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
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

# --- DEFINITIONS TỪ visualize_single_pose.py ---
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12), (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]
POSE_CONNECTIONS_UPPER_BODY = [
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8),
    (9, 10), (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24)
]
MOUTH_OUTER_LIP = list(zip(range(0, 11), range(1, 12))) + [(11, 0)]
MOUTH_INNER_LIP = list(zip(range(12, 19), range(13, 20))) + [(19, 12)]
MOUTH_CONNECTIONS_20 = MOUTH_OUTER_LIP + MOUTH_INNER_LIP

ALL_CONNECTIONS = []
ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 0, 'color': 'gray', 'lw': 2}
    for (s, e) in POSE_CONNECTIONS_UPPER_BODY
])
ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 33, 'color': 'blue', 'lw': 1.5}
    for (s, e) in HAND_CONNECTIONS
])
# Nối cổ tay vào body (dự đoán index 15/16 của body nối với gốc hand 0)
ALL_CONNECTIONS.append({'indices': (15, 0), 'offset': (0, 33), 'color': 'blue', 'lw': 2}) 
ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 54, 'color': 'green', 'lw': 1.5}
    for (s, e) in HAND_CONNECTIONS
])
ALL_CONNECTIONS.append({'indices': (16, 0), 'offset': (0, 54), 'color': 'green', 'lw': 2})
ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 75, 'color': 'red', 'lw': 1}
    for (s, e) in MOUTH_CONNECTIONS_20
])

# Indices để plot scatter
MANUAL_UPPER_BODY_IDXS = list(range(23))
LEFT_HAND_IDXS = list(range(33, 54))
RIGHT_HAND_IDXS = list(range(54, 75))
MOUTH_IDXS = list(range(75, 95))
PLOT_IDXS = MANUAL_UPPER_BODY_IDXS + LEFT_HAND_IDXS + RIGHT_HAND_IDXS + MOUTH_IDXS
VALID_POINT_THRESHOLD = 0.1

class SpecificSkeletonVisualizer:
    def prepare_data(self, pose_214):
        """Chuyển đổi từ [T, 214] sang format [T, 95, 2] dùng cho vẽ"""
        # Logic lấy từ load_and_prepare_pose của bạn
        manual_150 = pose_214[:, :150]
        manual_kps = manual_150.reshape(-1, 75, 2)
        
        # Bỏ qua đoạn giữa, lấy mouth
        mouth_40 = pose_214[:, 174:] 
        mouth_kps = mouth_40.reshape(-1, 20, 2)
        
        all_kps = np.concatenate([manual_kps, mouth_kps], axis=1) # [T, 95, 2]
        return all_kps

    def create_animation(self, real_pose_raw, gen_pose_raw, text, save_path):
        # 1. Prepare Data
        real_kps = self.prepare_data(real_pose_raw)
        gen_kps = self.prepare_data(gen_pose_raw)
        
        T = len(real_kps)
        
        # 2. Setup Figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
        fig.suptitle(f"Text: {text[:60]}...", fontsize=12)
        ax1.set_title("Ground Truth")
        ax2.set_title("Generated (Flow)")

        # Tính giới hạn khung hình dựa trên dữ liệu thật
        all_valid_points = real_kps[:, PLOT_IDXS]
        # Lọc điểm 0
        valid_mask = np.sum(np.abs(all_valid_points), axis=2) > VALID_POINT_THRESHOLD
        if valid_mask.any():
            valid_vals = all_valid_points[valid_mask]
            min_vals = np.min(valid_vals, axis=0)
            max_vals = np.max(valid_vals, axis=0)
            pad = 0.2
            
            # Set limit chung cho cả 2 hình
            for ax in [ax1, ax2]:
                ax.set_xlim(min_vals[0] - pad, max_vals[0] + pad)
                ax.set_ylim(max_vals[1] + pad, min_vals[1] - pad) # Invert Y axis
                ax.set_aspect('equal')
                ax.axis('off')
        else:
            for ax in [ax1, ax2]:
                ax.set_xlim(-1, 1); ax.set_ylim(1, -1); ax.axis('off')

        # 3. Setup Artists (Lines & Scatters) cho cả 2 axes
        def init_artists(ax):
            lines = []
            for item in ALL_CONNECTIONS:
                line = Line2D([], [], color=item['color'], lw=item['lw'], alpha=0.8)
                ax.add_line(line)
                lines.append({'line': line, 'item': item})
            
            scatter = ax.scatter([], [], s=2, c='black', alpha=0.4)
            return lines, scatter

        lines1, scat1 = init_artists(ax1)
        lines2, scat2 = init_artists(ax2)

        def update_frame(kps_frame, lines_dict, scat_obj):
            # Update Lines
            for obj in lines_dict:
                item = obj['item']
                line = obj['line']
                (s, e) = item['indices']
                offset = item['offset']
                
                if isinstance(offset, (tuple, list)):
                    idx_start, idx_end = s + offset[0], e + offset[1]
                else:
                    idx_start, idx_end = s + offset, e + offset
                
                p1 = kps_frame[idx_start]
                p2 = kps_frame[idx_end]
                
                # Check threshold để không vẽ đường về gốc 0,0
                if np.sum(np.abs(p1)) > VALID_POINT_THRESHOLD and np.sum(np.abs(p2)) > VALID_POINT_THRESHOLD:
                    line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                else:
                    line.set_data([], [])
            
            # Update Scatter
            points_to_plot = kps_frame[PLOT_IDXS]
            # Chỉ vẽ điểm valid
            valid_pts = points_to_plot[np.sum(np.abs(points_to_plot), axis=1) > VALID_POINT_THRESHOLD]
            scat_obj.set_offsets(valid_pts)
            
            return [obj['line'] for obj in lines_dict] + [scat_obj]

        def update(frame):
            artists1 = update_frame(real_kps[frame], lines1, scat1)
            
            # Xử lý độ dài lệch nhau (nếu Gen ngắn hơn Real)
            idx_gen = min(frame, len(gen_kps) - 1)
            artists2 = update_frame(gen_kps[idx_gen], lines2, scat2)
            
            return artists1 + artists2

        ani = animation.FuncAnimation(fig, update, frames=T, blit=True, interval=40)
        ani.save(save_path, writer='ffmpeg', fps=25)
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
        print("⚠️ No stats found, visual might be wrong scale.")
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

    # Visualizer MỚI
    visualizer = SpecificSkeletonVisualizer()

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
        
        # Inference
        # FIX: Gọi hàm _inference_forward với enable_grad để Sync Guidance hoạt động
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
        
        # Save
        save_path_base = os.path.join(args.output_dir, f"sample_{i}_{video_id}")
        
        # Save raw npz để debug thêm nếu cần
        np.savez(f"{save_path_base}.npz", real=real_pose_denorm, gen=gen_pose_denorm, text=text)

        try:
            # Truyền thẳng pose [T, 214] vào visualizer
            visualizer.create_animation(real_pose_denorm, gen_pose_denorm, text, f"{save_path_base}.mp4")
            print(f"   [{i+1}] 🎥 Video: {save_path_base}.mp4")
        except Exception as e:
            print(f"   ⚠️ Render error: {e}")
            import traceback
            traceback.print_exc()

    print("✅ Done!")

if __name__ == "__main__":
    main()