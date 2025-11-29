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

# --- TOPOLOGY CHUẨN CỦA MEDIAPIPE HOLISTIC ---
# Body (Pose Landmarks 0-32)
# Chú ý: MediaPipe Pose có cấu trúc cụ thể. Ta sẽ vẽ các đường chính.
BODY_CONNECTIONS = [
    (11, 12), (11, 23), (12, 24), (23, 24), # Thân
    (11, 13), (13, 15), (15, 21), (15, 17), (15, 19), (17, 19), # Tay trái (cánh tay)
    (12, 14), (14, 16), (16, 22), (16, 18), (16, 20), (18, 20), # Tay phải (cánh tay)
    (0, 1), (1, 2), (2, 3), (3, 7), (0, 4), (4, 5), (5, 6), (6, 8), # Mặt (sơ bộ)
    (9, 10) # Miệng (sơ bộ)
]

# Hand (0-20) - Chuẩn MediaPipe Hand
HAND_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),         # Thumb
    (0, 5), (5, 6), (6, 7), (7, 8),         # Index
    (5, 9), (9, 10), (10, 11), (11, 12),    # Middle
    (9, 13), (13, 14), (14, 15), (15, 16),  # Ring
    (13, 17), (17, 18), (18, 19), (19, 20), # Pinky
    (0, 17) # Palm base
]

# Mouth (20 points from Face Mesh) - Ta vẽ vòng tròn
MOUTH_CONNECTIONS = list(zip(range(0, 19), range(1, 20))) + [(19, 0)]

# --- TỔNG HỢP KẾT NỐI CHO VISUALIZER ---
ALL_CONNECTIONS = []

# 1. Body (Indices 0-32)
ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 0, 'color': 'gray', 'lw': 2}
    for (s, e) in BODY_CONNECTIONS
])

# 2. Left Hand (Indices 33-53) -> Offset 33
ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 33, 'color': 'green', 'lw': 1.5}
    for (s, e) in HAND_CONNECTIONS
])
# Nối cổ tay trái (Body 15) với gốc bàn tay trái (Hand 0 -> idx 33)
ALL_CONNECTIONS.append({'indices': (15, 33), 'offset': 0, 'color': 'green', 'lw': 2, 'type': 'link'})

# 3. Right Hand (Indices 54-74) -> Offset 54
ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 54, 'color': 'blue', 'lw': 1.5}
    for (s, e) in HAND_CONNECTIONS
])
# Nối cổ tay phải (Body 16) với gốc bàn tay phải (Hand 0 -> idx 54)
ALL_CONNECTIONS.append({'indices': (16, 54), 'offset': 0, 'color': 'blue', 'lw': 2, 'type': 'link'})

# 4. Mouth (Indices 174-213 -> 40 values -> 20 points) -> Offset trong array vẽ (sau khi lọc)
# Lưu ý: Ta sẽ xử lý riêng phần Mouth vì nó nằm tít ở index 174 (cách xa đám trên)

# --- INDICES ĐỂ VẼ ---
# Ta sẽ tạo một array rút gọn chỉ chứa các điểm toạ độ (x,y) để vẽ
# 1. Body: 0-32 (33 points)
# 2. LHand: 33-53 (21 points)
# 3. RHand: 54-74 (21 points)
# --- SKIP 150-173 (Non-coordinate features) ---
# 4. Mouth: 174-213 (20 points x 2)

VALID_POINT_THRESHOLD = 0.01 # Lọc điểm (0,0)

class CorrectSkeletonVisualizer:
    def prepare_data(self, pose_214):
        """
        Input: [T, 214]
        Output: [T, 95, 2] containing only coordinates (Body+Hands+Mouth)
        """
        T = pose_214.shape[0]
        
        # 1. Manual Parts (Body + Hands): Index 0-149 -> 75 points
        manual_part = pose_214[:, :150].reshape(T, 75, 2)
        
        # 2. Mouth Part: Index 174-213 -> 40 values -> 20 points
        mouth_part = pose_214[:, 174:].reshape(T, 20, 2)
        
        # Ghép lại: 75 + 20 = 95 points
        # Index mới:
        # 0-32: Body
        # 33-53: LHand
        # 54-74: RHand
        # 75-94: Mouth
        clean_pose = np.concatenate([manual_part, mouth_part], axis=1)
        return clean_pose

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

        # Tính giới hạn khung hình (Dựa trên Body points để ổn định)
        body_points = real_kps[:, :33] # Chỉ lấy body để tính scale
        valid_mask = np.sum(np.abs(body_points), axis=2) > VALID_POINT_THRESHOLD
        if valid_mask.any():
            valid_vals = body_points[valid_mask]
            min_vals = np.min(valid_vals, axis=0)
            max_vals = np.max(valid_vals, axis=0)
            pad = 0.1
            
            for ax in [ax1, ax2]:
                ax.set_xlim(min_vals[0] - pad, max_vals[0] + pad)
                ax.set_ylim(max_vals[1] + pad, min_vals[1] - pad) # Invert Y
                ax.set_aspect('equal')
                ax.axis('off')
        else:
            for ax in [ax1, ax2]:
                ax.set_xlim(0, 1); ax.set_ylim(1, 0); ax.axis('off')

        # 3. Setup Lines
        def init_lines(ax):
            lines = []
            
            # Body & Hands
            for item in ALL_CONNECTIONS:
                line = Line2D([], [], color=item['color'], lw=item['lw'], alpha=0.8)
                ax.add_line(line)
                lines.append({'line': line, 'item': item})
            
            # Mouth (Offset 75 trong mảng clean_pose)
            for (s, e) in MOUTH_CONNECTIONS:
                line = Line2D([], [], color='red', lw=1.5, alpha=0.8)
                ax.add_line(line)
                lines.append({'line': line, 'item': {'indices': (s, e), 'offset': 75}})
                
            return lines

        lines1 = init_lines(ax1)
        lines2 = init_lines(ax2)

        def update_frame(kps_frame, lines_list):
            for obj in lines_list:
                item = obj['item']
                line = obj['line']
                (s, e) = item['indices']
                
                # Tính index thực tế
                if 'type' in item and item['type'] == 'link':
                    # Trường hợp nối đặc biệt (vd: Cổ tay -> Bàn tay)
                    # item['indices'] là index tuyệt đối trong mảng 95
                    idx_start, idx_end = s, e
                else:
                    # Trường hợp offset thường
                    offset = item['offset']
                    idx_start, idx_end = s + offset, e + offset
                
                p1 = kps_frame[idx_start]
                p2 = kps_frame[idx_end]
                
                # Check threshold (0,0)
                if np.sum(np.abs(p1)) > VALID_POINT_THRESHOLD and np.sum(np.abs(p2)) > VALID_POINT_THRESHOLD:
                    line.set_data([p1[0], p2[0]], [p1[1], p2[1]])
                else:
                    line.set_data([], [])
            return [obj['line'] for obj in lines_list]

        def update(frame):
            artists1 = update_frame(real_kps[frame], lines1)
            idx_gen = min(frame, len(gen_kps) - 1)
            artists2 = update_frame(gen_kps[idx_gen], lines2)
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
    parser.add_argument("--output_dir", type=str, default="eval_results_clean")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--ae_hidden_dim", type=int, default=512)
    
    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Evaluation started (Correct Topology)...")

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
    
    ae = UnifiedPoseAutoencoder(latent_dim=args.latent_dim, hidden_dim=args.ae_hidden_dim).to(device)
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device)['model_state_dict'])
    ae.eval()
    
    flow_matcher = LatentFlowMatcher(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    flow_ckpt = torch.load(args.flow_ckpt, map_location=device)
    flow_matcher.load_state_dict(flow_ckpt['model_state_dict'], strict=False)
    flow_matcher.eval()
    
    latent_scale = float(flow_ckpt.get("latent_scale_factor", 1.0))
    print(f"📏 Scale: {latent_scale:.4f}")

    visualizer = CorrectSkeletonVisualizer()

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
        
        text_features, text_mask = flow_matcher.encode_text(text_tokens, attention_mask)
        
        gen_latent = flow_matcher._inference_forward(
            batch=None, 
            text_features=text_features, 
            text_mask=text_mask,
            num_steps=50
        )
        
        gen_latent_scaled = gen_latent * latent_scale
        gen_pose = ae.decode(gen_latent_scaled)
        gen_pose = gen_pose.squeeze(0).detach().cpu().numpy()
        
        real_pose_denorm = denormalize(pose_gt, mean, std)
        gen_pose_denorm = denormalize(gen_pose, mean, std)
        
        save_path_base = os.path.join(args.output_dir, f"sample_{i}_{video_id}")
        
        try:
            visualizer.create_animation(real_pose_denorm, gen_pose_denorm, text, f"{save_path_base}.mp4")
            print(f"   [{i+1}] 🎥 Video: {save_path_base}.mp4")
        except Exception as e:
            print(f"   ⚠️ Render error: {e}")
            import traceback
            traceback.print_exc()

    print("✅ Done!")

if __name__ == "__main__":
    main()