import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from torch.utils.data import DataLoader

# Thêm đường dẫn project
sys.path.append(os.getcwd())

try:
    from dataset import SignLanguageDataset, collate_fn
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher
except ImportError:
    # Fallback import nếu chạy trực tiếp trong thư mục training
    sys.path.append(os.path.join(os.getcwd(), '..'))
    from dataset import SignLanguageDataset, collate_fn
    from models.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher

def denormalize(pose, mean, std):
    """Đưa pose về giá trị gốc"""
    return pose * std + mean

def save_comparison_video(real_pose, gen_pose, text, save_path):
    """Tạo video so sánh đơn giản (2D projection)"""
    # real_pose, gen_pose: [T, 214]
    # Lấy 1 số khớp cơ bản để vẽ (ví dụ: Body + Hands cơ bản)
    # Giả định format 214 điểm (Body, Hand L, Hand R, Face...)
    
    frames = real_pose.shape[0]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    ax1.set_title("Ground Truth (Real)")
    ax2.set_title("Generated (Flow)")
    
    # Thiết lập giới hạn khung hình chung
    all_poses = np.concatenate([real_pose, gen_pose])
    # Reshape về [N, Points, 2] (giả sử dữ liệu 2D, nếu 3D thì lấy X,Y)
    # Lưu ý: Code này giả định pose đã được flatten. Cần reshape dựa trên config data của bạn
    # Nếu data là 214 float, có thể là 107 điểm x 2 coords hoặc 71 điểm x 3 coords?
    # Ở đây ta vẽ scatter plot đơn giản để visualize chuyển động
    
    # Heuristic reshape: Thường là (N_points, 2)
    n_points = real_pose.shape[1] // 2 
    
    scat1 = ax1.scatter([], [], s=5, c='blue')
    scat2 = ax2.scatter([], [], s=5, c='red')
    
    text_display = fig.suptitle(f"Text: {text[:50]}...", fontsize=10)

    def init():
        ax1.set_xlim(-2, 2) # Cần điều chỉnh theo data thực tế
        ax1.set_ylim(2, -2) # Đảo ngược trục Y cho ảnh
        ax2.set_xlim(-2, 2)
        ax2.set_ylim(2, -2)
        return scat1, scat2

    def update(frame):
        # Real
        p1 = real_pose[frame].reshape(-1, 2)
        scat1.set_offsets(p1)
        
        # Gen
        if frame < len(gen_pose):
            p2 = gen_pose[frame].reshape(-1, 2)
            scat2.set_offsets(p2)
        
        return scat1, scat2

    ani = animation.FuncAnimation(fig, update, frames=frames, init_func=init, blit=True)
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
    
    # Model config (cần khớp với lúc train)
    parser.add_argument("--latent_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=512)
    parser.add_argument("--ae_hidden_dim", type=int, default=512)
    
    args = parser.parse_args()
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    os.makedirs(args.output_dir, exist_ok=True)
    
    print(f"🚀 Evaluation started...")
    print(f"   Checkpoints: Flow={os.path.basename(args.flow_ckpt)}, AE={os.path.basename(args.ae_ckpt)}")

    # 1. Load Data
    # Lưu ý: Cần file stats để denormalize
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        print("⚠️ Warning: normalization_stats.npz not found. Denormalization might be wrong.")
        mean, std = 0, 1
    else:
        stats = np.load(stats_path)
        mean = stats['mean']
        std = stats['std']
    
    dataset = SignLanguageDataset(
        data_dir=args.data_dir,
        split=args.split,
        text_file=args.text_file, # Load text từ file test cụ thể
        max_seq_len=200 # Cho phép dài hơn lúc train để test
    )
    
    # Lấy ngẫu nhiên hoặc tuần tự
    indices = list(range(min(len(dataset), args.num_samples)))
    subset = torch.utils.data.Subset(dataset, indices)
    loader = DataLoader(subset, batch_size=1, collate_fn=collate_fn)
    
    # 2. Load Models
    # AE
    ae = UnifiedPoseAutoencoder(latent_dim=args.latent_dim, hidden_dim=args.ae_hidden_dim).to(device)
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device)['model_state_dict'])
    ae.eval()
    
    # Flow
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim
    ).to(device)
    
    flow_ckpt = torch.load(args.flow_ckpt, map_location=device)
    flow_matcher.load_state_dict(flow_ckpt['model_state_dict'], strict=False)
    flow_matcher.eval()
    
    # Lấy latent scale từ checkpoint (Quan trọng!)
    latent_scale = float(flow_ckpt.get("latent_scale_factor", 1.0))
    print(f"📏 Using Latent Scale Factor: {latent_scale:.4f}")

    # 3. Inference Loop
    print(f"🎬 Generating {len(subset)} samples...")
    
    for i, batch in enumerate(loader):
        video_id = batch['video_ids'][0]
        text = batch['texts'][0]
        pose_gt = batch['poses'][0].cpu().numpy() # [T, D]
        
        # Prepare inputs
        text_tokens = batch['text_tokens'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        # A. Encode Text
        text_features, text_mask = flow_matcher.encode_text(text_tokens, attention_mask)
        
        # B. Generate Latent (Inference)
        # Số bước inference (50 steps là chuẩn cho Flow Matching)
        with torch.no_grad():
            gen_latent = flow_matcher._inference_forward(
                batch=None,  # Hàm này không cần batch nếu đã có text_features
                text_features=text_features,
                text_mask=text_mask,
                num_steps=50
            )
            
            # C. Decode Latent -> Pose
            # Scale lại latent trước khi decode (Inverse scaling: latent * scale)
            # Lúc train: z = encode(x) / scale
            # Lúc infer: x = decode(z * scale)
            gen_latent_scaled = gen_latent * latent_scale
            gen_pose = ae.decode(gen_latent_scaled)
            
        gen_pose = gen_pose.squeeze(0).cpu().numpy()
        
        # D. Denormalize
        real_pose_denorm = denormalize(pose_gt, mean, std)
        gen_pose_denorm = denormalize(gen_pose, mean, std)
        
        # E. Save Results
        save_path_base = os.path.join(args.output_dir, f"sample_{i}_{video_id}")
        
        # Save Raw NPZ
        np.savez(f"{save_path_base}.npz", 
                 real=real_pose_denorm, 
                 gen=gen_pose_denorm, 
                 text=text)
        
        print(f"   [{i+1}/{len(subset)}] Saved: {save_path_base}.npz | Len: {len(gen_pose)}")
        
        # Optional: Render Video (Nếu có matplotlib animation)
        try:
            save_comparison_video(real_pose_denorm, gen_pose_denorm, text, f"{save_path_base}.mp4")
            print(f"      🎥 Video rendered: {save_path_base}.mp4")
        except Exception as e:
            print(f"      ⚠️ Cannot render video: {e}")

    print("✅ Done! Check results in:", args.output_dir)

if __name__ == "__main__":
    main()