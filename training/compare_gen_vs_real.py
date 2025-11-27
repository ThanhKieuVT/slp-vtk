"""
Script: Đánh giá & So sánh ngẫu nhiên các mẫu từ tập Test/Dev
Tự động sinh video so sánh Real vs Gen (Dạng Point Cloud chuẩn).
"""
import os
import sys
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from transformers import BertTokenizer

# --- SETUP PATH ---
sys.path.append(os.getcwd()) 

try:
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher
    from data_preparation import load_sample, denormalize_pose
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("💡 Chị nhớ chạy từ thư mục gốc dự án nhé!")
    sys.exit(1)

# === 1. HÀM TÁCH TỌA ĐỘ (Logic chuẩn: Body + Mouth, bỏ rác) ===
def extract_visual_coordinates(pose_214):
    # 1. Lấy Body + Hands (75 điểm x 2 = 150) -> Index: 0-150
    body_hands = pose_214[:, :150] 
    
    # 2. Lấy Mouth (20 điểm x 2 = 40) -> Index: 174-214
    mouth = pose_214[:, 174:214]
    
    # Gộp lại: 75 + 20 = 95 điểm
    visual_flat = np.concatenate([body_hands, mouth], axis=1)
    
    # Reshape thành (N, 2)
    return visual_flat.reshape(len(pose_214), -1, 2) 

# === 2. HÀM TẠO VIDEO (Point Cloud) ===
def create_comparison_video(real_pose, gen_pose, save_path, title):
    # Lấy tọa độ sạch
    real_data = extract_visual_coordinates(real_pose)
    gen_data = extract_visual_coordinates(gen_pose)
    
    min_len = min(len(real_data), len(gen_data))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Tính scale chung
    all_x = real_data[:, :, 0].flatten()
    all_y = real_data[:, :, 1].flatten()
    
    # Lọc điểm rác để tính khung hình
    valid = (all_x > 0.01) & (all_y > 0.01)
    if valid.sum() > 0:
        x_min, x_max = all_x[valid].min(), all_x[valid].max()
        y_min, y_max = all_y[valid].min(), all_y[valid].max()
    else:
        x_min, x_max, y_min, y_max = 0, 1, 0, 1
        
    margin_w = (x_max - x_min) * 0.1
    margin_h = (y_max - y_min) * 0.1
    
    for ax in [ax1, ax2]:
        ax.set_xlim(x_min - margin_w, x_max + margin_w)
        ax.set_ylim(y_max + margin_h, y_min - margin_h) # Đảo trục Y
        ax.axis('off')
        
    ax1.set_title("REAL (Ground Truth)", color='darkred', fontweight='bold')
    ax2.set_title("GENERATED (AI)", color='darkblue', fontweight='bold')
    fig.suptitle(title, fontsize=9)
    
    # Vẽ chấm (Scatter)
    scat_real = ax1.scatter([], [], s=10, c='red', alpha=0.6)
    scat_gen = ax2.scatter([], [], s=10, c='blue', alpha=0.6)
    
    def update(frame):
        p_r = real_data[frame]
        p_g = gen_data[frame]
        
        # Mask điểm rác (tọa độ 0,0)
        mask_r = (np.abs(p_r).sum(axis=1) > 1e-3)
        mask_g = (np.abs(p_g).sum(axis=1) > 1e-3)
        
        scat_real.set_offsets(p_r[mask_r])
        scat_gen.set_offsets(p_g[mask_g])
        return scat_real, scat_gen

    ani = animation.FuncAnimation(fig, update, frames=min_len, blit=True, interval=50)
    ani.save(save_path, writer='ffmpeg', fps=25)
    plt.close()

# === 3. MAIN ===
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--flow_ckpt', type=str, required=True)
    parser.add_argument('--ae_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--split', type=str, default='test')
    
    # Model config
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--ae_hidden_dim', type=int, default=512)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🚀 Bắt đầu đánh giá trên: {device}")

    # --- Load Models ---
    print("📦 Loading Models...")
    ae = UnifiedPoseAutoencoder(214, args.latent_dim, args.ae_hidden_dim).to(device)
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device)['model_state_dict'])
    ae.eval()
    
    ckpt = torch.load(args.flow_ckpt, map_location=device)
    scale_factor = ckpt.get('latent_scale_factor', 1.0)
    print(f"ℹ️ Scale Factor: {scale_factor:.4f}")
    
    flow = LatentFlowMatcher(
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        num_flow_layers=6, num_prior_layers=4, num_heads=8,
        use_ssm_prior=True, use_sync_guidance=True
    ).to(device)
    flow.load_state_dict(ckpt['model_state_dict'], strict=False)
    flow.eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # --- Load Stats ---
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path): stats_path = os.path.join(args.data_dir, "../normalization_stats.npz")
    stats = np.load(stats_path)
    mean, std = stats['mean'], stats['std']

    # --- Select Samples ---
    split_pose_dir = os.path.join(args.data_dir, args.split, "poses")
    available_ids = set([f.replace('.npz','') for f in os.listdir(split_pose_dir)])
    
    samples = []
    with open(args.text_file, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                vid_id = parts[0].strip()
                text = parts[-1].strip()
                if vid_id in available_ids:
                    samples.append((vid_id, text))
    
    if not samples:
        print("❌ Không tìm thấy mẫu nào!")
        sys.exit(1)
        
    selected = random.sample(samples, min(args.num_samples, len(samples)))
    print(f"✅ Đã chọn {len(selected)} mẫu.")

    # --- Run Inference ---
    for idx, (vid_id, text) in enumerate(selected):
        print(f"\n[{idx+1}/{len(selected)}] Generating: {vid_id}")
        
        # 1. Generate
        with torch.no_grad():
            encoded = tokenizer(text, return_tensors='pt', padding=True).to(device)
            text_feat, text_mask = flow.encode_text(encoded['input_ids'], encoded['attention_mask'])
            
            latent = flow._inference_forward(
                batch={}, text_features=text_feat, text_mask=text_mask, num_steps=50
            )
            
            latent = latent / scale_factor
            pose_norm = ae.decoder(latent).squeeze(0).cpu().numpy()
            
            # Smoothing
            from scipy.signal import savgol_filter
            if pose_norm.shape[0] > 7:
                pose_norm = savgol_filter(pose_norm, 7, 2, axis=0)
            
            gen_pose = denormalize_pose(pose_norm, mean, std)
            
        # 2. Get Real
        real_pose, _ = load_sample(vid_id, os.path.join(args.data_dir, args.split))
        
        # 3. Create Video
        out_path = os.path.join(args.output_dir, f"sample_{idx}_{vid_id}.mp4")
        create_comparison_video(real_pose, gen_pose, out_path, f"ID: {vid_id}")
        print(f"   💾 Saved: {out_path}")

    print("\n🎉 DONE!")

if __name__ == '__main__':
    main()