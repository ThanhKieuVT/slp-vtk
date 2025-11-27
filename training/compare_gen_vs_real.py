"""
Script: Đánh giá & So sánh ngẫu nhiên các mẫu từ tập Test/Dev
Tự động sinh video so sánh Real vs Gen.
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
    # Import các module có sẵn
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher
    from data_preparation import load_sample, denormalize_pose
    from compare_gen_vs_real import extract_visual_points # Tận dụng hàm vẽ từ script trước
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    print("💡 Chị nhớ chạy từ thư mục gốc dự án nhé!")
    sys.exit(1)

# === HÀM VẼ VIDEO (Copy lại logic chuẩn từ compare_gen_vs_real.py) ===
def create_comparison_video(real_pose, gen_pose, save_path, title):
    # Lấy phần tọa độ visual (Body + Mouth)
    real_data = extract_visual_points(real_pose)
    gen_data = extract_visual_points(gen_pose)
    
    min_len = min(len(real_data), len(gen_data))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    
    # Tính scale chung
    all_x = real_data[:, :, 0].flatten()
    all_y = real_data[:, :, 1].flatten()
    valid = (all_x > 0.01) & (all_y > 0.01)
    
    if valid.sum() > 0:
        x_min, x_max = all_x[valid].min(), all_x[valid].max()
        y_min, y_max = all_y[valid].min(), all_y[valid].max()
    else:
        x_min, x_max, y_min, y_max = 0, 1, 0, 1
        
    margin = 0.1
    w, h = x_max - x_min, y_max - y_min
    
    for ax in [ax1, ax2]:
        ax.set_xlim(x_min - margin*w, x_max + margin*w)
        ax.set_ylim(y_max + margin*h, y_min - margin*h)
        ax.axis('off')
        
    ax1.set_title("REAL (Ground Truth)", color='darkred')
    ax2.set_title("GENERATED (AI)", color='darkblue')
    fig.suptitle(title, fontsize=9)
    
    scat_real = ax1.scatter([], [], s=5, c='red', alpha=0.6)
    scat_gen = ax2.scatter([], [], s=5, c='blue', alpha=0.6)
    
    def update(frame):
        # Frame t
        p_r = real_data[frame]
        p_g = gen_data[frame]
        
        # Mask điểm rác
        mask_r = (np.abs(p_r).sum(axis=1) > 0.001)
        mask_g = (np.abs(p_g).sum(axis=1) > 0.001)
        
        scat_real.set_offsets(p_r[mask_r])
        scat_gen.set_offsets(p_g[mask_g])
        return scat_real, scat_gen

    ani = animation.FuncAnimation(fig, update, frames=min_len, blit=True, interval=50)
    ani.save(save_path, writer='ffmpeg', fps=25)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', type=str, required=True, help="File text nguồn (VD: test.txt)")
    parser.add_argument('--data_dir', type=str, required=True, help="Thư mục data gốc (processed_data/data)")
    parser.add_argument('--flow_ckpt', type=str, required=True)
    parser.add_argument('--ae_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--num_samples', type=int, default=5, help="Số lượng mẫu muốn test")
    parser.add_argument('--split', type=str, default='test', help="Split của data (train/dev/test)")
    
    # Model config (Khớp với lúc train)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--max_seq_len', type=int, default=120)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"🚀 Bắt đầu đánh giá trên device: {device}")

    # --- 1. LOAD MODELS ---
    print("📦 Loading Models...")
    # Autoencoder
    ae = UnifiedPoseAutoencoder(214, args.latent_dim, args.hidden_dim).to(device)
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device)['model_state_dict'])
    ae.eval()
    
    # Flow Matcher
    ckpt = torch.load(args.flow_ckpt, map_location=device)
    scale_factor = ckpt.get('latent_scale_factor', 1.0)
    print(f"ℹ️ Scale Factor: {scale_factor:.4f}")
    
    flow = LatentFlowMatcher(
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        num_flow_layers=6, num_prior_layers=6, num_heads=8, # Check lại config layer của chị
        use_ssm_prior=True, use_sync_guidance=True
    ).to(device)
    flow.load_state_dict(ckpt['model_state_dict'], strict=False)
    flow.eval()
    
    # Tokenizer & Stats
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if not os.path.exists(stats_path):
        # Thử tìm ở thư mục cha
        stats_path = os.path.join(args.data_dir, "../normalization_stats.npz")
    stats = np.load(stats_path)
    mean, std = stats['mean'], stats['std']

    # --- 2. ĐỌC FILE TEXT & CHỌN MẪU ---
    print(f"📖 Đọc file text: {args.text_file}")
    samples = []
    
    # Tìm các ID có sẵn file pose
    split_pose_dir = os.path.join(args.data_dir, args.split, "poses")
    available_ids = set([f.replace('.npz','') for f in os.listdir(split_pose_dir)])
    
    with open(args.text_file, 'r', encoding='utf-8') as f:
        lines = f.readlines()
        
    # Lọc lấy các dòng hợp lệ
    valid_lines = []
    for line in lines:
        parts = line.strip().split('|')
        if len(parts) >= 2:
            vid_id = parts[0].strip()
            text = parts[-1].strip() # Lấy cột cuối cùng (German text)
            if vid_id in available_ids:
                valid_lines.append((vid_id, text))
    
    # Chọn ngẫu nhiên
    selected_samples = random.sample(valid_lines, min(args.num_samples, len(valid_lines)))
    print(f"✅ Đã chọn {len(selected_samples)} mẫu ngẫu nhiên.")

    # --- 3. LOOP: INFERENCE & COMPARE ---
    for idx, (vid_id, text) in enumerate(selected_samples):
        print(f"\n[{idx+1}/{len(selected_samples)}] Processing: {vid_id}")
        print(f"   📝 Text: '{text}'")
        
        # A. Inference
        with torch.no_grad():
            encoded = tokenizer(text, return_tensors='pt', padding=True).to(device)
            text_feat, text_mask = flow.encode_text(encoded['input_ids'], encoded['attention_mask'])
            
            # Sinh Latent
            latent = flow._inference_forward(
                batch={}, text_features=text_feat, text_mask=text_mask, num_steps=50
            )
            
            # Decode & Denormalize
            latent = latent / scale_factor
            pose_norm = ae.decoder(latent).squeeze(0).cpu().numpy()
            
            # Làm mượt (Optional)
            from scipy.signal import savgol_filter
            if pose_norm.shape[0] > 7:
                pose_norm = savgol_filter(pose_norm, 7, 2, axis=0)
            
            gen_pose = denormalize_pose(pose_norm, mean, std)
            
        # B. Load Real Pose
        real_pose, _ = load_sample(vid_id, os.path.join(args.data_dir, args.split))
        
        # C. Create Video
        out_name = f"sample_{idx}_{vid_id}.mp4"
        out_path = os.path.join(args.output_dir, out_name)
        create_comparison_video(real_pose, gen_pose, out_path, f"ID: {vid_id}\nText: {text}")
        print(f"   💾 Saved: {out_path}")

    print(f"\n🎉 HOÀN TẤT! Kiểm tra thư mục '{args.output_dir}' nhé.")

if __name__ == '__main__':
    main()