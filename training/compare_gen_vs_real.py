"""
Script: Đánh giá & So sánh ngẫu nhiên các mẫu từ tập Test/Dev
VISUALIZATION: Skeleton Format (Đẹp, có màu, nối dây chuẩn)
"""
import os
import sys
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.lines import Line2D
from transformers import BertTokenizer
from scipy.signal import savgol_filter

# --- SETUP PATH ---
sys.path.append(os.getcwd()) 

try:
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher
    from data_preparation import load_sample, denormalize_pose
except ImportError as e:
    print(f"❌ Lỗi Import: {e}")
    sys.exit(1)

# ==========================================
# 1. CẤU HÌNH KẾT NỐI XƯƠNG (COPY TỪ FILE CỦA CHỊ)
# ==========================================
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
# Nối cổ tay vào thân
ALL_CONNECTIONS.append({'indices': (15, 0), 'offset': (0, 33), 'color': 'blue', 'lw': 2})

ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 54, 'color': 'green', 'lw': 1.5}
    for (s, e) in HAND_CONNECTIONS
])
# Nối cổ tay phải vào thân
ALL_CONNECTIONS.append({'indices': (16, 0), 'offset': (0, 54), 'color': 'green', 'lw': 2})

ALL_CONNECTIONS.extend([
    {'indices': (s, e), 'offset': 75, 'color': 'red', 'lw': 1}
    for (s, e) in MOUTH_CONNECTIONS_20
])

# Các điểm cần vẽ scatter
MANUAL_UPPER_BODY_IDXS = list(range(23))
LEFT_HAND_IDXS = list(range(33, 54))
RIGHT_HAND_IDXS = list(range(54, 75))
MOUTH_IDXS = list(range(75, 95))
PLOT_IDXS = MANUAL_UPPER_BODY_IDXS + LEFT_HAND_IDXS + RIGHT_HAND_IDXS + MOUTH_IDXS
VALID_POINT_THRESHOLD = 0.01 # Ngưỡng lọc điểm rác (chỉnh nhỏ xuống vì data đã normalize)

# ==========================================
# 2. HÀM XỬ LÝ DATA
# ==========================================
def load_and_prepare_pose(pose_214):
    """Chuyển đổi pose 214 chiều thành format (95, 2) để vẽ"""
    # Body + Hands (0-150) -> 75 điểm
    manual_150 = pose_214[:, :150]
    manual_kps = manual_150.reshape(-1, 75, 2)
    
    # Mouth (174-214) -> 20 điểm
    mouth_40 = pose_214[:, 174:214]
    mouth_kps = mouth_40.reshape(-1, 20, 2)
    
    # Ghép lại: 75 + 20 = 95 điểm
    all_kps = np.concatenate([manual_kps, mouth_kps], axis=1)
    return all_kps

# ==========================================
# 3. HÀM TẠO VIDEO SO SÁNH (SKELETON)
# ==========================================
def create_comparison_video(real_pose, gen_pose, save_path, title):
    # Prepare data
    kps_gt = load_and_prepare_pose(real_pose)
    kps_recon = load_and_prepare_pose(gen_pose)
    
    min_len = min(len(kps_gt), len(kps_recon))
    kps_gt = kps_gt[:min_len]
    kps_recon = kps_recon[:min_len]
    
    # Setup Figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # --- Config trục (Dựa trên GT) ---
    plot_points_gt = kps_gt[:, PLOT_IDXS]
    valid_kps = plot_points_gt[np.sum(np.abs(plot_points_gt), axis=2) > VALID_POINT_THRESHOLD]
    
    if valid_kps.shape[0] > 0:
         min_vals = np.min(valid_kps, axis=0)
         max_vals = np.max(valid_kps, axis=0)
         padding = 0.2 * (max_vals - min_vals)
         # Đảm bảo padding không quá nhỏ
         padding = np.maximum(padding, 0.1) 
         
         x_lims = (min_vals[0] - padding[0], max_vals[0] + padding[0])
         y_lims = (max_vals[1] + padding[1], min_vals[1] - padding[1]) # Invert Y
    else:
         x_lims = (-0.5, 0.5)
         y_lims = (0.5, -0.5)
    
    for ax in [ax1, ax2]:
        ax.set_xlim(x_lims)
        ax.set_ylim(y_lims)
        ax.axis('off')
    
    ax1.set_title("REAL (Ground Truth)", color='darkred', fontweight='bold')
    ax2.set_title("GENERATED (AI)", color='darkblue', fontweight='bold')
    fig.suptitle(title, fontsize=10)

    # --- Helper khởi tạo Lines & Scatter ---
    def init_artists(ax):
        lines = []
        for item in ALL_CONNECTIONS:
            line = Line2D([], [], color=item['color'], lw=item['lw'], alpha=0.7)
            ax.add_line(line)
            lines.append({'line': line, 'item': item}) # Lưu kèm item để lấy indices sau này
        
        scatter = ax.scatter([], [], s=2, c='black', alpha=0.4)
        return lines, scatter

    lines_real, scat_real = init_artists(ax1)
    lines_gen, scat_gen = init_artists(ax2)

    # --- Update Function ---
    def update(frame):
        kps_r_frame = kps_gt[frame]
        kps_g_frame = kps_recon[frame]
        
        # Hàm update cho 1 bên
        def update_one_side(kps, lines_obj, scat_obj):
            # Update Lines
            for obj in lines_obj:
                line = obj['line']
                item = obj['item']
                idx_start, idx_end = item['indices']
                
                # Xử lý offset (có thể là tuple hoặc int)
                offset = item['offset']
                if isinstance(offset, (tuple, list)):
                    s_off, e_off = offset[0], offset[1]
                else:
                    s_off, e_off = offset, offset
                
                idx_s = idx_start + s_off
                idx_e = idx_end + e_off
                
                # Check valid point
                if (np.sum(np.abs(kps[idx_s])) > VALID_POINT_THRESHOLD and 
                    np.sum(np.abs(kps[idx_e])) > VALID_POINT_THRESHOLD):
                    line.set_data([kps[idx_s, 0], kps[idx_e, 0]], 
                                  [kps[idx_s, 1], kps[idx_e, 1]])
                else:
                    line.set_data([], [])
            
            # Update Scatter
            pts = kps[PLOT_IDXS]
            valid = pts[np.sum(np.abs(pts), axis=1) > VALID_POINT_THRESHOLD]
            scat_obj.set_offsets(valid)

        update_one_side(kps_r_frame, lines_real, scat_real)
        update_one_side(kps_g_frame, lines_gen, scat_gen)
        
        # Return all artists to update
        return [x['line'] for x in lines_real] + [scat_real] + \
               [x['line'] for x in lines_gen] + [scat_gen]

    ani = animation.FuncAnimation(fig, update, frames=min_len, blit=True, interval=50)
    ani.save(save_path, writer='ffmpeg', fps=25)
    plt.close()

# ==========================================
# 4. MAIN LOGIC
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text_file', type=str, required=True)
    parser.add_argument('--data_dir', type=str, required=True)
    parser.add_argument('--flow_ckpt', type=str, required=True)
    parser.add_argument('--ae_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='evaluation_results')
    parser.add_argument('--num_samples', type=int, default=5)
    parser.add_argument('--split', type=str, default='test')
    
    # Model config (Khớp với lúc train)
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
        # A. Encode Text
        with torch.no_grad():
            encoded = tokenizer(text, return_tensors='pt', padding=True).to(device)
            text_feat, text_mask = flow.encode_text(encoded['input_ids'], encoded['attention_mask'])
            
        # B. Sinh Latent (KHÔNG dùng no_grad)
        latent = flow._inference_forward(
            batch={}, text_features=text_feat, text_mask=text_mask, num_steps=50
        )
        
        # C. Decode
        with torch.no_grad():
            latent = latent / scale_factor
            pose_norm = ae.decoder(latent).squeeze(0).cpu().numpy()
            
            # Smoothing (Quan trọng cho đẹp)
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