"""
Inference Script: Text → Latent Flow → Pose
FIXED: Bổ sung logic tải Normalization Stats và Denormalize Pose.
"""
import os
import argparse
import torch
import numpy as np
from transformers import BertTokenizer
import time

# --- IMPORT MODEL VÀ HÀM DENORMALIZE ---
try:
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher
    # ✅ FIX: Cần hàm denormalize_pose và hàm này nằm trong data_preparation
    from data_preparation import denormalize_pose 
except ImportError as e:
    print(f"❌ Lỗi Import: {e}. Hãy đảm bảo các module và hàm denormalize_pose đã được định nghĩa.")
    import sys
    sys.exit(1)

def inference_sota(
    text,
    flow_matcher,
    decoder,
    tokenizer,
    device,
    scale_factor=1.0, 
    num_steps=50,
    normalize_stats=None # ✅ FIX: Nhận normalization_stats
):
    flow_matcher.eval()
    decoder.eval()
    
    start_time = time.time()
    
    # 1. Tokenize text
    encoded = tokenizer(text, return_tensors='pt', padding=True).to(device)
    text_tokens = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    # 2. Encode Text Features 
    with torch.no_grad():
        text_features, text_mask = flow_matcher.encode_text(text_tokens, attention_mask)
    
    # 3. Flow Matching Inference
    print(f"🔄 Đang sinh Latent (Steps={num_steps})...")
    
    generated_latent = flow_matcher._inference_forward(
        batch={}, 
        text_features=text_features, 
        text_mask=text_mask, 
        num_steps=num_steps
    ) # [1, T, 256]
    
    # 4. UN-SCALE LATENT (QUAN TRỌNG NHẤT)
    generated_latent = generated_latent / scale_factor
    
    # 5. Decode ra Pose
    with torch.no_grad():
        pose_norm = decoder(generated_latent) # [1, T, 214] (Pose ĐÃ CHUẨN HÓA)
        pose = pose_norm.squeeze(0).cpu().numpy()  # [T, 214] (numpy)

    # 6. Post-process: DENORMALIZE BẮT BUỘC
    if normalize_stats is not None:
        mean = normalize_stats['mean']
        std = normalize_stats['std']
        # ✅ FIX: Áp dụng Denormalization để pose có tọa độ vật lý đúng
        pose = denormalize_pose(pose, mean, std) 
        print("✅ Pose Denormalized.")

    latency = time.time() - start_time
    
    return pose, latency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help='Câu text đầu vào')
    parser.add_argument('--flow_checkpoint', type=str, required=True, help='Checkpoint Flow (Stage 2)')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True, help='Checkpoint AE (Stage 1)')
    parser.add_argument('--output_path', type=str, default='output_pose.npy', help='Nơi lưu file .npy')
    parser.add_argument('--data_dir', type=str, required=True, help='Thư mục chứa normalization_stats.npz') # ✅ FIX: data_dir BẮT BUỘC
    
    # Các tham số Model (Phải khớp lúc train)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_steps', type=int, default=50, help='Số bước lấy mẫu (10-50)')
    
    # Các cờ tính năng (Phải khớp lúc train)
    parser.add_argument('--use_ssm_prior', action='store_true')
    parser.add_argument('--use_sync_guidance', action='store_true')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # --- LOAD NORMALIZATION STATS (FIX) ---
    normalize_stats = None
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if os.path.exists(stats_path):
        normalize_stats = np.load(stats_path)
        print(f"✅ Loaded normalization stats from {stats_path}")
    else:
        print(f"❌ ERROR: normalization_stats.npz NOT found at {stats_path}. Cannot Denormalize!")
        
    
    # 1. Load Autoencoder
    print(f"\n📦 Loading Autoencoder...")
    ae = UnifiedPoseAutoencoder(pose_dim=214, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    ae.load_state_dict(torch.load(args.autoencoder_checkpoint, map_location=device)['model_state_dict'])
    ae.eval()
    
    # 2. Load Flow Matcher & Scale Factor
    print(f"📦 Loading Flow Matcher...")
    ckpt = torch.load(args.flow_checkpoint, map_location=device)
    
    scale_factor = ckpt.get('latent_scale_factor', 1.0)
    print(f"✅ Tìm thấy Scale Factor: {scale_factor:.4f}")
    
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        num_flow_layers=6, num_prior_layers=4, num_heads=8, dropout=0.1,
        use_ssm_prior=args.use_ssm_prior, use_sync_guidance=args.use_sync_guidance
    ).to(device)
    
    # Load weights (strict=False để an toàn nếu thiếu key linh tinh)
    try:
        flow_matcher.load_state_dict(ckpt['model_state_dict'], strict=False)
    except Exception as e:
        print(f"⚠️ Warning load weights: {e}")
    
    flow_matcher.eval()
    
    # 3. Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # 4. Run Inference
    print(f"\n🎬 Input: '{args.text}'")
    
    pose, latency = inference_sota(
        text=args.text,
        flow_matcher=flow_matcher,
        decoder=ae.decoder,
        tokenizer=tokenizer,
        device=device,
        scale_factor=scale_factor, 
        num_steps=args.num_steps,
        normalize_stats=normalize_stats # ✅ FIX: Truyền stats vào hàm inference
    )
    
    print(f"✅ Done! Shape: {pose.shape}")
    print(f"⏱️ Latency: {latency:.2f}s")
    print(f"💾 Saving to {args.output_path}")
    np.save(args.output_path, pose)

    # 5. Gợi ý visualize
    print(f"\n💡 Tiếp theo: Chị hãy chạy lệnh sau để xem video:")
    print(f"python visualize_single_pose.py --npy_path {args.output_path} --output_video result.mp4")

if __name__ == '__main__':
    main()