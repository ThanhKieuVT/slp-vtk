"""
Inference Script: Text → Latent Flow → Pose
UPDATED: Hỗ trợ Length Predictor tự động & Scale Factor Correction
"""
import os
import argparse
import torch
import numpy as np
from transformers import BertTokenizer
import time

# --- IMPORT MODEL ---
try:
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher
    # from data_preparation import denormalize_pose # Bỏ qua nếu chị lưu trực tiếp 214 điểm
except ImportError as e:
    print(f"❌ Lỗi Import: {e}. Hãy chạy script từ thư mục gốc dự án.")
    import sys
    sys.exit(1)

def inference_sota(
    text,
    flow_matcher,
    decoder,
    tokenizer,
    device,
    scale_factor=1.0, # MỚI: Cần tham số này để pose không bị bé tí
    num_steps=50,
    manual_length=None # Nếu muốn ép độ dài (optional)
):
    flow_matcher.eval()
    decoder.eval()
    
    start_time = time.time()
    
    # 1. Tokenize text
    encoded = tokenizer(text, return_tensors='pt', padding=True).to(device)
    text_tokens = encoded['input_ids']
    attention_mask = encoded['attention_mask']
    
    # 2. Encode Text Features (Làm bên ngoài để clear logic)
    with torch.no_grad():
        text_features, text_mask = flow_matcher.encode_text(text_tokens, attention_mask)
    
    # 3. Flow Matching Inference
    # Model sẽ TỰ ĐỘNG dự đoán độ dài bên trong hàm này
    # batch={} vì ta không còn cần target_length từ ngoài nữa (trừ khi manual)
    
    # *Mẹo*: Nếu chị muốn ép độ dài thủ công để test, chị có thể hack vào hàm _inference_forward
    # nhưng mặc định hãy để model tự làm.
    
    print(f"🔄 Đang sinh Latent (Steps={num_steps})...")
    # KHÔNG bọc flow_matcher trong no_grad nếu dùng Guidance (nhưng ở đây infer thuần nên ok)
    # Tuy nhiên, hàm _inference_forward của chị đã có torch.set_grad_enabled bên trong logic guidance rồi
    # Nên gọi bình thường.
    
    generated_latent = flow_matcher._inference_forward(
        batch={}, 
        text_features=text_features, 
        text_mask=text_mask, 
        num_steps=num_steps
    ) # [1, T, 256]
    
    # 4. UN-SCALE LATENT (QUAN TRỌNG NHẤT)
    # Lúc train ta nhân scale_factor, giờ phải chia đi
    generated_latent = generated_latent / scale_factor
    
    # 5. Decode ra Pose
    with torch.no_grad():
        pose = decoder(generated_latent) # [1, T, 214]

    # 6. Post-process
    pose = pose.squeeze(0).cpu().numpy()  # [T, 214]
    latency = time.time() - start_time
    
    return pose, latency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True, help='Câu text đầu vào')
    parser.add_argument('--flow_checkpoint', type=str, required=True, help='Checkpoint Flow (Stage 2)')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True, help='Checkpoint AE (Stage 1)')
    parser.add_argument('--output_path', type=str, default='output_pose.npy', help='Nơi lưu file .npy')
    
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
    
    # 1. Load Autoencoder
    print(f"📦 Loading Autoencoder...")
    ae = UnifiedPoseAutoencoder(pose_dim=214, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    ae.load_state_dict(torch.load(args.autoencoder_checkpoint, map_location=device)['model_state_dict'])
    ae.eval()
    
    # 2. Load Flow Matcher & Scale Factor
    print(f"📦 Loading Flow Matcher...")
    ckpt = torch.load(args.flow_checkpoint, map_location=device)
    
    # LẤY SCALE FACTOR TỪ CHECKPOINT
    scale_factor = ckpt.get('latent_scale_factor', 1.0)
    print(f"✅ Tìm thấy Scale Factor: {scale_factor:.4f}")
    
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        num_flow_layers=6, num_prior_layers=4, num_heads=8, dropout=0.1,
        use_ssm_prior=args.use_ssm_prior, use_sync_guidance=args.use_sync_guidance
        # Các tham số loss weights không quan trọng lúc inference
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
        scale_factor=scale_factor, # Truyền scale factor vào
        num_steps=args.num_steps
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