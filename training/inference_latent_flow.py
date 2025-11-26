"""
Inference Script: Text -> Latent Flow -> Pose
FIXED: 
1. Enable Gradients for Sync Guidance (Fixed RuntimeError)
2. Match Prior Layers to Checkpoint (Fixed Warning)
3. Safe Checkpoint Loading & Full Error Handling
"""
import os
import argparse
import torch
import numpy as np
from transformers import BertTokenizer
import time
import sys
import inspect

try:
    from models.fml.autoencoder import UnifiedPoseAutoencoder
    from models.fml.latent_flow_matcher import LatentFlowMatcher
    from data_preparation import denormalize_pose 
except ImportError as e:
    print(f"❌ Lỗi Import: {e}. Hãy đảm bảo module và hàm denormalize_pose đã có.")
    sys.exit(1)

def inference_sota(text, flow_matcher, decoder, tokenizer, device, scale_factor=1.0, num_steps=50, normalize_stats=None):
    flow_matcher.eval()
    decoder.eval()
    start_time = time.time()
    
    # --- BƯỚC 1: Encode Text (Dùng no_grad cho nhẹ) ---
    with torch.no_grad():
        encoded = tokenizer(text, return_tensors='pt', padding=True).to(device)
        text_tokens = encoded['input_ids']
        attention_mask = encoded['attention_mask']
        
        text_features, text_mask = flow_matcher.encode_text(text_tokens, attention_mask)
        
    # --- BƯỚC 2: Inference Flow (QUAN TRỌNG: KHÔNG DÙNG no_grad) ---
    # Sync Guidance cần tính gradient để "lái" đường đi của latent
    print(f"🔄 Đang sinh Latent (Steps={num_steps})...")
    
    generated_latent = flow_matcher._inference_forward(
        batch={}, 
        text_features=text_features, 
        text_mask=text_mask, 
        num_steps=num_steps
    ) 
        
    # --- BƯỚC 3: Decode & Post-process (Lại dùng no_grad) ---
    with torch.no_grad():
        # UN-SCALE (Quan trọng)
        generated_latent = generated_latent / scale_factor
        
        # DECODE: Kiểm tra xem Decoder có cần tham số mask không
        T = generated_latent.shape[1]
        decoder_args = inspect.signature(decoder.forward).parameters
        
        if 'mask' in decoder_args:
            decode_mask = torch.ones(1, T, dtype=torch.bool, device=device)
            pose_norm = decoder(generated_latent, mask=decode_mask)
        else:
            pose_norm = decoder(generated_latent)
        
        pose = pose_norm.squeeze(0).cpu().numpy()

    # --- BƯỚC 4: Denormalize ---
    if normalize_stats is not None:
        mean = normalize_stats['mean']
        std = normalize_stats['std']
        pose = denormalize_pose(pose, mean, std) 
        print("✅ Pose Denormalized.")

    latency = time.time() - start_time
    return pose, latency

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True)
    parser.add_argument('--flow_checkpoint', type=str, required=True)
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='output_pose.npy')
    parser.add_argument('--data_dir', type=str, required=True)
    
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_steps', type=int, default=50)
    parser.add_argument('--use_ssm_prior', action='store_true')
    parser.add_argument('--use_sync_guidance', action='store_true')
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🚀 Using device: {device}")
    
    # Check normalization stats
    normalize_stats = None
    stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
    if os.path.exists(stats_path):
        normalize_stats = np.load(stats_path, allow_pickle=True)
    else:
        print(f"❌ Error: Không tìm thấy file stats tại {stats_path}")
        print("💡 Vui lòng tạo lại file stats hoặc copy vào đúng thư mục.")
        sys.exit(1)
    
    # 1. Load Autoencoder
    print(f"📦 Loading Autoencoder...")
    ae = UnifiedPoseAutoencoder(pose_dim=214, latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    ae_ckpt = torch.load(args.autoencoder_checkpoint, map_location=device)
    
    if 'model_state_dict' in ae_ckpt: 
        ae.load_state_dict(ae_ckpt['model_state_dict'])
    else: 
        ae.load_state_dict(ae_ckpt)
    ae.eval()
    
    # 2. Load Flow Matcher
    print(f"📦 Loading Flow Matcher...")
    flow_ckpt = torch.load(args.flow_checkpoint, map_location=device)
    scale_factor = flow_ckpt.get('latent_scale_factor', 1.0)
    print(f"ℹ️ Scale Factor: {scale_factor:.4f}")
    
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim, hidden_dim=args.hidden_dim,
        num_flow_layers=6, 
        num_prior_layers=6, # <--- FIXED: Sửa thành 6 để khớp với checkpoint
        num_heads=8, dropout=0.1,
        use_ssm_prior=args.use_ssm_prior, use_sync_guidance=args.use_sync_guidance
    ).to(device)
    
    # Load weights an toàn
    state_dict = flow_ckpt['model_state_dict'] if 'model_state_dict' in flow_ckpt else flow_ckpt
    try:
        flow_matcher.load_state_dict(state_dict, strict=True)
    except Exception as e:
        print(f"⚠️ Warning strict loading: {e}. Trying strict=False...")
        flow_matcher.load_state_dict(state_dict, strict=False)
    
    flow_matcher.eval()
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # 3. Run Inference
    pose, latency = inference_sota(args.text, flow_matcher, ae.decoder, tokenizer, device, scale_factor, args.num_steps, normalize_stats)
    
    print(f"✅ Done! Latency: {latency:.2f}s. Saved to {args.output_path}")
    np.save(args.output_path, pose)

if __name__ == '__main__':
    main()