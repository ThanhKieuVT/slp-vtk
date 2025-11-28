import os
import sys
import argparse
import torch
import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from transformers import BertTokenizer
from scipy.signal import savgol_filter

sys.path.append(os.getcwd())

from models.fml.latent_flow_matcher import LatentFlowMatcher
from models.autoencoder import UnifiedPoseAutoencoder # Check đúng đường dẫn
# from data_preparation import load_sample, denormalize_pose # Cần file này của bạn

# ==========================================
# HELPER FUNCTIONS (Placeholder nếu thiếu file data_preparation)
# ==========================================
def denormalize_pose(pose, mean, std):
    return pose * std + mean

def temporal_smooth_latent(latent, window=5, polyorder=2):
    if latent.shape[1] < window: return latent
    latent_np = latent.cpu().numpy()
    smoothed = np.zeros_like(latent_np)
    for b in range(latent_np.shape[0]):
        for d in range(latent_np.shape[2]):
            smoothed[b, :, d] = savgol_filter(latent_np[b, :, d], window, polyorder)
    return torch.from_numpy(smoothed).to(latent.device)

# ==========================================
# MAIN
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, default="Xin chào Việt Nam")
    parser.add_argument('--flow_ckpt', type=str, required=True)
    parser.add_argument('--ae_ckpt', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_steps', type=int, default=50)
    
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load Models
    print("Loading models...")
    ae = UnifiedPoseAutoencoder(latent_dim=args.latent_dim).to(device)
    ae.load_state_dict(torch.load(args.ae_ckpt, map_location=device)['model_state_dict'], strict=False)
    ae.eval()
    
    flow = LatentFlowMatcher(latent_dim=args.latent_dim, hidden_dim=args.hidden_dim).to(device)
    ckpt = torch.load(args.flow_ckpt, map_location=device)
    flow.load_state_dict(ckpt['model_state_dict'], strict=False)
    scale_factor = ckpt.get('latent_scale_factor', 1.0)
    flow.eval()
    
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # Inference
    print(f"Generating for: {args.text}")
    with torch.no_grad():
        encoded = tokenizer(args.text, return_tensors='pt', padding=True).to(device)
        text_feat, text_mask = flow.encode_text(encoded['input_ids'], encoded['attention_mask'])
        
        # 1. Dự đoán độ dài
        pred_len = flow.length_predictor(text_feat, text_mask)
        est_len = int(pred_len.round().item())
        est_len = max(30, min(est_len, 300))
        print(f"Predicted Length: {est_len}")
        
        # 2. Tạo batch giả
        fake_batch = {'seq_lengths': torch.tensor([est_len], device=device)}
        
        # 3. Flow Inference
        latent = flow._inference_forward(fake_batch, text_feat, text_mask, num_steps=args.num_steps, latent_scale=scale_factor)
        
        # 4. Smoothing
        latent = temporal_smooth_latent(latent, window=7, polyorder=2)
        
        # 5. Decode
        latent = latent / scale_factor
        pose = ae.decoder(latent) # [1, T, 214]
        
        # Lưu kết quả
        np.save(os.path.join(args.output_dir, "gen_pose.npy"), pose.cpu().numpy())
        print(f"Saved to {args.output_dir}/gen_pose.npy")

if __name__ == '__main__':
    main()