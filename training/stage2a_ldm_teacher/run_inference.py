# Tên file: run_inference.py
# === PHIÊN BẢN CHUẨN: BERT (768-dim) ===

import os
import argparse
import torch
import numpy as np
from transformers import BertModel, BertTokenizer # <<< DÙNG BERT

# (Import các model của chị)
from models.autoencoder import UnifiedPoseAutoencoder
from models.ldm_denoiser import LDM_TransformerDenoiser
from inference import generate_pose

def main():
    parser = argparse.ArgumentParser(description="Run LDM-BERT Inference")
    
    # --- Paths ---
    parser.add_argument('--ldm_checkpoint', type=str, required=True)
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True)
    parser.add_argument('--output_path', type=str, default='generated_pose.npy')

    # --- Generation Params ---
    parser.add_argument('--prompt', type=str, required=True)
    parser.add_argument('--seq_len', type=int, default=120)
    parser.add_argument('--cfg_scale', type=float, default=7.5)
    parser.add_argument('--steps', type=int, default=50)
    
    # --- Model Params (PHẢI KHỚP VỚI KHI TRAIN) ---
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--ae_hidden_dim', type=int, default=512)
    parser.add_argument('--text_embed_dim', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=6)
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Tải Autoencoder (Stage 1) ---
    print("Loading Stage 1 Autoencoder...")
    autoencoder = UnifiedPoseAutoencoder(
        pose_dim=214,
        latent_dim=args.latent_dim,
        hidden_dim=args.ae_hidden_dim
    )
    ae_checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
    autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
    autoencoder.to(device).eval().requires_grad_(False)

    # --- 2. Tải BERT Text Encoder ---
    print("Loading BERT Text Encoder...")
    bert_name = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    text_encoder = BertModel.from_pretrained(bert_name).to(device)
    text_encoder.eval().requires_grad_(False)
    
    # --- 3. Tải LDM Denoiser (Stage 2a) ---
    print("Loading LDM Teacher Model (Stage 2a)...")
    ldm_model = LDM_TransformerDenoiser(
        latent_dim=args.latent_dim,
        text_embed_dim=args.text_embed_dim, # 768
        hidden_dim=args.text_embed_dim,    # 768
        num_layers=args.num_layers,
        num_heads=12 # (BERT-base dùng 12 heads)
    ).to(device)
    ldm_checkpoint = torch.load(args.ldm_checkpoint, map_location=device)
    ldm_model.load_state_dict(ldm_checkpoint['model_state_dict'])
    ldm_model.eval().requires_grad_(False)
    
    # --- 4. Chạy Inference ---
    print(f"\nGenerating pose for: '{args.prompt}'")
    generated_pose = generate_pose(
        ldm_model=ldm_model,
        autoencoder=autoencoder,
        text_encoder=text_encoder,
        tokenizer=tokenizer,
        prompt=args.prompt,
        guidance_scale=args.cfg_scale,
        num_inference_steps=args.steps,
        target_seq_len=args.seq_len
    )
    
    print(f"Generated pose shape: {generated_pose.shape}")
    
    # --- 5. Lưu kết quả ---
    pose_np = generated_pose.squeeze(0).cpu().numpy()
    np.save(args.output_path, pose_np)
    
    print(f"\n✅ Đã lưu pose vào: {args.output_path}")

if __name__ == '__main__':
    main()