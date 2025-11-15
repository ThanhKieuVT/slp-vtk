# Tên file: run_inference.py
# === PHIÊN BẢN CẢI TIẾN: LDM-BERT Inference Robust ===

import os
import argparse
import torch
import numpy as np
from transformers import BertModel, BertTokenizer

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
    
    # --- Model Params ---
    parser.add_argument('--pose_dim', type=int, default=214)
    parser.add_argument('--latent_dim', type=int, default=256)
    parser.add_argument('--ae_hidden_dim', type=int, default=512)
    parser.add_argument('--text_embed_dim', type=int, default=768)
    parser.add_argument('--num_layers', type=int, default=6)
    
    # --- Seed for reproducibility ---
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    
    # --- Validate args ---
    assert args.steps > 0, "Number of inference steps must be > 0"
    assert args.cfg_scale >= 0, "Guidance scale must be >= 0"
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # --- 1. Load Autoencoder ---
    print("Loading Stage 1 Autoencoder...")
    autoencoder = UnifiedPoseAutoencoder(
        pose_dim=args.pose_dim,
        latent_dim=args.latent_dim,
        hidden_dim=args.ae_hidden_dim
    )
    try:
        ae_checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
        autoencoder.load_state_dict(ae_checkpoint['model_state_dict'])
    except KeyError:
        raise KeyError(f"Checkpoint {args.autoencoder_checkpoint} missing 'model_state_dict'")
    autoencoder.to(device).eval().requires_grad_(False)

    # --- 2. Load BERT Text Encoder ---
    print("Loading BERT Text Encoder...")
    bert_name = "bert-base-multilingual-cased"
    tokenizer = BertTokenizer.from_pretrained(bert_name)
    text_encoder = BertModel.from_pretrained(bert_name).to(device)
    text_encoder.eval().requires_grad_(False)
    
    # --- 3. Load LDM Denoiser ---
    print("Loading LDM Teacher Model (Stage 2a)...")
    ldm_model = LDM_TransformerDenoiser(
        latent_dim=args.latent_dim,
        text_embed_dim=args.text_embed_dim,
        hidden_dim=args.text_embed_dim,
        num_layers=args.num_layers,
        num_heads=12
    ).to(device)
    try:
        ldm_checkpoint = torch.load(args.ldm_checkpoint, map_location=device)
        ldm_model.load_state_dict(ldm_checkpoint['model_state_dict'])
    except KeyError:
        raise KeyError(f"Checkpoint {args.ldm_checkpoint} missing 'model_state_dict'")
    ldm_model.eval().requires_grad_(False)
    
    # --- Ensure output folder exists ---
    output_dir = os.path.dirname(args.output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # --- 4. Run Inference ---
    print(f"\nGenerating pose for prompt: '{args.prompt}'")
    with torch.no_grad():
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
    
    # --- 5. Save Result ---
    pose_np = generated_pose.squeeze(0).cpu().numpy()
    np.save(args.output_path, pose_np)
    
    print(f"\n✅ Pose saved to: {args.output_path}")


if __name__ == '__main__':
    main()
