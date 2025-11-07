"""
Inference Script: Text → Latent → Pose
"""
import os
import argparse
import torch
import numpy as np
from transformers import BertTokenizer

from models.fml.autoencoder import UnifiedPoseAutoencoder
from models.fml.latent_predictor import LatentPredictor
from data_preparation import denormalize_pose


def inference(
    text,
    predictor,
    decoder,
    tokenizer,
    device,
    target_length=50,
    normalize_stats=None
):
    """
    Inference: Text → Latent → Pose
    
    Args:
        text: str - input text
        predictor: LatentPredictor model
        decoder: PoseDecoder model (frozen)
        tokenizer: BertTokenizer
        device: torch.device
        target_length: int - độ dài target sequence
        normalize_stats: dict với 'mean' và 'std' để denormalize
    
    Returns:
        pose: [T, 214] numpy array (denormalized nếu có stats)
    """
    predictor.eval()
    decoder.eval()
    
    # Tokenize text
    encoded = tokenizer(
        text,
        max_length=64,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    text_tokens = encoded['input_ids'].to(device)  # [1, L]
    attention_mask = encoded['attention_mask'].to(device)  # [1, L]
    
    with torch.no_grad():
        # Predict latent
        latent = predictor(
            text_tokens,
            attention_mask,
            target_length=target_length,
            mask=None
        )  # [1, T, 256]
        
        # Decode to pose
        pose = decoder(latent, mask=None)  # [1, T, 214]
        
        # Convert to numpy
        pose = pose.squeeze(0).cpu().numpy()  # [T, 214]
    
    # Denormalize nếu có stats
    if normalize_stats is not None:
        mean = normalize_stats['mean']
        std = normalize_stats['std']
        pose = denormalize_pose(pose, mean, std)
    
    return pose


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True,
                        help='Input text')
    parser.add_argument('--predictor_checkpoint', type=str, required=True,
                        help='Checkpoint của Latent Predictor (Stage 2)')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                        help='Checkpoint của Autoencoder (Stage 1)')
    parser.add_argument('--output_path', type=str, default='output_pose.npy',
                        help='Đường dẫn lưu output pose')
    parser.add_argument('--target_length', type=int, default=50,
                        help='Độ dài target sequence')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data dir để load normalization stats')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    
    args = parser.parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load normalization stats
    normalize_stats = None
    if args.data_dir:
        stats_path = os.path.join(args.data_dir, "normalization_stats.npz")
        if os.path.exists(stats_path):
            normalize_stats = np.load(stats_path)
            print(f"Loaded normalization stats from {stats_path}")
    
    # Load Autoencoder (chỉ cần decoder)
    print(f"Loading autoencoder from {args.autoencoder_checkpoint}")
    autoencoder = UnifiedPoseAutoencoder(
        pose_dim=214,
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim
    )
    checkpoint = torch.load(args.autoencoder_checkpoint, map_location=device)
    autoencoder.load_state_dict(checkpoint['model_state_dict'])
    autoencoder.to(device)
    autoencoder.eval()
    decoder = autoencoder.decoder
    
    # Load Predictor
    print(f"Loading predictor from {args.predictor_checkpoint}")
    predictor = LatentPredictor(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_layers=6,
        num_heads=8,
        num_queries=args.target_length,
        dropout=0.1
    )
    checkpoint = torch.load(args.predictor_checkpoint, map_location=device)
    predictor.load_state_dict(checkpoint['model_state_dict'])
    predictor.to(device)
    predictor.eval()
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Inference
    print(f"\nInput text: {args.text}")
    print(f"Target length: {args.target_length}")
    print("Running inference...")
    
    pose = inference(
        args.text,
        predictor,
        decoder,
        tokenizer,
        device,
        target_length=args.target_length,
        normalize_stats=normalize_stats
    )
    
    print(f"Generated pose shape: {pose.shape}")
    print(f"Pose range: [{pose.min():.4f}, {pose.max():.4f}]")
    
    # Save
    np.save(args.output_path, pose)
    print(f"\n✅ Saved pose to {args.output_path}")


if __name__ == '__main__':
    main()

