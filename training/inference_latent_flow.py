"""
Inference Script: Text → Latent Flow → Pose
Hỗ trợ fast sampling (1-4 steps) sau khi distill
"""
import os
import argparse
import torch
import numpy as np
from transformers import BertTokenizer
import time

from models.fml.autoencoder import UnifiedPoseAutoencoder
from models.fml.latent_flow_matcher import LatentFlowMatcher
from data_preparation import denormalize_pose


def inference_fast(
    text,
    flow_matcher,
    decoder,
    tokenizer,
    device,
    target_length=50,
    num_steps=4,  # Fast sampling: 1-4 steps
    normalize_stats=None
):
    """
    Fast inference với few-step sampling
    
    Args:
        text: str - input text
        flow_matcher: LatentFlowMatcher model
        decoder: PoseDecoder model (frozen)
        tokenizer: BertTokenizer
        device: torch.device
        target_length: int - độ dài target sequence
        num_steps: int - số bước ODE (1-4 cho fast, 50 cho full quality)
        normalize_stats: dict với 'mean' và 'std' để denormalize
    
    Returns:
        pose: [T, 214] numpy array (denormalized nếu có stats)
        latency: float - thời gian inference (seconds)
    """
    flow_matcher.eval()
    decoder.eval()
    
    start_time = time.time()
    
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
    
    batch_dict = {
        'text_tokens': text_tokens,
        'attention_mask': attention_mask,
        'target_length': target_length
    }
    
    # === SỬA LỖI: BẮT ĐẦU ===
    
    # KHÔNG bọc flow_matcher trong no_grad()
    # vì Sync Guidance (bên trong model) cần tính grad
    latent = flow_matcher(
        batch_dict,
        gt_latent=None,
        mode='inference',
        num_inference_steps=num_steps
    )  # [1, T, 256]
    
    # Chỉ bọc decoder trong no_grad()
    with torch.no_grad():
        pose = decoder(latent, mask=None)  # [1, T, 214]
    
    # === SỬA LỖI: KẾT THÚC ===

    # Convert to numpy
    pose = pose.squeeze(0).cpu().numpy()  # [T, 214]
    
    latency = time.time() - start_time
    
    # Denormalize nếu có stats
    if normalize_stats is not None:
        mean = normalize_stats['mean']
        std = normalize_stats['std']
        pose = denormalize_pose(pose, mean, std)
    
    return pose, latency


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--text', type=str, required=True,
                        help='Input text')
    parser.add_argument('--flow_checkpoint', type=str, required=True,
                        help='Checkpoint của Latent Flow Matcher (Stage 2)')
    parser.add_argument('--autoencoder_checkpoint', type=str, required=True,
                        help='Checkpoint của Autoencoder (Stage 1)')
    parser.add_argument('--output_path', type=str, default='output_pose.npy',
                        help='Đường dẫn lưu output pose')
    parser.add_argument('--target_length', type=int, default=50,
                        help='Độ dài target sequence')
    parser.add_argument('--num_steps', type=int, default=4,
                        help='Số bước ODE (1-4 cho fast, 50 cho full quality)')
    parser.add_argument('--data_dir', type=str, default=None,
                        help='Data dir để load normalization stats')
    parser.add_argument('--latent_dim', type=int, default=256,
                        help='Latent dimension')
    parser.add_argument('--hidden_dim', type=int, default=512,
                        help='Hidden dimension')
    parser.add_argument('--use_ssm_prior', action='store_true',
                        help='Model có dùng SSM prior')
    parser.add_argument('--use_sync_guidance', action='store_true',
                        help='Model có dùng sync guidance')
    
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
            print(f"✅ Loaded normalization stats from {stats_path}")
    
    # Load Autoencoder (chỉ cần decoder)
    print(f"\n📦 Loading autoencoder from {args.autoencoder_checkpoint}")
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
    print("✅ Autoencoder loaded")
    
    # Load Flow Matcher
    print(f"\n📦 Loading flow matcher from {args.flow_checkpoint}")
    flow_matcher = LatentFlowMatcher(
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        num_flow_layers=6,
        num_prior_layers=4,
        num_heads=8,
        dropout=0.1,
        use_ssm_prior=args.use_ssm_prior,
        use_sync_guidance=args.use_sync_guidance,
        lambda_prior=0.1,
        gamma_guidance=0.01
    )
    checkpoint = torch.load(args.flow_checkpoint, map_location=device)
    flow_matcher.load_state_dict(checkpoint['model_state_dict'])
    flow_matcher.to(device)
    flow_matcher.eval()
    print("✅ Flow matcher loaded")
    
    # Tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    
    # Inference
    print(f"\n🚀 Running inference...")
    print(f"  Input text: {args.text}")
    print(f"  Target length: {args.target_length}")
    print(f"  Num steps: {args.num_steps} ({'FAST' if args.num_steps <= 4 else 'FULL QUALITY'})")
    
    pose, latency = inference_fast(
        args.text,
        flow_matcher,
        decoder,
        tokenizer,
        device,
        target_length=args.target_length,
        num_steps=args.num_steps,
        normalize_stats=normalize_stats
    )
    
    print(f"\n✅ Generated pose:")
    print(f"  Shape: {pose.shape}")
    print(f"  Range: [{pose.min():.4f}, {pose.max():.4f}]")
    print(f"  Latency: {latency*1000:.2f} ms")
    print(f"  FPS: {args.target_length/latency:.1f} frames/sec")
    
    # Save
    np.save(args.output_path, pose)
    print(f"\n💾 Saved pose to {args.output_path}")


if __name__ == '__main__':
    main()

