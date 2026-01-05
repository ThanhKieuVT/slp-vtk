"""
Data Preparation: Load và combine poses + nmms thành 214D vectors
✅ FIXED: normalize_pose signature, grouped normalization support
"""
import os
import pickle
import numpy as np
from tqdm import tqdm
from pathlib import Path


def load_sample(video_id, split_dir):
    """
    Load một sample từ extracted features
    
    Args:
        video_id: Tên video (ví dụ: "01April_2010_Thursday_heute_default-1")
        split_dir: Đường dẫn đến thư mục split (chứa poses/ và nmms/)
    
    Returns:
        pose_214: [T, 214] numpy array
        T: Độ dài thực tế
    """
    # Load manual poses
    pose_path = os.path.join(split_dir, "poses", f"{video_id}.npz")
    if not os.path.exists(pose_path):
        return None, 0
    
    poses = np.load(pose_path)
    kp = poses["keypoints"]  # [T, 75, 2]
    visibility = poses.get("visibility", np.ones((kp.shape[0], 75)))  # [T, 75]
    
    # Load NMMs
    nmm_path = os.path.join(split_dir, "nmms", f"{video_id}.pkl")
    if not os.path.exists(nmm_path):
        return None, 0
    
    with open(nmm_path, 'rb') as f:
        nmms = pickle.load(f)
    
    # Đảm bảo độ dài khớp nhau
    T_pose = len(kp)
    T_aus = len(nmms["facial_aus"])
    T = min(T_pose, T_aus)
    
    if T == 0:
        return None, 0
    
    # Combine manual: [T, 75, 2] -> [T, 150]
    manual = kp[:T].reshape(T, -1)  # [T, 150]
    
    # Combine NMMs: 17 + 3 + 4 + 40 = 64
    aus = nmms["facial_aus"][:T]  # [T, 17]
    head = nmms["head_pose"][:T]  # [T, 3]
    gaze = nmms["eye_gaze"][:T]  # [T, 4]
    mouth = nmms["mouth_shape"][:T]  # [T, 20, 2]
    mouth_flat = mouth.reshape(T, -1)  # [T, 40]
    
    nmm = np.concatenate([aus, head, gaze, mouth_flat], axis=-1)  # [T, 64]
    
    # Combine thành 214D
    pose_214 = np.concatenate([manual, nmm], axis=-1)  # [T, 214]
    
    return pose_214, T


def compute_grouped_normalization_stats(data_dir, split='train'):
    """
    ✅ FIXED: Tính grouped normalization stats
    - Manual features (150D): dùng chung 1 mean/std
    - NMM features (64D): dùng riêng mean/std cho từng chiều
    
    Args:
        data_dir: Thư mục chứa processed_data/data/
        split: 'train', 'dev', hoặc 'test'
    
    Returns:
        dict with keys: manual_mean, manual_std, nmm_mean, nmm_std
    """
    split_dir = os.path.join(data_dir, split)
    poses_dir = os.path.join(split_dir, "poses")
    
    if not os.path.exists(poses_dir):
        raise ValueError(f"Không tìm thấy {poses_dir}")
    
    video_ids = [f.replace('.npz', '') for f in os.listdir(poses_dir) if f.endswith('.npz')]
    
    print(f"Đang tính grouped normalization stats từ {len(video_ids)} samples...")
    
    all_manual = []
    all_nmm = []
    
    for video_id in tqdm(video_ids, desc="Loading poses"):
        pose_214, T = load_sample(video_id, split_dir)
        if pose_214 is not None and T > 0:
            manual = pose_214[:, :150]  # [T, 150]
            nmm = pose_214[:, 150:]     # [T, 64]
            
            all_manual.append(manual)
            all_nmm.append(nmm)
    
    if len(all_manual) == 0:
        raise ValueError("Không có dữ liệu hợp lệ!")
    
    # Stack
    all_manual = np.concatenate(all_manual, axis=0)  # [N, 150]
    all_nmm = np.concatenate(all_nmm, axis=0)        # [N, 64]
    
    # Tính stats
    # Manual: dùng chung 1 mean/std cho toàn bộ
    manual_mean = float(np.mean(all_manual))
    manual_std = float(np.std(all_manual))
    manual_std = max(manual_std, 1e-6)  # Tránh chia 0
    
    # NMM: dùng riêng mean/std cho từng feature
    nmm_mean = np.mean(all_nmm, axis=0)  # [64]
    nmm_std = np.std(all_nmm, axis=0)    # [64]
    nmm_std = np.where(nmm_std < 1e-6, 1.0, nmm_std)
    
    print(f"✅ Grouped Stats:")
    print(f"   Manual: mean={manual_mean:.4f}, std={manual_std:.4f}")
    print(f"   NMM: mean_range=[{nmm_mean.min():.4f}, {nmm_mean.max():.4f}]")
    print(f"        std_range=[{nmm_std.min():.4f}, {nmm_std.max():.4f}]")
    
    return {
        'manual_mean': manual_mean,
        'manual_std': manual_std,
        'nmm_mean': nmm_mean,  # [64]
        'nmm_std': nmm_std     # [64]
    }


def normalize_pose(pose_214, stats):
    """
    ✅ FIXED: Normalize pose với grouped stats
    
    Args:
        pose_214: [T, 214] hoặc [214] numpy array or torch tensor
        stats: dict với keys: manual_mean, manual_std, nmm_mean, nmm_std
    
    Returns:
        normalized_pose: cùng shape với pose_214
    """
    assert 'manual_mean' in stats, "Missing 'manual_mean' in stats"
    assert 'manual_std' in stats, "Missing 'manual_std' in stats"
    assert 'nmm_mean' in stats, "Missing 'nmm_mean' in stats"
    assert 'nmm_std' in stats, "Missing 'nmm_std' in stats"
    
    # Xử lý cả numpy và torch
    is_torch = hasattr(pose_214, 'device')
    
    if is_torch:
        import torch
        manual = pose_214[..., :150]
        nmm = pose_214[..., 150:]
        
        manual_mean = torch.tensor(stats['manual_mean'], device=pose_214.device, dtype=pose_214.dtype)
        manual_std = torch.tensor(stats['manual_std'], device=pose_214.device, dtype=pose_214.dtype)
        nmm_mean = torch.tensor(stats['nmm_mean'], device=pose_214.device, dtype=pose_214.dtype)
        nmm_std = torch.tensor(stats['nmm_std'], device=pose_214.device, dtype=pose_214.dtype)
        
        manual_norm = (manual - manual_mean) / manual_std
        nmm_norm = (nmm - nmm_mean) / nmm_std
        
        return torch.cat([manual_norm, nmm_norm], dim=-1)
    else:
        manual = pose_214[..., :150]
        nmm = pose_214[..., 150:]
        
        manual_norm = (manual - stats['manual_mean']) / stats['manual_std']
        nmm_norm = (nmm - stats['nmm_mean']) / stats['nmm_std']
        
        return np.concatenate([manual_norm, nmm_norm], axis=-1)


def denormalize_pose(normalized_pose, stats):
    """
    ✅ FIXED: Denormalize pose về scale gốc
    
    Args:
        normalized_pose: [T, 214] hoặc [214]
        stats: dict với grouped stats
    
    Returns:
        denormalized_pose: cùng shape
    """
    is_torch = hasattr(normalized_pose, 'device')
    
    if is_torch:
        import torch
        manual_norm = normalized_pose[..., :150]
        nmm_norm = normalized_pose[..., 150:]
        
        manual_mean = torch.tensor(stats['manual_mean'], device=normalized_pose.device, dtype=normalized_pose.dtype)
        manual_std = torch.tensor(stats['manual_std'], device=normalized_pose.device, dtype=normalized_pose.dtype)
        nmm_mean = torch.tensor(stats['nmm_mean'], device=normalized_pose.device, dtype=normalized_pose.dtype)
        nmm_std = torch.tensor(stats['nmm_std'], device=normalized_pose.device, dtype=normalized_pose.dtype)
        
        manual = manual_norm * manual_std + manual_mean
        nmm = nmm_norm * nmm_std + nmm_mean
        
        return torch.cat([manual, nmm], dim=-1)
    else:
        manual_norm = normalized_pose[..., :150]
        nmm_norm = normalized_pose[..., 150:]
        
        manual = manual_norm * stats['manual_std'] + stats['manual_mean']
        nmm = nmm_norm * stats['nmm_std'] + stats['nmm_mean']
        
        return np.concatenate([manual, nmm], axis=-1)


if __name__ == "__main__":
    # Test
    DATA_DIR = "/Users/kieuvo/Learn/Research/SL/Implement/SignFML/slp-vtk/data/RWTH/PHOENIX-2014-T-release-v3/processed_data/data"
    
    # Tính grouped stats
    stats = compute_grouped_normalization_stats(DATA_DIR, split='train')
    
    # Lưu stats
    stats_path = os.path.join(DATA_DIR, "normalization_stats.npz")
    np.savez_compressed(
        stats_path,
        manual_mean=stats['manual_mean'],
        manual_std=stats['manual_std'],
        nmm_mean=stats['nmm_mean'],
        nmm_std=stats['nmm_std']
    )
    print(f"\n✅ Đã lưu stats vào: {stats_path}")
    
    # Test load một sample
    split_dir = os.path.join(DATA_DIR, "train")
    video_ids = [f.replace('.npz', '') for f in os.listdir(os.path.join(split_dir, "poses")) if f.endswith('.npz')]
    
    if video_ids:
        test_id = video_ids[0]
        pose, T = load_sample(test_id, split_dir)
        
        if pose is not None:
            print(f"\n📊 Test sample: {test_id}")
            print(f"  Shape: {pose.shape}, T={T}")
            print(f"  Range: [{pose.min():.4f}, {pose.max():.4f}]")
            
            # Test normalize
            pose_norm = normalize_pose(pose, stats)
            print(f"  Normalized range: [{pose_norm.min():.4f}, {pose_norm.max():.4f}]")
            print(f"  Normalized mean: {pose_norm.mean():.6f}")
            print(f"  Normalized std: {pose_norm.std():.6f}")
            
            # Test denormalize
            pose_denorm = denormalize_pose(pose_norm, stats)
            error = np.abs(pose - pose_denorm).max()
            print(f"  Denormalized error: {error:.6f}")
            
            if error < 1e-5:
                print("  ✅ Normalization test PASSED!")
            else:
                print(f"  ⚠️ Normalization test FAILED! Error: {error}")