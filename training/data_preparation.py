"""
Data Preparation: Load và combine poses + nmms thành 214D vectors
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


def compute_normalization_stats(data_dir, split='train'):
    """
    Tính mean và std từ train set để normalize
    
    Args:
        data_dir: Thư mục chứa processed_data/data/
        split: 'train', 'dev', hoặc 'test'
    
    Returns:
        mean: [214] numpy array
        std: [214] numpy array
    """
    split_dir = os.path.join(data_dir, split)
    poses_dir = os.path.join(split_dir, "poses")
    
    if not os.path.exists(poses_dir):
        raise ValueError(f"Không tìm thấy {poses_dir}")
    
    # Lấy tất cả video IDs
    video_ids = [f.replace('.npz', '') for f in os.listdir(poses_dir) if f.endswith('.npz')]
    
    print(f"Đang tính normalization stats từ {len(video_ids)} samples...")
    
    all_poses = []
    for video_id in tqdm(video_ids, desc="Loading poses"):
        pose_214, T = load_sample(video_id, split_dir)
        if pose_214 is not None and T > 0:
            all_poses.append(pose_214)
    
    if len(all_poses) == 0:
        raise ValueError("Không có dữ liệu hợp lệ!")
    
    # Stack tất cả poses
    all_poses = np.concatenate(all_poses, axis=0)  # [N, 214]
    
    # Tính mean và std
    mean = np.mean(all_poses, axis=0)  # [214]
    std = np.std(all_poses, axis=0)  # [214]
    
    # Tránh chia cho 0
    std = np.where(std < 1e-6, 1.0, std)
    
    print(f"Mean shape: {mean.shape}, Std shape: {std.shape}")
    print(f"Mean range: [{mean.min():.4f}, {mean.max():.4f}]")
    print(f"Std range: [{std.min():.4f}, {std.max():.4f}]")
    
    return mean, std


def normalize_pose(pose_214, mean, std):
    """
    Normalize pose về zero mean, unit variance
    
    Args:
        pose_214: [T, 214] hoặc [214]
        mean: [214]
        std: [214]
    
    Returns:
        normalized_pose: cùng shape với pose_214
    """
    return (pose_214 - mean) / std


def denormalize_pose(normalized_pose, mean, std):
    """
    Denormalize pose về scale gốc
    
    Args:
        normalized_pose: [T, 214] hoặc [214]
        mean: [214]
        std: [214]
    
    Returns:
        denormalized_pose: cùng shape với normalized_pose
    """
    return normalized_pose * std + mean


if __name__ == "__main__":
    # Test
    DATA_DIR = "/Users/kieuvo/Learn/Research/SL/Implement/SignFML/slp-vtk/data/RWTH/PHOENIX-2014-T-release-v3/processed_data/data"
    
    # Tính stats
    mean, std = compute_normalization_stats(DATA_DIR, split='train')
    
    # Lưu stats
    stats_path = os.path.join(DATA_DIR, "normalization_stats.npz")
    np.savez_compressed(stats_path, mean=mean, std=std)
    print(f"\n✅ Đã lưu stats vào: {stats_path}")
    
    # Test load một sample
    split_dir = os.path.join(DATA_DIR, "train")
    video_ids = [f.replace('.npz', '') for f in os.listdir(os.path.join(split_dir, "poses")) if f.endswith('.npz')]
    if video_ids:
        test_id = video_ids[0]
        pose, T = load_sample(test_id, split_dir)
        if pose is not None:
            print(f"\nTest sample: {test_id}")
            print(f"  Shape: {pose.shape}, T={T}")
            print(f"  Range: [{pose.min():.4f}, {pose.max():.4f}]")
            
            # Test normalize
            pose_norm = normalize_pose(pose, mean, std)
            print(f"  Normalized range: [{pose_norm.min():.4f}, {pose_norm.max():.4f}]")
            
            # Test denormalize
            pose_denorm = denormalize_pose(pose_norm, mean, std)
            print(f"  Denormalized error: {np.abs(pose - pose_denorm).max():.6f}")

