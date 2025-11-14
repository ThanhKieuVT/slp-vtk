# Tên file: utils/data_loader.py
# (Bản gộp của data_preparation.py và dataset.py)

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from tqdm import tqdm
from pathlib import Path

# ==========================================================
# NỘI DUNG TỪ data_preparation.py
# ==========================================================

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
    """
    split_dir = os.path.join(data_dir, split)
    poses_dir = os.path.join(split_dir, "poses")
    
    if not os.path.exists(poses_dir):
        raise ValueError(f"Không tìm thấy {poses_dir}")
    
    video_ids = [f.replace('.npz', '') for f in os.listdir(poses_dir) if f.endswith('.npz')]
    
    print(f"Đang tính normalization stats từ {len(video_ids)} samples...")
    
    all_poses = []
    for video_id in tqdm(video_ids, desc="Loading poses"):
        pose_214, T = load_sample(video_id, split_dir)
        if pose_214 is not None and T > 0:
            all_poses.append(pose_214)
    
    if len(all_poses) == 0:
        raise ValueError("Không có dữ liệu hợp lệ!")
    
    all_poses = np.concatenate(all_poses, axis=0)  # [N, 214]
    
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
    """
    return (pose_214 - mean) / std


def denormalize_pose(normalized_pose, mean, std):
    """
    Denormalize pose về scale gốc
    """
    return normalized_pose * std + mean


# ==========================================================
# NỘI DUNG TỪ dataset.py
# (Bỏ dòng `from data_preparation import ...` vì đã có ở trên)
# ==========================================================

class SignLanguageDataset(Dataset):
    """
    Dataset cho Text → Pose
    """
    
    def __init__(
        self,
        data_dir,
        split='train',
        text_file=None,
        max_seq_len=120,
        max_text_len=64,
        normalize=True,
        stats_path=None
    ):
        self.data_dir = data_dir
        self.split = split
        self.split_dir = os.path.join(data_dir, split)
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.normalize = normalize
        
        # Load normalization stats
        if normalize:
            if stats_path is None:
                stats_path = os.path.join(data_dir, "normalization_stats.npz")
            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                self.mean = stats['mean']
                self.std = stats['std']
            else:
                print(f"⚠️  Không tìm thấy {stats_path}, sẽ không normalize!")
                self.normalize = False
        
        # Load text tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        
        # Load video IDs
        poses_dir = os.path.join(self.split_dir, "poses")
        if not os.path.exists(poses_dir):
            raise ValueError(f"Không tìm thấy {poses_dir}")
        
        self.video_ids = sorted([
            f.replace('.npz', '') 
            for f in os.listdir(poses_dir) 
            if f.endswith('.npz')
        ])
        
        # Load texts
        self.texts = {}
        if text_file is None:
            text_file = os.path.join(data_dir, f"{split}.txt")
        
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    parts = line.strip().split('|')
                    if len(parts) >= 2:
                        vid_id = parts[0].strip()
                        text = '|'.join(parts[1:]).strip()
                        self.texts[vid_id] = text
        else:
            print(f"⚠️  Không tìm thấy {text_file}, sẽ dùng text rỗng!")
        
        print(f"✅ Loaded {len(self.video_ids)} samples từ {split} split")
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # Load pose (Dùng hàm `load_sample` đã định nghĩa ở trên)
        pose_214, T = load_sample(video_id, self.split_dir)
        if pose_214 is None:
            pose_214 = np.zeros((1, 214), dtype=np.float32)
            T = 1
        
        # Normalize (Dùng hàm `normalize_pose` đã định nghĩa ở trên)
        if self.normalize:
            pose_214 = normalize_pose(pose_214, self.mean, self.std)
        
        # Pad hoặc truncate
        if T > self.max_seq_len:
            pose_214 = pose_214[:self.max_seq_len]
            T = self.max_seq_len
        else:
            pad_len = self.max_seq_len - T
            pose_214 = np.pad(pose_214, ((0, pad_len), (0, 0)), mode='constant')
        
        # Load text
        text = self.texts.get(video_id, "")
        
        # Tokenize text
        encoded = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        text_tokens = encoded['input_ids'].squeeze(0)
        attention_mask = encoded['attention_mask'].squeeze(0)
        
        return {
            'video_id': video_id,
            'pose': torch.FloatTensor(pose_214),
            'seq_length': T,
            'text_list': text, # Đổi 'text' thành 'text_list' để hợp với train script
            'text_tokens': text_tokens,
            'attention_mask': attention_mask,
        }
    
    def get_mean_std(self):
        """Trả về mean và std để denormalize"""
        if self.normalize:
            return self.mean, self.std
        return None, None


def collate_fn(batch):
    """
    Collate function cho DataLoader
    """
    video_ids = [item['video_id'] for item in batch]
    poses = torch.stack([item['pose'] for item in batch])
    seq_lengths = torch.LongTensor([item['seq_length'] for item in batch])
    texts = [item['text_list'] for item in batch] # Lấy list text
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    
    # Tạo mask cho poses (True = valid, False = padding)
    max_len = poses.shape[1]
    pose_mask = torch.arange(max_len)[None, :] < seq_lengths[:, None]
    
    return {
        'video_ids': video_ids,
        'poses': poses,
        'pose_mask': pose_mask,
        'seq_lengths': seq_lengths,
        'text_list': texts, # Trả về list text
        'text_tokens': text_tokens,
        'attention_mask': attention_masks,
    }


# ==========================================================
# NỘI DUNG __main__ TỪ data_preparation.py (để test)
# ==========================================================

if __name__ == "__main__":
    # Test
    # (Thay đổi đường dẫn này thành đường dẫn của chị)
    DATA_DIR = "/path/to/your/processed_data/data"
    
    if os.path.exists(DATA_DIR):
        # Tính stats
        mean, std = compute_normalization_stats(DATA_DIR, split='train')
        
        # Lưu stats
        stats_path = os.path.join(DATA_DIR, "normalization_stats.npz")
        np.savez_compressed(stats_path, mean=mean, std=std)
        print(f"\n✅ Đã lưu stats vào: {stats_path}")
        
        # Test Dataloader
        print("\nTesting SignLanguageDataset...")
        dataset = SignLanguageDataset(DATA_DIR, split='train', max_seq_len=120)
        
        if len(dataset) > 0:
            sample = dataset[0]
            print(f"Sample 0 keys: {sample.keys()}")
            print(f"  Pose shape: {sample['pose'].shape}")
            print(f"  Text tokens shape: {sample['text_tokens'].shape}")
            print(f"  Text: {sample['text_list']}")
            print(f"  Seq length: {sample['seq_length']}")
            
            # Test Collate
            print("\nTesting collate_fn...")
            loader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)
            batch = next(iter(loader))
            print(f"Batch keys: {batch.keys()}")
            print(f"  Poses shape: {batch['poses'].shape}")
            print(f"  Pose mask shape: {batch['pose_mask'].shape}")
            print(f"  Text tokens shape: {batch['text_tokens'].shape}")
            print(f"  Text list: {batch['text_list']}")
        else:
            print("Dataset rỗng, không thể test.")
    
    else:
        print(f"⚠️  Không tìm thấy DATA_DIR: {DATA_DIR}")
        print("Bỏ qua test. Vui lòng chạy lại với đúng đường dẫn.")