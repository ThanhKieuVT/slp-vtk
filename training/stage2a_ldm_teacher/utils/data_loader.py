# Tên file: utils/data_loader.py
# (Gộp từ 2 file gốc của chị + Sửa lỗi Token Type IDs)

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
    """
    # Load manual poses
    pose_path = os.path.join(split_dir, "poses", f"{video_id}.npz")
    if not os.path.exists(pose_path):
        return None, 0
    poses = np.load(pose_path)
    kp = poses["keypoints"]  #
    
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
    
    # Combine manual
    manual = kp[:T].reshape(T, -1)
    
    # Combine NMMs
    aus = nmms["facial_aus"][:T]
    head = nmms["head_pose"][:T]
    gaze = nmms["eye_gaze"][:T]
    mouth = nmms["mouth_shape"][:T]
    mouth_flat = mouth.reshape(T, -1) # [T, 40]
    
    nmm = np.concatenate([aus, head, gaze, mouth_flat], axis=-1)  #
    
    # Combine thành 214D
    pose_214 = np.concatenate([manual, nmm], axis=-1)
    
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
    
    all_poses = np.concatenate(all_poses, axis=0)  #
    
    # Tính mean và std
    mean = np.mean(all_poses, axis=0)
    std = np.std(all_poses, axis=0)
    
    std = np.where(std < 1e-6, 1.0, std) #
    
    return mean, std

def normalize_pose(pose_214, mean, std):
    """
    Normalize pose
    """
    return (pose_214 - mean) / std

def denormalize_pose(normalized_pose, mean, std):
    """
    Denormalize pose
    """
    return normalized_pose * std + mean

# ==========================================================
# NỘI DUNG TỪ dataset.py (ĐÃ SỬA LỖI TOKEN TYPE ID)
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
        
        # === GIỮ NGUYÊN LOGIC GỐC CỦA CHỊ ===
        # (Giả định file stats_path đã được tạo từ 'train' split)
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
        
        print(f"✅ Loaded {len(self.video_ids)} samples từ {split} split")
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # Load pose
        pose_214, T = load_sample(video_id, self.split_dir)
        if pose_214 is None:
            pose_214 = np.zeros((1, 214), dtype=np.float32)
            T = 1
        
        if self.normalize:
            pose_214 = normalize_pose(pose_214, self.mean, self.std)
        
        # Pad hoặc truncate
        if T > self.max_seq_len:
            pose_214 = pose_214[:self.max_seq_len]
            T = self.max_seq_len
        else:
            pad_len = self.max_seq_len - T
            pose_214 = np.pad(pose_214, ((0, pad_len), (0, 0)), mode='constant')
        
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
        
        # === SỬA LỖI 2 (GPT): Thêm Token Type IDs (BẮT BUỘC cho BERT) ===
        if 'token_type_ids' in encoded:
            token_type_ids = encoded['token_type_ids'].squeeze(0)
        else:
            token_type_ids = torch.zeros_like(text_tokens) 
        
        return {
            'video_id': video_id,
            'pose': torch.FloatTensor(pose_214),
            'seq_length': T,
            'text_list': text, # (Đổi 'text' thành 'text_list' để khớp train_script)
            'text_tokens': text_tokens,
            'attention_mask': attention_mask,
            'token_type_ids': token_type_ids # <<< TRẢ VỀ TYPE IDS
        }
    
    def get_mean_std(self):
        if self.normalize:
            return self.mean, self.std
        return None, None

def collate_fn(batch):
    video_ids = [item['video_id'] for item in batch]
    poses = torch.stack([item['pose'] for item in batch])
    seq_lengths = torch.LongTensor([item['seq_length'] for item in batch])
    texts = [item['text_list'] for item in batch] # (Đã đổi tên)
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    
    # === SỬA LỖI 2 (GPT): Collate token_type_ids ===
    token_type_ids = torch.stack([item['token_type_ids'] for item in batch])
    
    # Tạo mask cho poses
    max_len = poses.shape[1]
    pose_mask = torch.arange(max_len)[None, :] < seq_lengths[:, None]
    
    return {
        'video_ids': video_ids,
        'poses': poses,
        'pose_mask': pose_mask,
        'seq_lengths': seq_lengths,
        'text_list': texts,
        'text_tokens': text_tokens,
        'attention_mask': attention_masks,
        'token_type_ids': token_type_ids # <<< TRẢ VỀ TYPE IDS
    }