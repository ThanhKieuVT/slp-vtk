# Tên file: utils/data_loader.py
# (Bản gộp của data_preparation.py và dataset.py)
# === PHIÊN BẢN CHUẨN: BERT ===

import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer # <<< DÙNG BERT
from tqdm import tqdm
from pathlib import Path

# ==========================================================
# NỘI DUNG TỪ data_preparation.py (Giữ nguyên)
# ==========================================================
def load_sample(video_id, split_dir):
    # (Code load_sample... giữ nguyên)
    pose_path = os.path.join(split_dir, "poses", f"{video_id}.npz")
    if not os.path.exists(pose_path): return None, 0
    poses = np.load(pose_path)
    kp = poses["keypoints"]
    nmm_path = os.path.join(split_dir, "nmms", f"{video_id}.pkl")
    if not os.path.exists(nmm_path): return None, 0
    with open(nmm_path, 'rb') as f: nmms = pickle.load(f)
    T_pose, T_aus = len(kp), len(nmms["facial_aus"])
    T = min(T_pose, T_aus)
    if T == 0: return None, 0
    manual = kp[:T].reshape(T, -1)
    aus = nmms["facial_aus"][:T]
    head = nmms["head_pose"][:T]
    gaze = nmms["eye_gaze"][:T]
    mouth = nmms["mouth_shape"][:T]
    mouth_flat = mouth.reshape(T, -1)
    nmm = np.concatenate([aus, head, gaze, mouth_flat], axis=-1)
    pose_214 = np.concatenate([manual, nmm], axis=-1)
    return pose_214, T

def compute_normalization_stats(data_dir, split='train'):
    # (Code compute_normalization_stats... giữ nguyên)
    split_dir = os.path.join(data_dir, split)
    poses_dir = os.path.join(split_dir, "poses")
    if not os.path.exists(poses_dir): raise ValueError(f"Không tìm thấy {poses_dir}")
    video_ids = [f.replace('.npz', '') for f in os.listdir(poses_dir) if f.endswith('.npz')]
    print(f"Đang tính normalization stats từ {len(video_ids)} samples...")
    all_poses = []
    for video_id in tqdm(video_ids, desc="Loading poses"):
        pose_214, T = load_sample(video_id, split_dir)
        if pose_214 is not None and T > 0: all_poses.append(pose_214)
    if len(all_poses) == 0: raise ValueError("Không có dữ liệu hợp lệ!")
    all_poses = np.concatenate(all_poses, axis=0)
    mean = np.mean(all_poses, axis=0)
    std = np.std(all_poses, axis=0)
    std = np.where(std < 1e-6, 1.0, std)
    return mean, std

def normalize_pose(pose_214, mean, std):
    return (pose_214 - mean) / std

def denormalize_pose(normalized_pose, mean, std):
    return normalized_pose * std + mean

# ==========================================================
# NỘI DUNG TỪ dataset.py (Phiên bản BERT)
# ==========================================================
class SignLanguageDataset(Dataset):
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
        
        # === DÙNG BERT ===
        self.bert_name = 'bert-base-multilingual-cased'
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_name)
        
        poses_dir = os.path.join(self.split_dir, "poses")
        if not os.path.exists(poses_dir):
            raise ValueError(f"Không tìm thấy {poses_dir}")
        self.video_ids = sorted([f.replace('.npz', '') for f in os.listdir(poses_dir) if f.endswith('.npz')])
        
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
        print(f"✅ Using Tokenizer: {self.bert_name}")

    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        pose_214, T = load_sample(video_id, self.split_dir)
        if pose_214 is None:
            pose_214 = np.zeros((1, 214), dtype=np.float32)
            T = 1
        if self.normalize:
            pose_214 = normalize_pose(pose_214, self.mean, self.std)
        if T > self.max_seq_len:
            pose_214 = pose_214[:self.max_seq_len]
            T = self.max_seq_len
        else:
            pad_len = self.max_seq_len - T
            pose_214 = np.pad(pose_214, ((0, pad_len), (0, 0)), mode='constant')
        
        text = self.texts.get(video_id, "")
        
        # === Tokenize bằng BERT ===
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
            'text_list': text, 
            'text_tokens': text_tokens,
            'attention_mask': attention_mask,
        }
    
    def get_mean_std(self):
        if self.normalize:
            return self.mean, self.std
        return None, None

def collate_fn(batch):
    # (Code collate_fn... giữ nguyên)
    video_ids = [item['video_id'] for item in batch]
    poses = torch.stack([item['pose'] for item in batch])
    seq_lengths = torch.LongTensor([item['seq_length'] for item in batch])
    texts = [item['text_list'] for item in batch]
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    max_len = poses.shape[1]
    pose_mask = torch.arange(max_len)[None, :] < seq_lengths[:, None]
    return {
        'video_ids': video_ids, 'poses': poses, 'pose_mask': pose_mask,
        'seq_lengths': seq_lengths, 'text_list': texts,
        'text_tokens': text_tokens, 'attention_mask': attention_masks,
    }