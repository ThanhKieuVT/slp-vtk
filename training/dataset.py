"""
PyTorch Dataset cho Sign Language Production
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer
from data_preparation import load_sample, normalize_pose, denormalize_pose


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
        """
        Args:
            data_dir: Thư mục chứa processed_data/data/
            split: 'train', 'dev', hoặc 'test'
            text_file: Đường dẫn đến file text (nếu None, sẽ tìm trong data_dir)
            max_seq_len: Độ dài tối đa của pose sequence
            max_text_len: Độ dài tối đa của text tokens
            normalize: Có normalize pose không
            stats_path: Đường dẫn đến file normalization stats
        """
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
            # Tìm file text trong data_dir
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
        
        # Load pose
        pose_214, T = load_sample(video_id, self.split_dir)
        if pose_214 is None:
            # Fallback: tạo pose zero
            pose_214 = np.zeros((1, 214), dtype=np.float32)
            T = 1
        
        # Normalize
        if self.normalize:
            pose_214 = normalize_pose(pose_214, self.mean, self.std)
        
        # Pad hoặc truncate
        if T > self.max_seq_len:
            pose_214 = pose_214[:self.max_seq_len]
            T = self.max_seq_len
        else:
            # Pad với zeros
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
        
        text_tokens = encoded['input_ids'].squeeze(0)  # [max_text_len]
        attention_mask = encoded['attention_mask'].squeeze(0)  # [max_text_len]
        
        return {
            'video_id': video_id,
            'pose': torch.FloatTensor(pose_214),  # [max_seq_len, 214]
            'seq_length': T,
            'text': text,
            'text_tokens': text_tokens,  # [max_text_len]
            'attention_mask': attention_mask,  # [max_text_len]
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
    poses = torch.stack([item['pose'] for item in batch])  # [B, T, 214]
    seq_lengths = torch.LongTensor([item['seq_length'] for item in batch])  # [B]
    texts = [item['text'] for item in batch]
    text_tokens = torch.stack([item['text_tokens'] for item in batch])  # [B, L]
    attention_masks = torch.stack([item['attention_mask'] for item in batch])  # [B, L]
    
    # Tạo mask cho poses (True = valid, False = padding)
    max_len = poses.shape[1]
    pose_mask = torch.arange(max_len)[None, :] < seq_lengths[:, None]  # [B, T]
    
    return {
        'video_ids': video_ids,
        'poses': poses,  # [B, T, 214]
        'pose_mask': pose_mask,  # [B, T]
        'seq_lengths': seq_lengths,  # [B]
        'texts': texts,
        'text_tokens': text_tokens,  # [B, L]
        'attention_mask': attention_masks,  # [B, L]
    }

