"""
PyTorch Dataset cho Sign Language Production
FIXED: Hỗ trợ cả 2 cách gọi:
  1. SignLanguageDataset(data_dir="root", split="train")  # Cách cũ
  2. SignLanguageDataset(data_dir="root/train")          # Cách mới (auto-detect)
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
    
    Flexible initialization:
    - Option 1: data_dir="root", split="train" → uses "root/train"
    - Option 2: data_dir="root/train", split=None → auto-detect split
    """
    
    def __init__(
        self,
        data_dir,
        split=None,  # ✅ CHANGED: None = auto-detect
        text_file=None,
        max_seq_len=120,
        max_text_len=64,
        normalize=True,
        stats_path=None
    ):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.normalize = normalize
        
        # ✅ AUTO-DETECT SPLIT
        if split is None:
            # Check if data_dir itself is a split folder (train/dev/test)
            basename = os.path.basename(data_dir.rstrip('/'))
            if basename in ['train', 'dev', 'test']:
                self.split = basename
                self.split_dir = data_dir
                # Root dir là parent
                self.root_dir = os.path.dirname(data_dir)
            else:
                # Assume it's root dir, default to 'train'
                self.split = 'train'
                self.split_dir = os.path.join(data_dir, self.split)
                self.root_dir = data_dir
        else:
            # Traditional way: data_dir is root, split is specified
            self.split = split
            self.split_dir = os.path.join(data_dir, split)
            self.root_dir = data_dir
        
        print(f"📁 Dataset Config:")
        print(f"   • Root Dir:  {self.root_dir}")
        print(f"   • Split:     {self.split}")
        print(f"   • Split Dir: {self.split_dir}")
        
        # 1. LOAD STATS
        if normalize:
            if stats_path is None:
                # Look in root dir, not split dir
                stats_path = os.path.join(self.root_dir, "normalization_stats.npz")
            if os.path.exists(stats_path):
                stats = np.load(stats_path)
                self.mean = stats['mean']
                self.std = stats['std']
                print(f"   ✅ Loaded normalization stats from {stats_path}")
            else:
                print(f"   ⚠️  Stats not found: {stats_path}, disabling normalization")
                self.normalize = False
        
        # 2. SETUP TOKENIZER
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        
        # 3. LOAD VIDEO IDs FROM POSES FOLDER
        poses_dir = os.path.join(self.split_dir, "poses")
        if not os.path.exists(poses_dir):
            raise ValueError(f"❌ Poses folder not found: {poses_dir}")
            
        # Get available pose files
        available_poses = set([
            f.replace('.npz', '') 
            for f in os.listdir(poses_dir) 
            if f.endswith('.npz')
        ])
        
        self.video_ids = []
        
        # 4. LOAD TEXT ANNOTATIONS
        self.texts = {}
        if text_file is None:
            # ✅ FIXED: Look for text file in root dir
            text_file = os.path.join(self.root_dir, f"{self.split}.txt")
        
        if os.path.exists(text_file):
            print(f"\n📖 Loading text from: {text_file}")
            print(f"{'='*60}")
            
            count_debug = 0
            
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    parts = line.split('|')
                    
                    # Format: ID | ... | Text (last column)
                    if len(parts) >= 2:
                        vid_id = parts[0].strip()
                        text_content = parts[-1].strip()
                        
                        # Only include if pose file exists
                        if vid_id in available_poses:
                            self.texts[vid_id] = text_content
                            self.video_ids.append(vid_id)
                            
                            # Debug first 5 samples
                            if count_debug < 5:
                                print(f"👀 [Sample {count_debug+1}]")
                                print(f"   🎥 ID:   {vid_id}")
                                print(f"   📝 Text: {text_content}")
                                print(f"   -----------------------")
                                count_debug += 1
                                
            print(f"{'='*60}")
            print(f"✅ Loaded {len(self.texts)} text annotations matching poses")
        else:
            print(f"⚠️  WARNING: Text file not found: {text_file}")
            print(f"   All samples will have empty text!")
        
        # Sort for consistency
        self.video_ids = sorted(self.video_ids)
        print(f"✅ Dataset {self.split}: {len(self.video_ids)} samples\n")
    
    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # Load Pose
        pose_214, T = load_sample(video_id, self.split_dir)
        if pose_214 is None:
            pose_214 = np.zeros((1, 214), dtype=np.float32)
            T = 1
        
        # Normalize
        if self.normalize:
            pose_214 = normalize_pose(pose_214, self.mean, self.std)
        
        # Pad/Crop
        if T > self.max_seq_len:
            pose_214 = pose_214[:self.max_seq_len]
            T = self.max_seq_len
        else:
            pad_len = self.max_seq_len - T
            pose_214 = np.pad(pose_214, ((0, pad_len), (0, 0)), mode='constant')
        
        # Get Text
        text = self.texts.get(video_id, "")
        
        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'video_id': video_id,
            'pose': torch.FloatTensor(pose_214),
            'seq_length': T,
            'text': text,
            'text_tokens': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }
    
    def get_mean_std(self):
        if self.normalize: return self.mean, self.std
        return None, None


def collate_fn(batch):
    video_ids = [item['video_id'] for item in batch]
    poses = torch.stack([item['pose'] for item in batch])
    seq_lengths = torch.LongTensor([item['seq_length'] for item in batch])
    texts = [item['text'] for item in batch]
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    
    max_len = poses.shape[1]
    pose_mask = torch.arange(max_len)[None, :] < seq_lengths[:, None]
    
    return {
        'video_ids': video_ids,
        'poses': poses,         
        'pose_mask': pose_mask, 
        'seq_lengths': seq_lengths,
        'text_tokens': text_tokens,
        'attention_mask': attention_masks,
        'texts': texts
    }