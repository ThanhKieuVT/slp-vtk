"""
PyTorch Dataset cho Sign Language Production (Phoenix-2014T)
✅ FIXED: 
  - Tăng max_seq_len lên 400
  - Hỗ trợ grouped normalization
  - Padding động trong collate_fn
  - Sửa lỗi đường dẫn cứng (Hardcoded path)
"""
import os
import pickle
import numpy as np
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer

# Fallback import nếu chạy test độc lập
try:
    from data_preparation import load_sample, normalize_pose
except ImportError:
    def load_sample(video_id, split_dir): return np.random.randn(100, 214), 100
    def normalize_pose(pose, stats): return pose

class SignLanguageDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split=None,  
        text_file=None,
        max_seq_len=400,
        max_text_len=128,
        normalize=True,
        stats_path=None
    ):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.normalize = normalize
        
        # 1. TỰ ĐỘNG PHÁT HIỆN SPLIT VÀ ĐƯỜNG DẪN
        if split is None:
            basename = os.path.basename(data_dir.rstrip('/'))
            if basename in ['train', 'dev', 'test']:
                self.split = basename
                self.split_dir = data_dir
                self.root_dir = os.path.dirname(data_dir)
            else:
                self.split = 'train'
                self.split_dir = os.path.join(data_dir, self.split)
                self.root_dir = data_dir
        else:
            self.split = split
            self.split_dir = os.path.join(data_dir, split)
            self.root_dir = data_dir
        
        print(f"📁 Dataset Config: Split={self.split}, Dir={self.split_dir}")
        
        # 2. LOAD GROUPED STATS (Mean/Std)
        if normalize:
            if stats_path is None:
                stats_path = os.path.join(self.root_dir, "normalization_stats.npz")
            
            if os.path.exists(stats_path):
                stats_data = np.load(stats_path)
                self.stats = {
                    'manual_mean': float(stats_data['manual_mean']),
                    'manual_std': float(stats_data['manual_std']),
                    'nmm_mean': stats_data['nmm_mean'],
                    'nmm_std': stats_data['nmm_std']
                }
                print(f"   ✅ Loaded grouped normalization stats")
            else:
                print(f"   ⚠️ Stats not found at {stats_path}, disabling normalize.")
                self.normalize = False
        
        # 3. SETUP TOKENIZER
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        
        # 4. LOAD VIDEO IDs VÀ TEXT
        poses_dir = os.path.join(self.split_dir, "poses")
        if not os.path.exists(poses_dir):
            print(f"   ❌ Error: Directory not found: {poses_dir}")
            self.video_ids = []
            self.texts = {}
            return

        available_poses = set([f.replace('.npz', '') for f in os.listdir(poses_dir) if f.endswith('.npz')])
        
        self.video_ids = []
        self.texts = {}
        
        if text_file is None:
            text_file = os.path.join(self.root_dir, f"{self.split}.txt")
            
        if os.path.exists(text_file):
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    # ✅ FIX: Split an toàn
                    parts = line.strip().split('|', 1)
                    if len(parts) >= 2:
                        vid_id = parts[0].strip()
                        content = parts[-1].strip()
                        if vid_id in available_poses:
                            self.texts[vid_id] = content
                            self.video_ids.append(vid_id)
            print(f"   ✅ Loaded {len(self.video_ids)} samples with text.")
        else:
            print(f"   ⚠️ Warning: Không tìm thấy file text {text_file}")
            self.video_ids = sorted(list(available_poses))

    def __len__(self):
        return len(self.video_ids)
    
    def __getitem__(self, idx):
        video_id = self.video_ids[idx]
        
        # Load Pose [T, 214]
        pose, T = load_sample(video_id, self.split_dir)
        if pose is None: 
            return None
        
        # Normalize
        if self.normalize:
            pose = normalize_pose(pose, self.stats)
        
        # Crop nếu quá dài
        if T > self.max_seq_len:
            pose = pose[:self.max_seq_len]
            T = self.max_seq_len
            
        # ✅ Tối ưu convert tensor
        pose_tensor = torch.from_numpy(pose).float()
        
        # Get & Tokenize Text
        text = self.texts.get(video_id, "")
        encoded = self.tokenizer(
            text,
            max_length=self.max_text_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'video_id': video_id,
            'pose': pose_tensor,        # [T, 214]
            'seq_length': T,
            'text': text,
            'text_tokens': encoded['input_ids'].squeeze(0),
            'attention_mask': encoded['attention_mask'].squeeze(0),
        }

def collate_fn(batch):
    """
    Gom nhóm dữ liệu và Padding linh hoạt
    """
    batch = [item for item in batch if item is not None]
    if not batch: 
        return None

    video_ids = [item['video_id'] for item in batch]
    texts = [item['text'] for item in batch]
    seq_lengths = torch.LongTensor([item['seq_length'] for item in batch])
    text_tokens = torch.stack([item['text_tokens'] for item in batch])
    attention_masks = torch.stack([item['attention_mask'] for item in batch])
    
    poses_list = [item['pose'] for item in batch]
    poses_padded = torch.nn.utils.rnn.pad_sequence(
        poses_list, 
        batch_first=True, 
        padding_value=0.0
    ) 
    
    # Tạo Mask cho Pose: True = Valid, False = Padding
    max_len = poses_padded.shape[1]
    # ✅ Đảm bảo là BoolTensor
    pose_mask = (torch.arange(max_len)[None, :] < seq_lengths[:, None]).bool()
    
    return {
        'video_ids': video_ids,
        'poses': poses_padded,         
        'pose_mask': pose_mask, 
        'seq_lengths': seq_lengths,
        'text_tokens': text_tokens,
        'attention_mask': attention_masks,
        'texts': texts
    }

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # ✅ FIX: Dùng đường dẫn mặc định an toàn
    parser.add_argument("--data_dir", type=str, default="./processed_data/data", 
                        help="Path to processed data directory")
    args = parser.parse_args()
    
    if os.path.exists(args.data_dir):
        print(f"Testing with: {args.data_dir}")
        dataset = SignLanguageDataset(data_dir=args.data_dir, split='train')
        print(f"Size: {len(dataset)}")
    else:
        print("Please check data path.")