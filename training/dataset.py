"""
PyTorch Dataset cho Sign Language Production
FIXED: Lấy text ở cột cuối cùng [-1] và IN RA MÀN HÌNH để kiểm tra.
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
        self.data_dir = data_dir
        self.split = split
        self.split_dir = os.path.join(data_dir, split)
        self.max_seq_len = max_seq_len
        self.max_text_len = max_text_len
        self.normalize = normalize
        
        # 1. LOAD STATS
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
        
        # 2. SETUP TOKENIZER
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
        
        # 3. LOAD VIDEO IDs TỪ FOLDER POSES
        poses_dir = os.path.join(self.split_dir, "poses")
        if not os.path.exists(poses_dir):
            raise ValueError(f"Không tìm thấy thư mục pose: {poses_dir}")
            
        # Lấy danh sách video có sẵn file pose
        # (Để đảm bảo chỉ load text của những video có file pose)
        available_poses = set([
            f.replace('.npz', '') 
            for f in os.listdir(poses_dir) 
            if f.endswith('.npz')
        ])
        
        self.video_ids = []
        
        # 4. LOAD TEXT & DEBUG PRINT
        self.texts = {}
        if text_file is None:
            text_file = os.path.join(data_dir, f"{split}.txt")
        
        if os.path.exists(text_file):
            print(f"\n📖 Đang đọc text từ: {text_file}")
            print(f"{'='*50}")
            
            count_debug = 0 # Biến đếm để chỉ in 5 dòng đầu tiên
            
            with open(text_file, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if not line: continue
                    
                    # Cắt theo dấu gạch đứng |
                    parts = line.split('|')
                    
                    # Format PHOENIX: ID | ... | ... | German Text
                    if len(parts) >= 2:
                        vid_id = parts[0].strip()
                        
                        # --- SỬA Ở ĐÂY: Lấy phần tử CUỐI CÙNG [-1] ---
                        text_content = parts[-1].strip()
                        
                        # Chỉ lấy nếu video này có file pose
                        if vid_id in available_poses:
                            self.texts[vid_id] = text_content
                            self.video_ids.append(vid_id)
                            
                            # --- 🔥 DEBUG: IN RA 5 DÒNG ĐẦU TIÊN ĐỂ CHỊ CHECK ---
                            if count_debug < 5:
                                print(f"👀 [CHECK DÒNG {count_debug+1}]")
                                print(f"   🎥 ID:   {vid_id}")
                                print(f"   📝 Text: {text_content}")
                                print(f"   -----------------------")
                                count_debug += 1
                                
            print(f"{'='*50}")
            print(f"✅ Đã load {len(self.texts)} dòng text khớp với pose.")
        else:
            print(f"⚠️ CẢNH BÁO: Không tìm thấy file text {text_file}! Text sẽ bị rỗng.")
        
        # Sort lại cho nhất quán
        self.video_ids = sorted(self.video_ids)
        print(f"✅ Init Dataset {split}: {len(self.video_ids)} samples.")
    
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
        
        # Lấy Text
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