# datasets/phoenix_dataset.py (FIXED VERSION)
"""
PyTorch Dataset for Phoenix2014T
Loads preprocessed data for training
(FIXED: Per-feature normalization & IO bugs)
"""
import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from transformers import BertTokenizer
import os
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence # Dùng hàm pad chuẩn của PyTorch

class PhoenixFlowDataset(Dataset):
    """
    Dataset for Hierarchical Flow + NMM training
    (FIXED: Per-feature normalization & IO bugs)
    """
    def __init__(
        self, 
        split='train',
        # (Em đã sửa đường dẫn này cho khớp với script 05 của chị)
        data_root='SL/Implement/SignTran/data/RWTH/PHOENIX-2014-T-release-v3/processed_data/final', 
        max_length=128,
        augment=False
    ):
        self.split = split
        self.max_length = max_length
        self.augment = augment and (split == 'train')
        
        # --- (FIX 1: LƯU data_root VÀO self) ---
        self.data_root = data_root 
        
        # Load dataset (Giờ dòng này sẽ chạy đúng)
        dataset_file = f"{self.data_root}/{split}_dataset.pkl"
        try:
            with open(dataset_file, 'rb') as f:
                self.data = pickle.load(f)
        except FileNotFoundError:
            print(f"LỖI: Không tìm thấy file {dataset_file}")
            print("Vui lòng chạy script '05_prepare_annotations.py' trước.")
            exit()
            
        print(f"Loaded {len(self.data)} samples for {split}")
        
        # German BERT tokenizer
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-german-cased')
        
        # --- (FIX 3: LOGIC CHUẨN HÓA MỚI - PER-FEATURE) ---
        # 1. Định nghĩa các features chị muốn chuẩn hóa
        self.pose_keys = ['pose_coarse', 'pose_medium', 'pose_fine']
        self.nmm_keys = ['nmm_facial_aus', 'nmm_head_pose', 'nmm_eye_gaze', 'nmm_mouth_shape']
        self.stats = {} # Sẽ chứa mean/std cho TỪNG key

        # 2. Tính toán (nếu là 'train') hoặc Tải (nếu là 'dev'/'test')
        stats_file = f"{self.data_root}/normalization_stats.pkl"
        if split == 'train':
            self._compute_normalization_stats(stats_file)
        else:
            try:
                # --- (FIX 2: Sửa 'wb' -> 'rb' (READ BINARY)) ---
                with open(stats_file, 'rb') as f:
                    self.stats = pickle.load(f)
                print(f"Loaded normalization stats from {stats_file}")
            except FileNotFoundError:
                print(f"LỖI: Không tìm thấy file {stats_file}. Vui lòng chạy split 'train' trước.")
                exit()
        # --- (HẾT FIX 3) ---

    def _compute_normalization_stats(self, stats_file):
        """
        (FIXED) Tính toán mean/std RIÊNG BIỆT cho từng feature
        Chạy trên TOÀN BỘ tập train (không sample)
        """
        print("Computing normalization statistics (trên toàn bộ tập train)...")
        
        data_to_normalize = {key: [] for key in self.pose_keys + self.nmm_keys}
        
        # 1. Gom data (chỉ load key cần, tiết kiệm RAM)
        for sample in tqdm(self.data, desc="Đọc dữ liệu train"):
            for key in data_to_normalize.keys():
                data_to_normalize[key].append(sample[key])
                
        # 2. Tính toán và lưu mean/std cho từng key
        for key, data_list in data_to_normalize.items():
            if not data_list:
                print(f"Warning: Không có data cho key '{key}'")
                continue
                
            # Nối tất cả video của 1 key lại
            concatenated_data = np.concatenate(data_list, axis=0) # Shape: [TotalFrames, D]
            
            # Tính mean, std theo chiều feature (axis=0)
            mean = np.mean(concatenated_data, axis=0)
            std = np.std(concatenated_data, axis=0)
            
            # Xử lý trường hợp std = 0 (gây lỗi /0)
            std[std == 0] = 1.0
            
            self.stats[key] = {
                'mean': torch.FloatTensor(mean),
                'std': torch.FloatTensor(std)
            }
            print(f"  Stats for '{key}': mean_shape={mean.shape}, std_shape={std.shape}")

        # 3. Lưu file stats
        with open(stats_file, 'wb') as f:
            pickle.dump(self.stats, f)
        print(f"Đã lưu normalization stats vào: {stats_file}")
    
    def normalize(self, data, key):
        """(FIXED) Chuẩn hóa data dùng mean/std của key tương ứng"""
        if key not in self.stats:
            return data
        
        # Chuyển data sang tensor nếu nó là numpy
        if isinstance(data, np.ndarray):
            data = torch.FloatTensor(data)
            
        mean = self.stats[key]['mean']
        std = self.stats[key]['std']
        
        return (data - mean) / (std + 1e-8) # Thêm 1e-8 để tránh /0
    
    def denormalize(self, normalized_data, key):
        """(FIXED) Giải chuẩn hóa"""
        if key not in self.stats:
            return normalized_data
        
        mean = self.stats[key]['mean'].to(normalized_data.device)
        std = self.stats[key]['std'].to(normalized_data.device)
        
        return normalized_data * std + mean
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        
        # Tokenize text
        text_tokens = self.tokenizer(
            sample['german_text'],
            return_tensors='pt',
            padding='max_length',
            max_length=self.max_length,
            truncation=True
        )
        
        # --- (FIXED) NORMALIZE RIÊNG BIỆT ---
        pose_coarse = self.normalize(sample['pose_coarse'], 'pose_coarse')
        pose_medium = self.normalize(sample['pose_medium'], 'pose_medium')
        pose_fine = self.normalize(sample['pose_fine'], 'pose_fine')
        
        nmm_facial = self.normalize(sample['nmm_facial_aus'], 'nmm_facial_aus')
        nmm_head = self.normalize(sample['nmm_head_pose'], 'nmm_head_pose')
        nmm_gaze = self.normalize(sample['nmm_eye_gaze'], 'nmm_eye_gaze')
        nmm_mouth = self.normalize(sample['nmm_mouth_shape'], 'nmm_mouth_shape')
        # --- (HẾT FIX) ---

        # Augmentation
        if self.augment:
            # (Augmentation nên được làm *TRƯỚC KHI* normalize
            #  nhưng ta giữ logic cũ của chị cho đơn giản)
            pose_coarse, pose_medium, pose_fine = self._augment_poses(
                pose_coarse, pose_medium, pose_fine
            )
        
        return {
            'video_id': sample['video_id'],
            'german_text': sample['german_text'],
            
            # Text
            'text_tokens': text_tokens['input_ids'].squeeze(0),
            'attention_mask': text_tokens['attention_mask'].squeeze(0),
            
            # Manual (hierarchical)
            'manual_coarse': pose_coarse,
            'manual_medium': pose_medium,
            'manual_fine': pose_fine,
            
            # NMMs
            'nmm_facial_aus': nmm_facial,
            'nmm_head_pose': nmm_head,
            'nmm_eye_gaze': nmm_gaze,
            'nmm_mouth_shape': nmm_mouth,
            
            # Metadata
            'seq_length': sample['duration'],
            'signer': sample['signer']
        }
    
    def _augment_poses(self, coarse, medium, fine):
        """Data augmentation (Giữ nguyên)"""
        speed_factor = np.random.uniform(0.9, 1.1)
        T = len(coarse)
        new_T = int(T * speed_factor)
        
        if new_T > 10:
            indices = np.linspace(0, T-1, new_T).astype(int)
            coarse = coarse[indices]
            medium = medium[indices]
            fine = fine[indices]
        
        # Jitter (nên dùng jitter_std nhỏ hơn vì data đã normalize)
        jitter_std = 0.05 # (Tăng nhẹ so với 0.02)
        coarse = coarse + (torch.randn_like(coarse) * jitter_std)
        medium = medium + (torch.randn_like(medium) * jitter_std)
        fine = fine + (torch.randn_like(fine) * jitter_std)
        
        return coarse, medium, fine

def collate_fn(batch):
    """
    (FIXED) Dùng hàm pad_sequence chuẩn của PyTorch
    """
    # Lọc bỏ sample bị lỗi (nếu có)
    batch = [item for item in batch if item is not None]
    if not batch:
        return None

    # Tách các loại data ra
    keys = batch[0].keys()
    batch_data = {key: [] for key in keys}
    
    for item in batch:
        for key in keys:
            batch_data[key].append(item[key])
            
    # Các keys không cần pad
    final_batch = {
        'video_ids': batch_data['video_id'],
        'german_texts': batch_data['german_text'],
        'signers': batch_data['signer'],
        'text_tokens': torch.stack(batch_data['text_tokens']),
        'attention_mask': torch.stack(batch_data['attention_mask']),
        'seq_lengths': torch.LongTensor(batch_data['seq_length'])
    }
    
    # Tìm max_len thực tế trong batch (sau augmentation)
    max_len = max(final_batch['seq_lengths']).item()
    
    # Các keys cần pad (là các chuỗi pose)
    pad_keys = [
        'manual_coarse', 'manual_medium', 'manual_fine',
        'nmm_facial_aus', 'nmm_head_pose', 'nmm_eye_gaze', 'nmm_mouth_shape'
    ]
    
    for key in pad_keys:
        # Cắt các sequence (đã augment) về đúng max_len
        sequences = [item[key][:max_len] for item in batch]
        # Dùng pad_sequence chuẩn
        padded = pad_sequence(sequences, batch_first=True, padding_value=0.0)
        final_batch[key] = padded
    
    return final_batch

# Test dataset
if __name__ == "__main__":
    from torch.utils.data import DataLoader
    
    print("Testing dataset...")
    # (Sửa đường dẫn data_root cho khớp)
    dataset = PhoenixFlowDataset(
        split='train',
        data_root='SL/Implement/SignTran/data/RWTH/PHOENIX-2014-T-release-v3/processed_data/final'
    )
    
    loader = DataLoader(
        dataset,
        batch_size=4,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2
    )
    
    print("\nTesting one batch:")
    try:
        for batch in loader:
            if batch is None:
                print("Skipped an empty batch")
                continue
            
            print(f"  Batch size: {len(batch['video_ids'])}")
            print(f"  Text tokens: {batch['text_tokens'].shape}")
            print(f"  Manual coarse: {batch['manual_coarse'].shape}")
            print(f"  NMM facial: {batch['nmm_facial_aus'].shape}")
            print(f"  Seq lengths: {batch['seq_lengths']}")
            print(f"  Coarse mean (should be ~0): {batch['manual_coarse'].mean():.4f}")
            break
        
        print("\n✅ Dataset working!")
        
    except Exception as e:
        print(f"\n❌ LỖI KHI TEST DATALOADER: {e}")
        import traceback
        traceback.print_exc()