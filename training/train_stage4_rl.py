#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# KHÔNG CẦN THIẾT LÚC NÀY

"""
STAGE 4: REINFORCEMENT LEARNING FINE-TUNING
Kỹ thuật: Reward-Weighted Regression (RWR) - Ổn định hơn PPO cho Flow Matching.
"""
import os
import sys
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(os.getcwd())
from dataset import SignLanguageDataset, collate_fn
from models.fml.latent_flow_matcher import LatentFlowMatcher
# Giả sử chị có module đánh giá (nếu chưa có thì dùng hàm dummy bên dưới)
# from metrics import compute_back_translation_score 

def dummy_reward_function(poses, text_tokens):
    """
    Hàm giả lập phần thưởng (Reward).
    Trong thực tế, chị thay thế bằng:
    1. Back-Translation Model (Dịch pose ngược lại text -> so sánh độ đúng)
    2. Smoothness (1 / độ rung lắc của joint)
    """
    # Ví dụ: Thưởng cho chuyển động mượt (vận tốc nhỏ)
    # poses: [B, T, D]
    velocity = poses[:, 1:] - poses[:, :-1]
    smoothness_reward = -torch.mean(velocity.abs(), dim=(1,2)) # Càng ít rung càng tốt
    
    # Ở đây em giả lập điểm ngẫu nhiên để code chạy được
    # Chị hãy PLUG MODEL SIGN RECOGNITION CỦA CHỊ VÀO ĐÂY
    fake_semantic_score = torch.rand(poses.shape[0], device=poses.device)
    
    total_reward = fake_semantic_score + (0.1 * smoothness_reward)
    return total_reward

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, required=True)
    p.add_argument("--student_ckpt", type=str, required=True, help="Checkpoint từ Stage 3")
    p.add_argument("--save_dir", type=str, default="./ckpts_stage4_rl")
    p.add_argument("--lr", type=float, default=1e-5) # LR cực nhỏ cho RL
    p.add_argument("--samples_per_prompt", type=int, default=4, help="Sinh bao nhiêu mẫu để chọn lọc")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.makedirs(args.save_dir, exist_ok=True)

    # Load Data & Model
    train_loader = DataLoader(
        SignLanguageDataset(args.data_dir, split="train", max_seq_len=80), # Sequence ngắn hơn cho dễ học
        batch_size=8, shuffle=True, collate_fn=collate_fn
    )
    
    student = LatentFlowMatcher(latent_dim=256, hidden_dim=384).to(device)
    ckpt = torch.load(args.student_ckpt, map_location=device)
    student.load_state_dict(ckpt['model_state_dict'], strict=False)
    student.train()
    
    latent_scale = float(ckpt.get("latent_scale_factor", 1.0))
    optimizer = torch.optim.AdamW(student.parameters(), lr=args.lr)

    print("🚀 START STAGE 4: RL FINE-TUNING (Reward Weighted)")
    
    for epoch in range(10): # RL chỉ cần vài epoch
        total_reward_avg = 0
        pbar = tqdm(train_loader, desc=f"Ep {epoch+1}")
        
        for batch in pbar:
            # 1. Chuẩn bị dữ liệu
            text_tokens = batch[1].to(device)
            attention_mask = batch[2].to(device)
            batch_dict = {'text_tokens': text_tokens, 'attention_mask': attention_mask}
            
            # 2. Sinh mẫu (Sampling)
            # Student sinh ra K mẫu cho cùng 1 câu lệnh
            # Lưu ý: Flow Matcher cần chế độ inference để sinh
            student.eval() 
            generated_trajs = []
            
            with torch.no_grad():
                # Lặp lại batch K lần để sinh nhiều biến thể
                expanded_batch_dict = {
                    k: v.repeat_interleave(args.samples_per_prompt, dim=0) 
                    for k, v in batch_dict.items()
                }
                
                # Gọi hàm sample (sinh từ noise -> latent)
                # Chị cần đảm bảo class LatentFlowMatcher có hàm .sample()
                # Nếu chưa có, nó là quá trình giải ODE (Euler step)
                latents_pred = student.sample(
                    batch=expanded_batch_dict, 
                    steps=10, # Ít step cho nhanh
                    device=device
                )
                
            # 3. Tính Reward
            rewards = dummy_reward_function(latents_pred, expanded_batch_dict['text_tokens'])
            
            # Chuẩn hóa Reward trong batch (để ổn định)
            rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
            weights = torch.exp(rewards) # Biến Reward thành trọng số (Weight)

            # 4. Update Student (Maximizing Reward)
            # Chúng ta train Student sao cho nó sinh ra output GIỐNG với các mẫu có Reward cao
            student.train()
            optimizer.zero_grad()
            
            # Forward pass lại với các mẫu đã sinh (nhưng giờ có Gradient)
            # Mục tiêu: Tối ưu hóa xác suất sinh ra các mẫu tốt (Weighted Regression)
            losses = student(
                batch=expanded_batch_dict,
                gt_latent=latents_pred.detach(), # Coi mẫu vừa sinh là Target
                pose_gt=None,
                mode="train"
            )
            
            # Loss được nhân với Weight (Reward)
            # Mẫu nào Reward thấp -> Weight thấp -> Ít ảnh hưởng Gradient
            weighted_loss = (losses['total'] * weights).mean()
            
            weighted_loss.backward()
            torch.nn.utils.clip_grad_norm_(student.parameters(), 1.0)
            optimizer.step()
            
            total_reward_avg += rewards.mean().item()
            pbar.set_postfix({'Rw': f"{total_reward_avg / (pbar.n + 1):.4f}"})

        # Save
        torch.save(student.state_dict(), os.path.join(args.save_dir, "best_student_rl.pt"))

if __name__ == "__main__":
    main()