# Latent Flow Matching với SSM Prior + Fast Sampler cho Sign Language Production

**Hướng 2 nâng cao**: Flow Matching trên latent space với SSM Prior và Few-step Distillation

Pipeline hoàn chỉnh từ data preparation đến training cho Text → Pose generation.

## Cấu trúc

```
.
├── data_preparation.py          # Load và combine poses + nmms thành 214D
├── dataset.py                   # PyTorch Dataset class
├── models/
│   ├── autoencoder.py          # Stage 1: Autoencoder (Encoder + Decoder)
│   ├── latent_predictor.py     # Stage 2 (Hướng 1): Latent Predictor (Text → Latent)
│   ├── flow_matching.py        # Flow Matching components
│   ├── mamba_prior.py          # SSM/Mamba Prior
│   ├── sync_guidance.py        # Sync Guidance Head
│   ├── latent_flow_matcher.py  # Stage 2 (Hướng 2): Latent Flow Matcher
│   └── consistency_distillation.py  # Few-step Distillation
├── train_stage1_autoencoder.py # Training script cho Stage 1
├── train_stage2_predictor.py   # Training script cho Stage 2 (Hướng 1)
├── train_stage2_latent_flow.py # Training script cho Stage 2 (Hướng 2)
├── inference.py                # Inference script (Hướng 1)
├── inference_latent_flow.py    # Inference script (Hướng 2, fast sampling)
├── evaluate.py                 # Evaluation script
└── README.md                   # File này
```

## Bước 1: Chuẩn bị Data

### 1.1. Tính normalization stats

```bash
python data_preparation.py
```

File này sẽ:
- Load tất cả samples từ `train` split
- Combine poses (150D) + nmms (64D) → 214D
- Tính mean và std
- Lưu vào `{data_dir}/normalization_stats.npz`

**Lưu ý**: Cần chỉnh `DATA_DIR` trong file `data_preparation.py` trước khi chạy.

### 1.2. Cấu trúc data mong đợi

```
processed_data/data/
├── train/
│   ├── poses/
│   │   ├── video1.npz
│   │   └── ...
│   └── nmms/
│       ├── video1.pkl
│       └── ...
├── dev/
│   ├── poses/
│   └── nmms/
├── test/
│   ├── poses/
│   └── nmms/
└── normalization_stats.npz
```

### 1.3. Format file text (optional)

Nếu có file text, đặt ở `{data_dir}/{split}.txt` với format:
```
video_id1|text1
video_id2|text2
...
```

Nếu không có, dataset sẽ dùng text rỗng.

## Bước 2: Training Stage 1 - Autoencoder

Train autoencoder để học latent space:

```bash
python train_stage1_autoencoder.py \
    --data_dir /path/to/processed_data/data \
    --output_dir /path/to/checkpoints/stage1 \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --latent_dim 256 \
    --hidden_dim 512 \
    --max_seq_len 120 \
    --num_workers 4
```

**Tham số quan trọng**:
- `--batch_size`: 32-48 cho A100, 16-24 cho 3090/4090
- `--num_epochs`: 80-120 epochs (early stopping với patience 10-15)
- `--latent_dim`: 256 (có thể tune)
- `--max_seq_len`: 120 (tùy dataset)

**Output**: Checkpoint sẽ được lưu ở `{output_dir}/best_model.pt`

## Bước 3: Training Stage 2 - Chọn Hướng

### Hướng 1: Latent Predictor (Baseline)

Train predictor để map text → latent (direct prediction):

```bash
python train_stage2_predictor.py \
    --data_dir /path/to/processed_data/data \
    --output_dir /path/to/checkpoints/stage2_predictor \
    --autoencoder_checkpoint /path/to/checkpoints/stage1/best_model.pt \
    --batch_size 32 \
    --num_epochs 80 \
    --learning_rate 5e-4 \
    --latent_dim 256 \
    --hidden_dim 512 \
    --max_seq_len 120 \
    --num_workers 4
```

### Hướng 2: Latent Flow Matching (Nâng cao) ⭐

Train Flow Matching trên latent space với SSM Prior và Sync Guidance:

```bash
python train_stage2_latent_flow.py \
    --data_dir /path/to/processed_data/data \
    --output_dir /path/to/checkpoints/stage2_flow \
    --autoencoder_checkpoint /path/to/checkpoints/stage1/best_model.pt \
    --batch_size 32 \
    --num_epochs 100 \
    --learning_rate 1e-4 \
    --latent_dim 256 \
    --hidden_dim 512 \
    --max_seq_len 120 \
    --use_ssm_prior \
    --use_sync_guidance \
    --lambda_prior 0.1 \
    --gamma_guidance 0.01 \
    --num_inference_steps 50 \
    --num_workers 4
```

**Tham số Hướng 2**:
- `--use_ssm_prior`: Bật SSM Prior (structured flow)
- `--use_sync_guidance`: Bật Sync Guidance (đồng bộ tay-mặt)
- `--lambda_prior`: Weight cho SSM prior (0.1, sẽ anneal theo t)
- `--gamma_guidance`: Weight cho sync guidance (0.01)
- `--num_inference_steps`: Số bước ODE khi inference (50 cho full quality)

**Output**: Checkpoint sẽ được lưu ở `{output_dir}/best_model.pt`

## Bước 4: Evaluation

Đánh giá model trên dev/test set với metrics chi tiết:

```bash
python evaluate.py \
    --data_dir /path/to/processed_data/data \
    --predictor_checkpoint /path/to/checkpoints/stage2/best_model.pt \
    --autoencoder_checkpoint /path/to/checkpoints/stage1/best_model.pt \
    --split dev \
    --batch_size 32 \
    --save_predictions \
    --output_dir /path/to/evaluation_results
```

**Metrics được tính**:
- **DTW Distance**: Độ tương đồng về temporal alignment (lower is better)
- **Sync Correlation**: Correlation giữa tay và mặt (higher is better)
- **Sync Lag**: Lag tối ưu giữa tay và mặt (lower is better)
- **MSE Error**: Full pose, Manual (hands), NMM (face) (lower is better)
- **BLEU-1, BLEU-2, BLEU-3, BLEU-4**: Text similarity metrics (higher is better)
- **ROUGE-L**: Text similarity metric (higher is better)

**Lưu ý về BLEU/ROUGE**:
- Hiện tại BLEU/ROUGE chỉ tính được nếu có text ground truth trong dataset
- Để tính đúng, cần có **Pose → Text model** để generate text từ predicted pose
- Hiện tại code sẽ báo warning nếu không có text data hoặc chưa có pose → text model

**Lưu ý**: 
- `--split dev` hoặc `--split test` để chọn split
- `--save_predictions` để lưu predictions chi tiết
- Kết quả sẽ được lưu vào `{output_dir}/metrics_{split}.pkl`

## Bước 5: Inference

### Hướng 1: Direct Prediction

```bash
python inference.py \
    --text "Hello world" \
    --predictor_checkpoint /path/to/checkpoints/stage2_predictor/best_model.pt \
    --autoencoder_checkpoint /path/to/checkpoints/stage1/best_model.pt \
    --output_path output_pose.npy \
    --target_length 50 \
    --data_dir /path/to/processed_data/data
```

### Hướng 2: Flow Matching (Fast Sampling) ⭐

```bash
python inference_latent_flow.py \
    --text "Hello world" \
    --flow_checkpoint /path/to/checkpoints/stage2_flow/best_model.pt \
    --autoencoder_checkpoint /path/to/checkpoints/stage1/best_model.pt \
    --output_path output_pose.npy \
    --target_length 50 \
    --num_steps 4 \
    --use_ssm_prior \
    --use_sync_guidance \
    --data_dir /path/to/processed_data/data
```

**Tham số quan trọng**:
- `--num_steps 4`: Fast sampling (1-4 steps, ~0.25-0.6s/clip)
- `--num_steps 50`: Full quality (30-60 steps, ~0.8-2.0s/clip)

**Output**: File `output_pose.npy` chứa pose [T, 214] (đã denormalize) + latency info

## Cấu hình Model

### Autoencoder (Stage 1)

- **Encoder**: 6 Transformer layers, 8 heads, hidden_dim=512
- **Decoder**: Hierarchical (4 coarse + 4 medium + 6 fine layers)
- **Latent dim**: 256
- **Pose dim**: 214 (150 manual + 64 NMM)

### Latent Predictor (Stage 2)

- **Text encoder**: BERT multilingual
- **Predictor**: 6 Transformer decoder layers, 8 heads
- **Learnable queries**: 100 (sẽ pad/truncate theo target_length)
- **Latent dim**: 256

## Loss Functions

### Stage 1
- **Loss**: MSE reconstruction trên pose space
- Mask out padding positions
- Evaluate trên **dev** set mỗi epoch
- Final evaluation trên **test** set sau khi train xong

### Stage 2
- **Pose loss**: MSE trên pose space (sau decode)
- **Latent loss**: MSE trên latent space (0.1 weight)
- Combined: `loss = pose_loss + 0.1 * latent_loss`
- Evaluate trên **dev** set mỗi epoch
- Final evaluation trên **test** set sau khi train xong

## Evaluation Metrics

Sau khi train xong, chạy `evaluate.py` để có metrics chi tiết:

1. **DTW Distance**: Đo độ tương đồng temporal alignment
   - Lower is better
   - Tính trên toàn bộ pose sequence

2. **Sync Quality (Correlation)**: Đo độ đồng bộ giữa tay và mặt
   - Higher is better (0-1)
   - Correlation giữa manual (tay) và NMM (mặt)

3. **Sync Lag**: Lag tối ưu giữa tay và mặt
   - Lower is better (frames)
   - Tìm lag để maximize correlation

4. **MSE Errors**:
   - Full Pose: MSE trên toàn bộ 214D
   - Manual (Hands): MSE trên 150D đầu (tay)
   - NMM (Face): MSE trên 64D cuối (mặt)

5. **BLEU Scores (1-4)**: Đo độ tương đồng n-gram giữa text reference và candidate
   - Higher is better (0-1)
   - BLEU-1: unigram precision
   - BLEU-2: bigram precision
   - BLEU-3: trigram precision
   - BLEU-4: 4-gram precision
   - **Cần**: Text ground truth + Pose → Text model để generate candidate

6. **ROUGE-L**: Longest Common Subsequence (LCS) based F1 score
   - Higher is better (0-1)
   - Đo độ tương đồng về sequence structure
   - **Cần**: Text ground truth + Pose → Text model để generate candidate

## Data Splits

Dataset đã được tách riêng:
- **Train**: Dùng để training (không bao giờ dùng cho validation/test)
- **Dev**: Dùng để validation trong quá trình training
- **Test**: Chỉ dùng để final evaluation sau khi train xong

**Đảm bảo**:
- Training scripts chỉ load `train` split để train
- Validation chỉ dùng `dev` split
- Test chỉ được evaluate sau khi training hoàn tất

## Tips

1. **Batch size**: Tăng batch size nếu có GPU memory, giảm nếu OOM
2. **Gradient accumulation**: Nếu batch size nhỏ, dùng accumulation để đạt effective batch size lớn hơn
3. **Learning rate**: Có thể dùng warmup (chưa implement, có thể thêm)
4. **Early stopping**: Monitor val loss, stop nếu không cải thiện 10-15 epochs
5. **Resume training**: Dùng `--resume_from {checkpoint_path}` để tiếp tục từ checkpoint

## Troubleshooting

### OOM (Out of Memory)
- Giảm `--batch_size`
- Giảm `--max_seq_len`
- Giảm `--hidden_dim` hoặc `--latent_dim`
- Dùng gradient accumulation

### Training không hội tụ
- Kiểm tra normalization stats đã đúng chưa
- Kiểm tra learning rate (có thể quá lớn/nhỏ)
- Kiểm tra data loading (có lỗi không)

### Latent space không tốt
- Tăng số epochs cho Stage 1
- Tăng `--latent_dim` (nhưng sẽ tốn memory hơn)
- Kiểm tra reconstruction quality trên validation set

## Kiến trúc Hướng 2: Latent Flow + SSM Prior + Fast Sampler

### 1. **Latent Flow Matching**
- Flow Matching trên latent space (256D) thay vì pose space (214D)
- Velocity: `v(z,t|x) = v_flow(z,t|x) + λ(t) * v_prior(z,t) - γ * ∇_z L_sync`

### 2. **SSM Prior**
- Mamba/SSM học drift có cấu trúc thời gian dài hạn
- Regularization: `L_prior = ||v_flow - v_prior||^2`
- Lambda anneal: `λ(t) = λ * (1-t)` (ưu tiên prior ở early timesteps)

### 3. **Sync Guidance**
- Head học correlation tay-mặt trong latent
- Loss: `L_sync = -correlation + lag_penalty`
- Guidance: `v = v - γ * ∇_z L_sync` (chỉ ở middle steps)

### 4. **Few-step Distillation**
- Teacher: ODE 50 steps (full quality)
- Student: ODE 1-4 steps (fast sampling)
- Consistency loss: `||z_student - z_teacher||^2`

## Ablation Studies

Để paper mạnh, cần so sánh:
1. Flow latent vs Flow pose
2. +SSM prior vs không prior
3. +Sync guidance vs không guidance
4. Distill 1/2/4 steps vs full ODE (50 steps)
5. Ảnh hưởng số bước inference lên BLEU/DTW/Sync/latency

## Kỳ vọng Cải thiện

So với Hướng 1:
- **BLEU**: +0.5-1.0
- **DTW**: -5% đến -10%
- **Sync**: +0.03-0.07 corr, -1-3 frames lag
- **Latency**: 0.25-0.6s/clip (với 4 steps) vs 0.2-0.3s/clip (Hướng 1)

## Dependencies

```bash
# Cơ bản
pip install torch transformers accelerate

# Cho SSM Prior (optional, nếu muốn dùng Mamba thật)
pip install mamba-ssm

# Nếu không có mamba-ssm, sẽ tự động dùng SimpleSSMPrior (Transformer-based)
```

