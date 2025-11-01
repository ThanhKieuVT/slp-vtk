#!/bin/bash

# Complete training pipeline for Hierarchical Flow + NMMs
# (LƯU Ý: Script này chỉ chạy được trên Server, SẼ THẤT BẠI trên Colab do timeout)

echo "======================================"
echo "HIERARCHICAL FLOW + NMM TRAINING"
echo "======================================"

# Configuration
DATA_ROOT="./data/RWTH/processed_data/final"
OUTPUT_ROOT="./checkpoints"
WANDB_PROJECT="slp-hierarchical-flow-phoenix"

# ========== STAGE 1: Manual Flow ==========
echo ""
echo "Stage 1: Training Manual Flow (Coarse + Medium + Fine)"
echo "Expected time: 16-24 hours"
echo "--------------------------------------"

accelerate launch training/train_manual_flow.py \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_ROOT/manual_flow \
    --batch_size 4 \
    --num_epochs 50 \
    --learning_rate 1e-4 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "manual_flow_hierarchical"

if [ $? -ne 0 ]; then
    echo "❌ Manual flow training failed!"
    exit 1
fi

echo "✅ Stage 1 complete!"

# ========== STAGE 2: NMM Flow ==========
echo ""
echo "Stage 2: Training NMM Flow (AUs + Head + Gaze + Mouth)"
echo "Expected time: 12-16 hours"
echo "--------------------------------------"

accelerate launch training/train_nmm_flow.py \
    --data_root $DATA_ROOT \
    --output_dir $OUTPUT_ROOT/nmm_flow \
    --batch_size 8 \
    --num_epochs 40 \
    --learning_rate 1e-4 \
    --share_text_encoder $OUTPUT_ROOT/manual_flow/best_model.pt \
    --freeze_text_encoder \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "nmm_flow_all"

if [ $? -ne 0 ]; then
    echo "❌ NMM flow training failed!"
    exit 1
fi

echo "✅ Stage 2 complete!"

# ========== STAGE 3: Joint Synchronization ==========
echo ""
echo "Stage 3: Training Temporal Synchronization"
echo "Expected time: 8-12 hours"
echo "--------------------------------------"

# --- (PHẦN CODE BỊ THIẾU CỦA CHỊ ĐÂY) ---
accelerate launch training/train_joint_sync.py \
    --data_root $DATA_ROOT \
    --manual_checkpoint $OUTPUT_ROOT/manual_flow/best_model.pt \
    --nmm_checkpoint $OUTPUT_ROOT/nmm_flow/best_model.pt \
    --output_dir $OUTPUT_ROOT/sync \
    --batch_size 4 \
    --num_epochs 30 \
    --learning_rate 5e-5 \
    --wandb_project $WANDB_PROJECT \
    --wandb_run_name "joint_sync"

if [ $? -ne 0 ]; then
    echo "❌ Joint sync training failed!"
    exit 1
fi
# --- (HẾT PHẦN SỬA) ---

echo "✅ Stage 3 complete!"
echo "🎉 ALL STAGES COMPLETE! 🎉"
