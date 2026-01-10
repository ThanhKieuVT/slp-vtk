#!/bin/bash
# =============================================================================
# FULL TRAINING PIPELINE FOR SIGN LANGUAGE GENERATION
# Stages: 1) Autoencoder  2) Flow Matching  3) Distillation
# =============================================================================

set -e  # Exit on error

# =============================================================================
# CONFIGURATION
# =============================================================================

# Paths
DATA_DIR="data/RWTH/PHOENIX-2014-T-release-v3/processed_data/data"
BASE_CKPT_DIR="checkpoints"

# Stage 1: Autoencoder with Velocity Loss
STAGE1_DIR="${BASE_CKPT_DIR}/stage1_velocity"
STAGE1_EPOCHS=100
STAGE1_BATCH_SIZE=32
STAGE1_LR=1e-4

# Stage 2: Flow Matching with CFG
STAGE2_DIR="${BASE_CKPT_DIR}/stage2_cfg"
STAGE2_EPOCHS=200
STAGE2_BATCH_SIZE=32
STAGE2_LR=1e-4

# Stage 3: Distillation
STAGE3_DIR="${BASE_CKPT_DIR}/stage3_distilled"
STAGE3_EPOCHS=100
STAGE3_BATCH_SIZE=32
STAGE3_LR=5e-5

# Model hyperparameters
LATENT_DIM=256
HIDDEN_DIM=512
MAX_SEQ_LEN=400

# =============================================================================
# UTILITIES
# =============================================================================

print_header() {
    echo ""
    echo "========================================================================"
    echo "  $1"
    echo "========================================================================"
    echo ""
}

check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        echo "✅ GPU detected:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
        DEVICE="cuda"
    else
        echo "⚠️  No GPU detected, using CPU (will be slow!)"
        DEVICE="cpu"
    fi
}

check_data() {
    if [ ! -d "$DATA_DIR" ]; then
        echo "❌ Error: Data directory not found: $DATA_DIR"
        echo "Please set DATA_DIR to your processed dataset path"
        exit 1
    fi
    echo "✅ Data directory found: $DATA_DIR"
}

# =============================================================================
# STAGE 1: AUTOENCODER WITH VELOCITY LOSS
# =============================================================================

train_stage1() {
    print_header "STAGE 1: Training Autoencoder (with Velocity Loss)"
    
    echo "📦 Configuration:"
    echo "   Output dir: $STAGE1_DIR"
    echo "   Epochs: $STAGE1_EPOCHS"
    echo "   Batch size: $STAGE1_BATCH_SIZE"
    echo "   Learning rate: $STAGE1_LR"
    echo "   Latent dim: $LATENT_DIM"
    echo ""
    
    # Check if already trained
    if [ -f "${STAGE1_DIR}/best_model.pt" ]; then
        echo "⚠️  Stage 1 checkpoint exists. Skip? (y/n)"
        read -r response
        if [ "$response" = "y" ]; then
            echo "⏭️  Skipping Stage 1"
            return
        fi
    fi
    
    python training/train_stage1_autoencoder.py \
        --data_dir "$DATA_DIR" \
        --output_dir "$STAGE1_DIR" \
        --batch_size $STAGE1_BATCH_SIZE \
        --num_epochs $STAGE1_EPOCHS \
        --learning_rate $STAGE1_LR \
        --latent_dim $LATENT_DIM \
        --hidden_dim $HIDDEN_DIM \
        --max_seq_len $MAX_SEQ_LEN \
        --num_workers 4
    
    if [ $? -eq 0 ]; then
        echo "✅ Stage 1 completed successfully!"
    else
        echo "❌ Stage 1 failed!"
        exit 1
    fi
}

# =============================================================================
# STAGE 2: FLOW MATCHING WITH CFG
# =============================================================================

train_stage2() {
    print_header "STAGE 2: Training Flow Matcher (with CFG)"
    
    # Check Stage 1 checkpoint
    if [ ! -f "${STAGE1_DIR}/best_model.pt" ]; then
        echo "❌ Error: Stage 1 checkpoint not found!"
        echo "Please train Stage 1 first"
        exit 1
    fi
    
    echo "📦 Configuration:"
    echo "   AE checkpoint: ${STAGE1_DIR}/best_model.pt"
    echo "   Output dir: $STAGE2_DIR"
    echo "   Epochs: $STAGE2_EPOCHS"
    echo "   Batch size: $STAGE2_BATCH_SIZE"
    echo "   CFG dropout: 0.15"
    echo ""
    
    # Check if already trained
    if [ -f "${STAGE2_DIR}/best_model.pt" ]; then
        echo "⚠️  Stage 2 checkpoint exists. Skip? (y/n)"
        read -r response
        if [ "$response" = "y" ]; then
            echo "⏭️  Skipping Stage 2"
            return
        fi
    fi
    
    python training/train_stage2_latent_flow.py \
        --data_dir "$DATA_DIR" \
        --ae_ckpt "${STAGE1_DIR}/best_model.pt" \
        --save_dir "$STAGE2_DIR" \
        --epochs $STAGE2_EPOCHS \
        --batch_size $STAGE2_BATCH_SIZE \
        --lr $STAGE2_LR \
        --latent_dim $LATENT_DIM \
        --hidden_dim $HIDDEN_DIM \
        --ae_hidden_dim $HIDDEN_DIM \
        --max_seq_len $MAX_SEQ_LEN \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo "✅ Stage 2 completed successfully!"
    else
        echo "❌ Stage 2 failed!"
        exit 1
    fi
}

# =============================================================================
# STAGE 3: DISTILLATION (Optional but recommended for speed)
# =============================================================================

train_stage3() {
    print_header "STAGE 3: Distillation (Optional - for Fast Inference)"
    
    # Check Stage 2 checkpoint
    if [ ! -f "${STAGE2_DIR}/best_model.pt" ]; then
        echo "❌ Error: Stage 2 checkpoint not found!"
        echo "Please train Stage 2 first"
        exit 1
    fi
    
    echo "📦 Configuration:"
    echo "   Teacher: ${STAGE2_DIR}/best_model.pt"
    echo "   Output dir: $STAGE3_DIR"
    echo "   Epochs: $STAGE3_EPOCHS"
    echo ""
    
    echo "⚠️  Stage 3 (Distillation) is optional but recommended."
    echo "   Benefits: 5x faster inference (10 steps vs 50)"
    echo "   Cost: Slightly lower quality (~2% BLEU drop)"
    echo ""
    echo "Train Stage 3? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "⏭️  Skipping Stage 3"
        return
    fi
    
    python training/train_stage3_distill.py \
        --data_dir "$DATA_DIR" \
        --ae_ckpt "${STAGE1_DIR}/best_model.pt" \
        --teacher_ckpt "${STAGE2_DIR}/best_model.pt" \
        --save_dir "$STAGE3_DIR" \
        --epochs $STAGE3_EPOCHS \
        --batch_size $STAGE3_BATCH_SIZE \
        --lr $STAGE3_LR \
        --latent_dim $LATENT_DIM \
        --hidden_dim $HIDDEN_DIM \
        --max_seq_len $MAX_SEQ_LEN \
        --device "$DEVICE"
    
    if [ $? -eq 0 ]; then
        echo "✅ Stage 3 completed successfully!"
    else
        echo "❌ Stage 3 failed!"
        exit 1
    fi
}

# =============================================================================
# EVALUATION
# =============================================================================

run_evaluation() {
    print_header "EVALUATION: Testing on Test Set"
    
    # Determine which checkpoint to evaluate
    if [ -f "${STAGE3_DIR}/best_model.pt" ]; then
        FLOW_CKPT="${STAGE3_DIR}/best_model.pt"
        echo "📊 Using distilled model (Stage 3)"
    elif [ -f "${STAGE2_DIR}/best_model.pt" ]; then
        FLOW_CKPT="${STAGE2_DIR}/best_model.pt"
        echo "📊 Using flow model (Stage 2)"
    else
        echo "❌ No trained model found!"
        exit 1
    fi
    
    echo ""
    echo "Run evaluation? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "⏭️  Skipping evaluation"
        return
    fi
    
    python evaluate.py \
        --data_dir "$DATA_DIR" \
        --ae_ckpt "${STAGE1_DIR}/best_model.pt" \
        --flow_ckpt "$FLOW_CKPT" \
        --test_split test \
        --cfg_scale 1.5 \
        --inference_steps 50 \
        --batch_size 16 \
        --output_dir "eval_results"
    
    if [ $? -eq 0 ]; then
        echo "✅ Evaluation completed!"
        echo "📄 Results saved to: eval_results/eval_results_cfg1.5.json"
    fi
}

# =============================================================================
# MAIN PIPELINE
# =============================================================================

main() {
    print_header "🚀 SIGN LANGUAGE GENERATION - FULL TRAINING PIPELINE"
    
    echo "This script will train the complete 3-stage pipeline:"
    echo "   Stage 1: Autoencoder (with Velocity Loss)"
    echo "   Stage 2: Flow Matching (with CFG)"
    echo "   Stage 3: Distillation (optional)"
    echo ""
    echo "Estimated time:"
    echo "   Stage 1: 6-12 hours (single GPU)"
    echo "   Stage 2: 12-24 hours (single GPU)"
    echo "   Stage 3: 6-12 hours (optional)"
    echo ""
    
    # Checks
    check_gpu
    check_data
    
    echo ""
    echo "Continue? (y/n)"
    read -r response
    if [ "$response" != "y" ]; then
        echo "Aborted."
        exit 0
    fi
    
    # Create checkpoint directories
    mkdir -p "$STAGE1_DIR" "$STAGE2_DIR" "$STAGE3_DIR"
    
    # Training pipeline
    train_stage1
    train_stage2
    train_stage3
    
    # Evaluation
    run_evaluation
    
    # Summary
    print_header "✅ TRAINING PIPELINE COMPLETED!"
    
    echo "📦 Checkpoints saved to:"
    echo "   Stage 1: $STAGE1_DIR"
    echo "   Stage 2: $STAGE2_DIR"
    if [ -f "${STAGE3_DIR}/best_model.pt" ]; then
        echo "   Stage 3: $STAGE3_DIR"
    fi
    echo ""
    echo "🎯 Next steps:"
    echo "   1. Check eval_results/ for metrics"
    echo "   2. Try CFG scale sweep (1.0-3.0)"
    echo "   3. Visualize results with visualize_reconstruction.py"
    echo "   4. Write paper! 📝"
    echo ""
}

# =============================================================================
# RUN
# =============================================================================

main "$@"
