#!/bin/bash

# GRPO training script for Qwen-VL with video understanding

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
export OMP_NUM_THREADS=1

# Dataset selection affects format rewards:
# - Names with "no_think": Only <answer>...</answer> expected
# - Names without "no_think": <think>...</think> <answer>...</answer> expected
DATASET_NAME="assy07_grpo_no_think"

# Wandb run name
WANDB_RUN_NAME="qwen2.5-vl-7b-grpo-${DATASET_NAME}-per_token_loss"

# Model and data paths
MODEL_PATH="Qwen/Qwen2.5-VL-7B-Instruct"  # Change to your model path
OUTPUT_DIR="./output/${WANDB_RUN_NAME}"
CACHE_DIR="./cache"

# Training parameters
NUM_TRAIN_EPOCHS=1
BATCH_SIZE=1
GRADIENT_ACCUMULATION_STEPS=8
LEARNING_RATE=2e-5
WARMUP_RATIO=0.03
MODEL_MAX_LENGTH=2048

# GRPO specific parameters
GRPO_ALPHA=0.5
GRPO_BETA=0.1
FORMAT_REWARD_WEIGHT=1.0
ACCURACY_REWARD_WEIGHT=1.0
GENERATION_MAX_LENGTH=64
GENERATION_TEMPERATURE=0.7
GENERATION_TOP_P=0.9
GRPO_SAMPLE_SIZE=1
GRPO_LOGGING_STEPS=1

# Video parameters
VIDEO_MAX_FRAMES=4
VIDEO_MIN_FRAMES=4
BASE_INTERVAL=2

# Run training
torchrun --nproc_per_node=8 --master_port=29500 \
    qwenvl/train/train_grpo.py \
    --model_name_or_path $MODEL_PATH \
    --dataset_use $DATASET_NAME \
    --data_flatten True \
    --output_dir $OUTPUT_DIR \
    --cache_dir $CACHE_DIR \
    --num_train_epochs $NUM_TRAIN_EPOCHS \
    --per_device_train_batch_size $BATCH_SIZE \
    --per_device_eval_batch_size $BATCH_SIZE \
    --gradient_accumulation_steps $GRADIENT_ACCUMULATION_STEPS \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 100 \
    --save_total_limit 3 \
    --learning_rate $LEARNING_RATE \
    --weight_decay 0.1 \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --adam_epsilon 1e-8 \
    --max_grad_norm 1.0 \
    --warmup_ratio $WARMUP_RATIO \
    --lr_scheduler_type "cosine" \
    --logging_steps 10 \
    --tf32 True \
    --model_max_length $MODEL_MAX_LENGTH \
    --gradient_checkpointing False \
    --dataloader_num_workers 4 \
    --report_to none \
    --run_name $WANDB_RUN_NAME \
    --bf16 True \
    --tune_mm_llm True \
    --tune_mm_vision False \
    --tune_mm_mlp True \
    --mm_projector_lr 2e-5 \
    --video_max_frames $VIDEO_MAX_FRAMES \
    --video_min_frames $VIDEO_MIN_FRAMES \
    --base_interval $BASE_INTERVAL \
    --grpo_alpha $GRPO_ALPHA \
    --grpo_beta $GRPO_BETA \
    --format_reward_weight $FORMAT_REWARD_WEIGHT \
    --accuracy_reward_weight $ACCURACY_REWARD_WEIGHT \
    --generation_max_length $GENERATION_MAX_LENGTH \
    --generation_temperature $GENERATION_TEMPERATURE \
    --generation_top_p $GENERATION_TOP_P \
    --grpo_sample_size $GRPO_SAMPLE_SIZE \
    --grpo_logging_steps $GRPO_LOGGING_STEPS