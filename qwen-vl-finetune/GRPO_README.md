# GRPO Training for Qwen-VL

This implementation adds Generative Reward Penalized Optimization (GRPO) training support for Qwen-VL models, specifically designed for video understanding tasks with format and accuracy rewards.

## Overview

GRPO (Generative Reward Penalized Optimization) is a reinforcement learning approach that optimizes language models by:
1. Generating multiple response samples for each input
2. Computing rewards for each sample (format + accuracy)
3. Using reward-weighted loss to encourage better responses
4. Adding KL penalty to prevent divergence from the reference model

## Key Features

- **Dual Reward System**: 
  - Format reward: Checks if output follows `<think>...</think> <answer>...</answer>` format
  - Accuracy reward: Compares extracted actions with ground truth
- **Generation-aware**: Properly handles video/image inputs during generation
- **Flash Attention Compatible**: Works with the custom flash attention implementation
- **Multi-GPU Support**: Distributed training with DeepSpeed/FSDP

## Installation

Ensure you have the base requirements installed, then the GRPO implementation is ready to use.

## Usage

### Basic Training

```bash
# Make the script executable
chmod +x scripts/train_grpo.sh

# Run GRPO training
./scripts/train_grpo.sh
```

### Custom Training

```bash
python qwenvl/train/train_grpo.py \
    --model_name_or_path "Qwen/Qwen2.5-VL-3B-Instruct" \
    --dataset_use "assy07_grpo" \
    --output_dir "./output/grpo_model" \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --grpo_alpha 0.5 \
    --grpo_beta 0.1 \
    --format_reward_weight 0.3 \
    --accuracy_reward_weight 0.7 \
    --grpo_sample_size 4
```

### Key Parameters

#### GRPO-specific parameters:
- `grpo_alpha`: Reward scaling factor (default: 0.5)
- `grpo_beta`: KL penalty weight (default: 0.1)
- `format_reward_weight`: Weight for format compliance (default: 0.3)
- `accuracy_reward_weight`: Weight for answer accuracy (default: 0.7)
- `grpo_sample_size`: Number of samples to generate per example (default: 4)
- `generation_temperature`: Temperature for sampling (default: 0.7)
- `generation_top_p`: Top-p for nucleus sampling (default: 0.9)

#### Model tuning parameters:
- `tune_mm_llm`: Fine-tune the language model (default: True)
- `tune_mm_vision`: Fine-tune the vision encoder (default: False)
- `tune_mm_mlp`: Fine-tune the vision-language connector (default: True)

## Dataset Format

The GRPO implementation expects data in the following format:

```json
{
    "id": 0,
    "conversations": [
        {
            "from": "human",
            "value": "<video>\n[Question with options]\nOutput the thinking process in <think> </think> and final answer in <answer> </answer> tags."
        },
        {
            "from": "gpt",
            "value": "<answer>(1) Action 1 (2) Action 2</answer>"
        }
    ],
    "video": "path/to/video.mp4",
    "meta": {
        "frame_cnts": [15, 11],
        "min_frames": 4,
        "max_frames": 8,
        "dynamic_sample": true
    }
}
```

## Reward Functions

### Format Reward (30% by default)
- Checks for `<think>` tags: +0.3
- Checks for `<answer>` tags: +0.3
- Correct order (think before answer): +0.4

### Accuracy Reward (70% by default)
- Exact match: 1.0
- Partial match: IoU of action sets
- No match: 0.0

## Monitoring Training

The trainer logs:
- Training loss
- Format reward (average)
- Accuracy reward (average)
- Total reward (weighted combination)
- KL divergence from reference model

## Tips for Best Results

1. **Batch Size**: Use small batch sizes (1-2) due to generation overhead
2. **Sample Size**: 4-8 samples per example provides good diversity
3. **Learning Rate**: Use lower learning rates (2e-5) for stability
4. **Rewards**: Adjust weights based on your priority (format vs accuracy)
5. **KL Penalty**: Increase `grpo_beta` if model diverges too much

## Troubleshooting

### Out of Memory
- Reduce `grpo_sample_size`
- Reduce `generation_max_length`
- Enable gradient checkpointing
- Reduce batch size

### Poor Generation Quality
- Check if flash attention override is working
- Verify tokenizer chat template
- Ensure proper video frame sampling

### Low Rewards
- Check dataset format matches expected structure
- Verify ground truth answers are properly formatted
- Adjust reward weights

## Implementation Details

The GRPO trainer:
1. Inherits from HuggingFace Trainer
2. Overrides `compute_loss` for GRPO objective
3. Maintains a frozen reference model for KL computation
4. Handles multi-modal inputs during generation
5. Provides custom evaluation with reward metrics

## Citation

If you use this GRPO implementation, please cite:
```bibtex
@misc{qwenvl-grpo,
  title={GRPO Training for Qwen-VL},
  author={Your Name},
  year={2024}
}
``` 