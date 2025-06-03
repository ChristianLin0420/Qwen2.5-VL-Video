# GRPO (Group Relative Policy Optimization) for Qwen2-VL

This implementation adds GRPO training capability to the Qwen2-VL fine-tuning framework for multiple-choice video understanding tasks.

## Overview

GRPO is a preference optimization method that uses group-wise comparisons instead of pairwise comparisons. For each training example, it creates a group containing:
- The correct answer (positive example)
- Multiple incorrect answers (negative examples)

The model is trained to assign higher probability to correct answers relative to incorrect ones within each group.

## Implementation Details

### New Components

1. **`qwenvl/data/data_qwen_grpo.py`**: GRPO dataset class that:
   - Loads multiple-choice video data
   - Extracts options from questions
   - Creates groups with correct and incorrect answers

2. **`qwenvl/train/grpo_trainer.py`**: GRPO trainer implementing:
   - Group-wise relative policy optimization loss
   - Temperature-controlled optimization (beta parameter)
   - Seamless integration with existing training infrastructure
   - **Comprehensive GRPO metrics logging to wandb**

3. **`qwenvl/train/train_grpo.py`**: Training script with GRPO-specific configurations

4. **Training Scripts**: Ready-to-use shell scripts for single and multi-GPU training

## GRPO Metrics Logging

The GRPO trainer now provides detailed metrics logging to wandb:

### Tracked Metrics

- **`grpo/policy_seq_log_prob`**: Log probability of generated sequences under the policy model
- **`grpo/kl_div`**: KL divergence between policy and reference model 
- **`grpo/format_reward`**: Reward for following the expected format (`<think>...</think> <answer>...</answer>`)
- **`grpo/accuracy_reward`**: Reward for generating correct answers
- **`grpo/total_reward`**: Weighted combination of format and accuracy rewards
- **`grpo/advantage`**: Advantage values used in GRPO loss computation

### Configuration

You can control the logging frequency using the `--grpo_logging_steps` parameter:

```bash
# Log GRPO metrics every 50 steps (default)
--grpo_logging_steps 50

# Log more frequently (every 10 steps)
--grpo_logging_steps 10

# Log less frequently (every 100 steps)  
--grpo_logging_steps 100
```

### Usage Example

```bash
# In your training script
torchrun --nproc_per_node=8 qwenvl/train/train_grpo.py \
    --model_name_or_path Qwen/Qwen2.5-VL-3B-Instruct \
    --dataset_use "assy07_grpo" \
    --report_to wandb \
    --run_name "qwen2.5-vl-grpo-experiment" \
    --grpo_logging_steps 50 \
    --grpo_beta 0.1 \
    --format_reward_weight 0.3 \
    --accuracy_reward_weight 0.7 \
    # ... other parameters
```

## Dataset Format

Your dataset should contain multiple-choice questions with video understanding tasks. The trainer will automatically:
- Extract multiple choice options from questions
- Create comparison groups for GRPO training
- Compute format and accuracy rewards
- Log detailed metrics to wandb

### Dataset Types and Configuration

The GRPO trainer automatically detects the expected format based on the dataset name:

#### **Thinking Datasets**
- **Dataset name**: Any name that doesn't contain "no_think" (e.g., "assy07_grpo", "video_qa_grpo")
- **Expected output**: `<think>...</think> <answer>...</answer>`
- **Use case**: When you want the model to show its reasoning process

#### **No-Thinking Datasets**  
- **Dataset name**: Must contain "no_think" (e.g., "assy07_grpo_no_think", "video_qa_grpo_no_think")
- **Expected output**: `<answer>...</answer>` only
- **Use case**: When you want concise answers without reasoning

#### **Switching Between Types**

```bash
# For thinking dataset
--dataset_use "assy07_grpo"           # Expects <think>...</think> <answer>...</answer>

# For no-thinking dataset  
--dataset_use "assy07_grpo_no_think"  # Expects <answer>...</answer> only
```

The format reward system will automatically adapt based on your dataset name choice.

### Expected Results

**Thinking Dataset** (`assy07_grpo`):
- ✅ `<think>...</think> <answer>...</answer>` → Reward: 1.0
- ❌ `<answer>...</answer>` → Reward: 0.0 (missing think)
- ❌ Extra content before/after → Reward: 0.0

**No-Thinking Dataset** (`assy07_grpo_no_think`):
- ✅ `<answer>...</answer>` → Reward: 1.0  
- ❌ `<think>...</think> <answer>...</answer>` → Reward: 0.0 (unwanted think)
- ❌ Extra content before/after → Reward: 0.0

## Key Features

### Advantages over Standard Fine-tuning
- **Better preference learning**: Direct optimization for choosing correct answers
- **Robust training**: Less prone to distribution collapse than other preference methods
- **Detailed monitoring**: Comprehensive metrics for understanding training dynamics
- **Format enforcement**: Built-in rewards for following expected response format

### Training Parameters
- `grpo_alpha`: Controls reward weighting (default: 0.5)
- `grpo_beta`: KL penalty strength (default: 0.1) 
- `format_reward_weight`: Weight for format compliance (default: 0.3)
- `accuracy_reward_weight`: Weight for answer correctness (default: 0.7)
- `grpo_sample_size`: Number of responses generated per example (default: 4)
- `grpo_logging_steps`: Frequency of metric logging (default: 50)

The metrics provide valuable insights into:
- **Policy model behavior**: How confidently the model generates responses
- **Training stability**: KL divergence tracking prevents excessive drift
- **Reward components**: Separate tracking of format vs accuracy improvements
- **Optimization dynamics**: Advantage values show how the model learns preferences

This comprehensive logging helps you monitor training progress and adjust hyperparameters for optimal GRPO performance.

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

## Reward Functions

### Format Reward (Binary: 1.0 or 0.0)

The format reward system automatically adapts based on the dataset type:

#### **Thinking Datasets** (when dataset name doesn't contain "no_think")
Expected format: `<think>...</think> <answer>...</answer>`

**Reward = 1.0** when:
- Both `<think>` and `<answer>` tags are present
- `<think>` appears before `<answer>` 
- No extra content before `<think>` or after `</answer>`

**Reward = 0.0** for any format violations:
- Missing `<think>` or `<answer>` tags
- Wrong order (answer before think)
- Extra content outside the expected structure

#### **No-Thinking Datasets** (when dataset name contains "no_think") 
Expected format: `<answer>...</answer>` only

**Reward = 1.0** when:
- `<answer>` tag is present
- No `<think>` tags (discouraged for this dataset type)
- No extra content before `<answer>` or after `</answer>`

**Reward = 0.0** for any format violations:
- Missing `<answer>` tag
- Presence of `<think>` tags
- Extra content outside the expected structure

### Accuracy Reward (0.0 to 1.0)
- **Exact match**: 1.0
- **Partial match**: IoU of action sets  
- **No match**: 0.0

The accuracy reward computation remains the same for both dataset types.

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