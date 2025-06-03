import os
import re
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Union, Any
from dataclasses import dataclass, field
import numpy as np
from tqdm import tqdm
from transformers import Trainer, TrainingArguments
from transformers.trainer_utils import EvalLoopOutput, has_length
from transformers.modeling_outputs import CausalLMOutputWithPast
from torch.utils.data import DataLoader
import transformers
from collections import defaultdict
import copy
import wandb  # Add direct wandb import
from datetime import datetime

from qwenvl.train.trainer import replace_qwen2_vl_attention_class
from qwenvl.train.grpo_flash_attn_override import replace_qwen2_vl_attention_class_grpo


@dataclass
class GRPOTrainingArguments(TrainingArguments):
    """Training arguments for GRPO training."""
    # GRPO specific arguments
    grpo_alpha: float = field(default=0.5, metadata={"help": "GRPO alpha parameter for reward weighting"})
    grpo_beta: float = field(default=0.1, metadata={"help": "GRPO beta parameter for KL penalty"})
    format_reward_weight: float = field(default=1.0, metadata={"help": "Weight for format reward"})
    accuracy_reward_weight: float = field(default=1.0, metadata={"help": "Weight for accuracy reward"})
    generation_max_length: int = field(default=512, metadata={"help": "Maximum length for generation"})
    generation_temperature: float = field(default=0.7, metadata={"help": "Temperature for generation"})
    generation_top_p: float = field(default=0.9, metadata={"help": "Top-p for generation"})
    generation_num_beams: int = field(default=1, metadata={"help": "Number of beams for generation"})
    grpo_sample_size: int = field(default=4, metadata={"help": "Number of samples to generate per example"})
    grpo_logging_steps: int = field(default=50, metadata={"help": "Log GRPO metrics every N steps"})
    
    # Extend parent class arguments
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(default=512)
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None


class GRPOTrainer(Trainer):
    """GRPO Trainer for Qwen-VL models with custom reward functions."""
    
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        processing_class=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        wandb_run=None,  # Add wandb_run parameter
    ):
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )
        
        # Store wandb run instance
        self.wandb_run = wandb_run
        
        # Initialize reference model for KL computation
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # Initialize metric tracking for GRPO logging
        self.grpo_metrics = {
            'policy_seq_log_prob': [],
            'kl_div': [],
            'format_reward': [],
            'accuracy_reward': [],
            'total_reward': [],
            'advantage': []
        }
        self._global_step_count = 0
        
        # Determine if dataset requires thinking process based on dataset name
        dataset_name = getattr(args, 'dataset_use', '')
        self.requires_thinking = 'no_think' not in dataset_name.lower()
        
        # Initialize JSON output logging
        self.output_log_file = os.path.join(args.output_dir, f"grpo_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.outputs_buffer = []
        self.max_buffer_size = 50  # Save to file every 50 samples
        
        # DON'T replace attention here - only during training
        
    def log_grpo_metrics(self, step: Optional[int] = None):
        """Log accumulated GRPO metrics directly to wandb."""
        if not self.grpo_metrics['policy_seq_log_prob']:  # No metrics collected yet
            return
            
        # Calculate averages
        avg_metrics = {}
        for key, values in self.grpo_metrics.items():
            if values:
                avg_metrics[f"grpo/{key}"] = sum(values) / len(values)
        
        # Add step information
        if step is None:
            step = self.state.global_step
            
        # Log directly to wandb using passed wandb_run instance
        try:
            if self.wandb_run is not None and torch.distributed.get_rank() == 0:
                self.wandb_run.log(avg_metrics, step=step)
        except Exception as e:
            print(f"Error logging to wandb: {e}")
            print(f"Metrics that would be logged: {avg_metrics}")
        
        # Clear metrics after logging
        for key in self.grpo_metrics:
            self.grpo_metrics[key].clear()
            
        # Print metrics for monitoring
        metric_str = ", ".join([f"{k.split('/')[-1]}: {v:.4f}" for k, v in avg_metrics.items()])
        print(f"Step {step} - GRPO Metrics: {metric_str}")
    
    def save_output_sample(self, prompt: str, generated_text: str, ground_truth: str, 
                          format_reward: float, accuracy_reward: float, step: int):
        """Save a sample of generated output to buffer and file."""
        sample = {
            "step": step,
            "timestamp": datetime.now().isoformat(),
            "prompt": prompt,
            "generated_text": generated_text,
            "ground_truth": ground_truth,
            "format_reward": format_reward,
            "accuracy_reward": accuracy_reward,
            "total_reward": (self.args.format_reward_weight * format_reward + 
                           self.args.accuracy_reward_weight * accuracy_reward)
        }
        
        self.outputs_buffer.append(sample)
        
        # Save to file when buffer is full
        if len(self.outputs_buffer) >= self.max_buffer_size:
            self._flush_outputs_to_file()
    
    def _flush_outputs_to_file(self):
        """Flush buffered outputs to JSON file."""
        if not self.outputs_buffer:
            return
            
        # Read existing data if file exists
        existing_data = []
        if os.path.exists(self.output_log_file):
            try:
                with open(self.output_log_file, 'r', encoding='utf-8') as f:
                    existing_data = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                existing_data = []
        
        # Append new data
        existing_data.extend(self.outputs_buffer)
        
        # Write back to file
        os.makedirs(os.path.dirname(self.output_log_file), exist_ok=True)
        with open(self.output_log_file, 'w', encoding='utf-8') as f:
            json.dump(existing_data, f, indent=2, ensure_ascii=False)
        
        print(f"Saved {len(self.outputs_buffer)} output samples to {self.output_log_file}")
        self.outputs_buffer.clear()
    
    def _extract_prompt_from_input(self, input_ids: torch.Tensor) -> str:
        """Extract the prompt part (before assistant response) from input_ids."""
        # Find assistant token position
        assistant_token = "<|im_start|>assistant\n"
        assistant_ids = self.processing_class.encode(assistant_token, add_special_tokens=False)
        
        seq = input_ids.cpu().tolist()
        
        # Find where assistant response starts
        for i in range(len(seq) - len(assistant_ids)):
            if seq[i:i+len(assistant_ids)] == assistant_ids:
                # Extract everything before assistant response
                prompt_ids = seq[:i + len(assistant_ids)]
                return self.processing_class.decode(prompt_ids, skip_special_tokens=False)
        
        # If no assistant token found, return full sequence
        return self.processing_class.decode(seq, skip_special_tokens=False)
    
    def training_step(self, model, inputs, num_items_in_batch=None):
        """Override training step to handle GRPO metric logging."""
        # Increment step counter
        self._global_step_count += 1
        
        # Call parent training step
        loss = super().training_step(model, inputs, num_items_in_batch)
        
        # Log GRPO metrics if it's time
        if self._global_step_count % self.args.grpo_logging_steps == 0:
            self.log_grpo_metrics(step=self.state.global_step)
        
        return loss
        
    def compute_format_reward(self, generated_text: str) -> float:
        """
        Compute format reward based on dataset type and expected format.
        - Thinking datasets: <think>...</think> <answer>...</answer>
        - No-thinking datasets: <answer>...</answer> only
        Returns 1.0 for correct format, 0.0 otherwise.
        """
        # Clean the generated text - strip whitespace
        text = generated_text.strip()
        
        if self.requires_thinking:
            # Dataset requires thinking process
            # Expected format: <think>...</think> <answer>...</answer>
            
            # Check for both patterns
            think_pattern = r'<think>.*?</think>'
            answer_pattern = r'<answer>.*?</answer>'
            
            think_match = re.search(think_pattern, text, re.DOTALL)
            answer_match = re.search(answer_pattern, text, re.DOTALL)
            
            # Both must be present
            if not (think_match and answer_match):
                return 0.0
            
            # Check correct order (think before answer)
            if think_match.end() > answer_match.start():
                return 0.0
            
            # Check for unwanted content before <think>
            think_start = think_match.start()
            if think_start > 0 and text[:think_start].strip():
                return 0.0  # Extra content before <think>
            
            # Check for unwanted content after </answer>
            answer_end = answer_match.end()
            if answer_end < len(text) and text[answer_end:].strip():
                return 0.0  # Extra content after </answer>
            
            return 1.0
            
        else:
            # Dataset does not require thinking process
            # Expected format: <answer>...</answer> only
            
            answer_pattern = r'<answer>.*?</answer>'
            answer_match = re.search(answer_pattern, text, re.DOTALL)
            
            # Answer tag must be present
            if not answer_match:
                return 0.0
            
            # Check for unwanted content before <answer>
            answer_start = answer_match.start()
            if answer_start > 0 and text[:answer_start].strip():
                return 0.0  # Extra content before <answer>
            
            # Check for unwanted content after </answer>
            answer_end = answer_match.end()
            if answer_end < len(text) and text[answer_end:].strip():
                return 0.0  # Extra content after </answer>
            
            # Check that there's no thinking tag (should be discouraged)
            think_pattern = r'<think>.*?</think>'
            if re.search(think_pattern, text, re.DOTALL):
                return 0.0  # Thinking tag present when not expected
            
            return 1.0
    
    def compute_accuracy_reward(self, generated_text: str, ground_truth: str) -> float:
        """
        Compute accuracy reward based on whether the generated answer matches ground truth.
        """
        # Extract answer from generated text
        answer_pattern = r'<answer>(.*?)</answer>'
        generated_match = re.search(answer_pattern, generated_text, re.DOTALL)
        
        if not generated_match:
            return 0.0
            
        generated_answer = generated_match.group(1).strip()
        
        # Extract answer from ground truth
        gt_match = re.search(answer_pattern, ground_truth, re.DOTALL)
        if gt_match:
            gt_answer = gt_match.group(1).strip()
        else:
            gt_answer = ground_truth.strip()
        
        # Normalize answers - extract action numbers
        def extract_actions(text):
            # Pattern to match action numbers like (1), (2), etc.
            actions = re.findall(r'\((\d+)\)', text)
            return sorted(actions)
        
        generated_actions = extract_actions(generated_answer)
        gt_actions = extract_actions(gt_answer)
        
        # Calculate accuracy
        if generated_actions == gt_actions:
            return 1.0
        elif set(generated_actions) & set(gt_actions):  # Partial match
            intersection = len(set(generated_actions) & set(gt_actions))
            union = len(set(generated_actions) | set(gt_actions))
            return intersection / union if union > 0 else 0.0
        else:
            return 0.0
    
    def compute_rewards(self, generated_texts: List[str], ground_truths: List[str]) -> torch.Tensor:
        """Compute combined rewards for generated texts."""
        rewards = []
        
        for gen_text, gt_text in zip(generated_texts, ground_truths):
            format_reward = self.compute_format_reward(gen_text)
            accuracy_reward = self.compute_accuracy_reward(gen_text, gt_text)
            
            # Combine rewards with weights
            total_reward = (
                self.args.format_reward_weight * format_reward +
                self.args.accuracy_reward_weight * accuracy_reward
            )
            rewards.append(total_reward)
            
        return torch.tensor(rewards, device=self.model.device)
    
    def generate_responses(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[str], torch.Tensor]:
        """Generate multiple responses for each input in the batch."""
        self.model.eval()
        
        input_ids = batch["input_ids"]
        batch_size = input_ids.shape[0]
        
        # Check if we're using flattened data format
        is_flattened = "attention_mask" in batch and batch["attention_mask"].dim() == 1
        
        # Find where the assistant response should start
        eos_token_id = self.processing_class.eos_token_id
        assistant_token = "<|im_start|>assistant\n"
        assistant_ids = self.processing_class.encode(assistant_token, add_special_tokens=False)
        
        # Generate multiple samples per input
        all_generated_texts = []
        all_generated_ids = []
        
        with torch.no_grad():
            for i in range(batch_size):
                # Find the position to start generation
                single_input_ids = input_ids[i:i+1]
                
                # Find where to cut off for generation (before assistant response)
                seq = single_input_ids[0].cpu().tolist()
                cut_position = len(seq)
                
                # Search for the last occurrence of assistant marker (search backwards)
                for j in range(len(seq) - len(assistant_ids), -1, -1):
                    if j >= 0 and seq[j:j+len(assistant_ids)] == assistant_ids:
                        # Found assistant marker, cut just before it
                        cut_position = j
                        break
                
                # Prepare generation input
                gen_input_ids = single_input_ids[:, :cut_position]
                
                # Prepare other inputs based on data format
                gen_batch = {
                    "input_ids": gen_input_ids,
                }
                
                if is_flattened:
                    # For flattened format, create standard attention mask for generation
                    gen_batch["attention_mask"] = torch.ones_like(gen_input_ids, dtype=torch.bool)
                    # Handle position_ids for flattened format
                    if "position_ids" in batch and batch["position_ids"] is not None:
                        # Position IDs are 3D: [3, batch_size, seq_len]
                        gen_batch["position_ids"] = batch["position_ids"][:, i:i+1, :cut_position]
                else:
                    # Standard format
                    gen_batch["attention_mask"] = batch["attention_mask"][i:i+1, :cut_position] if "attention_mask" in batch else None
                    gen_batch["position_ids"] = batch["position_ids"][i:i+1, :cut_position] if "position_ids" in batch else None
                
                # Add visual inputs if present
                if "pixel_values" in batch and batch["pixel_values"] is not None:
                    gen_batch["pixel_values"] = batch["pixel_values"][i:i+1] if batch["pixel_values"].dim() > 1 else batch["pixel_values"]
                    if "image_grid_thw" in batch and batch["image_grid_thw"] is not None:
                        gen_batch["image_grid_thw"] = batch["image_grid_thw"][i:i+1] if batch["image_grid_thw"].dim() > 1 else batch["image_grid_thw"]
                        
                if "pixel_values_videos" in batch and batch["pixel_values_videos"] is not None:
                    # Handle video data - might be already extracted for the specific sample
                    if i == 0 and batch["pixel_values_videos"].shape[0] > 0:
                        gen_batch["pixel_values_videos"] = batch["pixel_values_videos"]
                        if "video_grid_thw" in batch and batch["video_grid_thw"] is not None:
                            gen_batch["video_grid_thw"] = batch["video_grid_thw"]
                
                # Generate multiple samples
                for _ in range(self.args.grpo_sample_size):
                    outputs = self.model.generate(
                        **gen_batch,
                        max_new_tokens=self.args.generation_max_length,
                        temperature=self.args.generation_temperature,
                        top_p=self.args.generation_top_p,
                        num_beams=self.args.generation_num_beams,
                        do_sample=True,
                        pad_token_id=self.processing_class.pad_token_id,
                        eos_token_id=eos_token_id,
                    )
                    
                    # Extract only the generated part
                    generated_ids = outputs[:, gen_input_ids.shape[1]:]
                    generated_text = self.processing_class.decode(generated_ids[0], skip_special_tokens=True)
                    
                    all_generated_texts.append(generated_text)
                    all_generated_ids.append(generated_ids)
        
        # Stack all generated ids
        max_len = max(ids.shape[1] for ids in all_generated_ids)
        padded_ids = []
        for ids in all_generated_ids:
            if ids.shape[1] < max_len:
                padding = torch.full(
                    (ids.shape[0], max_len - ids.shape[1]), 
                    self.processing_class.pad_token_id,
                    device=ids.device
                )
                ids = torch.cat([ids, padding], dim=1)
            padded_ids.append(ids)
        
        all_generated_ids = torch.cat(padded_ids, dim=0)
        
        return all_generated_texts, all_generated_ids
    
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute GRPO loss while maintaining visual conditioning."""
        
        # DON'T apply GRPO flash attention override during loss computation
        # because we're modifying sequence lengths
        
        # Get ground truth and generate responses
        labels = inputs.get("labels")
        batch_size = inputs["input_ids"].shape[0]
        
        ground_truths = self._extract_ground_truths(labels)
        
        # Apply GRPO flash attention override ONLY for generation
        from qwenvl.train.grpo_flash_attn_override import replace_qwen2_vl_attention_class_grpo
        replace_qwen2_vl_attention_class_grpo()
        
        generated_texts, generated_ids = self.generate_responses(inputs)
        
        # Restore standard flash attention for loss computation
        # from qwenvl.train.trainer import replace_qwen2_vl_attention_class
        # replace_qwen2_vl_attention_class()
        
        # Compute rewards and advantages
        rewards = self.compute_rewards(generated_texts, ground_truths * self.args.grpo_sample_size)
        rewards = rewards.view(batch_size, self.args.grpo_sample_size)
        reward_advantages = rewards - rewards.mean(dim=1, keepdim=True)
        
        # Collect individual reward components for logging
        batch_format_rewards = []
        batch_accuracy_rewards = []
        for gen_text, gt_text in zip(generated_texts, ground_truths * self.args.grpo_sample_size):
            format_reward = self.compute_format_reward(gen_text)
            accuracy_reward = self.compute_accuracy_reward(gen_text, gt_text)
            batch_format_rewards.append(format_reward)
            batch_accuracy_rewards.append(accuracy_reward)
        
        # Save sample outputs to JSON (save every few steps to avoid too much I/O)
        if self._global_step_count % (self.args.grpo_logging_steps * 2) == 0:
            for i in range(min(batch_size, 2)):  # Save up to 2 samples per logging step
                sample_idx = i * self.args.grpo_sample_size  # Take first sample for each input
                prompt = self._extract_prompt_from_input(inputs["input_ids"][i])
                gen_text = generated_texts[sample_idx]
                gt_text = ground_truths[i]
                format_reward = batch_format_rewards[sample_idx]
                accuracy_reward = batch_accuracy_rewards[sample_idx]
                
                self.save_output_sample(
                    prompt=prompt,
                    generated_text=gen_text,
                    ground_truth=gt_text,
                    format_reward=format_reward,
                    accuracy_reward=accuracy_reward,
                    step=self.state.global_step
                )
        
        total_loss = 0.0
        model.train()
        
        # Lists to collect metrics for this batch
        batch_policy_log_probs = []
        batch_kl_divs = []
        batch_advantages = []
        
        for i in range(batch_size):
            for j in range(self.args.grpo_sample_size):
                idx = i * self.args.grpo_sample_size + j
                sample_generated_ids = generated_ids[idx:idx+1]
                
                # Use original sequence structure, replace assistant response
                modified_input_ids, modified_labels = self._replace_assistant_response(
                    inputs["input_ids"][i:i+1], 
                    labels[i:i+1], 
                    sample_generated_ids
                )

                modified_input_ids = inputs["input_ids"][i:i+1]
                modified_labels = labels[i:i+1]
                
                # Create standard attention mask for modified sequence
                modified_attention_mask = torch.ones_like(modified_input_ids, dtype=torch.bool)
                
                # Forward pass with standard attention (no cu_seqlens)
                outputs = model(
                    input_ids=modified_input_ids,
                    labels=modified_labels,
                    attention_mask=modified_attention_mask,
                    # Don't use position_ids from flattened format
                    pixel_values=inputs.get("pixel_values")[i:i+1] if "pixel_values" in inputs and inputs["pixel_values"] is not None else None,
                    image_grid_thw=inputs.get("image_grid_thw")[i:i+1] if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None else None,
                    pixel_values_videos=inputs.get("pixel_values_videos") if "pixel_values_videos" in inputs and inputs["pixel_values_videos"] is not None else None,
                    video_grid_thw=inputs.get("video_grid_thw") if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None else None,
                )
                
                # Reference model with standard attention
                with torch.no_grad():
                    ref_outputs = self.ref_model(
                        input_ids=modified_input_ids,
                        attention_mask=modified_attention_mask,
                        pixel_values=inputs.get("pixel_values")[i:i+1] if "pixel_values" in inputs and inputs["pixel_values"] is not None else None,
                        image_grid_thw=inputs.get("image_grid_thw")[i:i+1] if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None else None,
                        pixel_values_videos=inputs.get("pixel_values_videos") if "pixel_values_videos" in inputs and inputs["pixel_values_videos"] is not None else None,
                        video_grid_thw=inputs.get("video_grid_thw") if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None else None,
                    )
                
                # ===== GRPO LOSS CALCULATION =====
                
                # Get logits for generated tokens only
                policy_logits = outputs.logits
                ref_logits = ref_outputs.logits
                
                # Find the positions of generated tokens
                gen_start, gen_end = self._find_generated_token_positions(modified_labels[0])
                
                if gen_start >= gen_end:
                    continue  # Skip if no generated tokens
                
                # Extract logits for generated sequence (shift by 1 for causal LM)
                policy_gen_logits = policy_logits[0, gen_start-1:gen_end-1, :]  # [seq_len, vocab_size]
                ref_gen_logits = ref_logits[0, gen_start-1:gen_end-1, :]
                
                # Get target tokens
                target_tokens = modified_input_ids[0, gen_start:gen_end]  # [seq_len]
                
                # Compute log probabilities
                policy_log_probs = F.log_softmax(policy_gen_logits, dim=-1)  # [seq_len, vocab_size]
                ref_log_probs = F.log_softmax(ref_gen_logits, dim=-1)
                
                # Get log probabilities for actual tokens
                policy_token_log_probs = policy_log_probs.gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)  # [seq_len]
                ref_token_log_probs = ref_log_probs.gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)
                
                # Sum log probabilities over sequence length
                policy_seq_log_prob = policy_token_log_probs.sum()  # Scalar
                ref_seq_log_prob = ref_token_log_probs.sum()
                
                # Compute KL divergence
                kl_div = torch.exp(ref_seq_log_prob - policy_seq_log_prob) - (ref_seq_log_prob - policy_seq_log_prob) - 1
                
                # GRPO Loss: -advantage * log_prob + beta * KL
                advantage = reward_advantages[i, j]
                sample_loss = -advantage * policy_seq_log_prob + self.args.grpo_beta * kl_div
                
                total_loss += sample_loss
                
                # Collect metrics for this batch
                batch_policy_log_probs.append(policy_seq_log_prob.item())
                batch_kl_divs.append(kl_div.item())
                batch_advantages.append(advantage.item())
        
        # Average loss across all samples
        loss = total_loss / (batch_size * self.args.grpo_sample_size)
        
        # Log GRPO metrics for this batch
        self.grpo_metrics['policy_seq_log_prob'].extend(batch_policy_log_probs)
        self.grpo_metrics['kl_div'].extend(batch_kl_divs)
        self.grpo_metrics['format_reward'].extend(batch_format_rewards)
        self.grpo_metrics['accuracy_reward'].extend(batch_accuracy_rewards)
        self.grpo_metrics['total_reward'].extend(rewards.flatten().cpu().tolist())
        self.grpo_metrics['advantage'].extend(batch_advantages)
        
        return (loss, outputs) if return_outputs else loss

    def _extract_ground_truths(self, labels):
        """Extract ground truth texts from labels."""
        ground_truths = []
        for i in range(labels.shape[0]):
            label_seq = labels[i]
            valid_labels = label_seq[label_seq != -100]
            if len(valid_labels) > 0:
                gt_text = self.processing_class.decode(valid_labels, skip_special_tokens=True)
            else:
                gt_text = ""
            ground_truths.append(gt_text)
        return ground_truths

    def _replace_assistant_response(self, input_ids, labels, generated_ids):
        """Replace assistant response in original sequence while keeping same length."""
        
        # Find assistant response boundaries in original sequence
        assistant_start, assistant_end = self._find_assistant_boundaries(input_ids[0])
        
        # Create new input_ids with generated response
        new_input_ids = input_ids.clone()
        gen_length = generated_ids.shape[1]
        
        if assistant_end - assistant_start >= gen_length:
            # Replace part of the assistant response
            new_input_ids[0, assistant_start:assistant_start+gen_length] = generated_ids[0]
            # Pad the rest if needed
            if assistant_end - assistant_start > gen_length:
                new_input_ids[0, assistant_start+gen_length:assistant_end] = self.processing_class.pad_token_id
        else:
            # Truncate generated response to fit
            new_input_ids[0, assistant_start:assistant_end] = generated_ids[0, :assistant_end-assistant_start]
        
        # Create corresponding labels (only compute loss on generated part)
        new_labels = torch.full_like(new_input_ids, -100)
        actual_gen_length = min(gen_length, assistant_end - assistant_start)
        new_labels[0, assistant_start:assistant_start+actual_gen_length] = \
            new_input_ids[0, assistant_start:assistant_start+actual_gen_length]
        
        return new_input_ids, new_labels

    def _find_assistant_boundaries(self, input_ids):
        """Find start and end positions of assistant response."""
        assistant_token = "<|im_start|>assistant\n"
        assistant_ids = self.processing_class.encode(assistant_token, add_special_tokens=False)
        eos_id = self.processing_class.eos_token_id
        
        seq = input_ids.cpu().tolist()
        
        # Find assistant start
        assistant_start = None
        for j in range(len(seq) - len(assistant_ids)):
            if seq[j:j+len(assistant_ids)] == assistant_ids:
                assistant_start = j + len(assistant_ids)
                break
        
        # Find assistant end (next eos token)
        assistant_end = len(seq)
        if assistant_start is not None:
            for j in range(assistant_start, len(seq)):
                if seq[j] == eos_id:
                    assistant_end = j
                    break
        
        return assistant_start or len(seq), assistant_end

    def _find_generated_token_positions(self, labels):
        """Find start and end positions of tokens to compute loss on."""
        valid_positions = (labels != -100).nonzero(as_tuple=True)[0]
        
        if len(valid_positions) == 0:
            return 0, 0
        
        gen_start = valid_positions[0].item()
        gen_end = valid_positions[-1].item() + 1
        
        return gen_start, gen_end

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:
        """Custom evaluation loop with reward metrics."""
        eval_results = super().evaluation_loop(
            dataloader, description, prediction_loss_only, ignore_keys, metric_key_prefix
        )
        
        # Additional evaluation with reward metrics
        self.model.eval()
        total_format_reward = 0.0
        total_accuracy_reward = 0.0
        total_samples = 0
        
        for step, inputs in enumerate(dataloader):
            inputs = self._prepare_inputs(inputs)
            
            # Get ground truths
            labels = inputs.get("labels")
            batch_size = inputs["input_ids"].shape[0]
            ground_truths = []
            
            for i in range(batch_size):
                label_seq = labels[i]
                valid_labels = label_seq[label_seq != -100]
                if len(valid_labels) > 0:
                    gt_text = self.processing_class.decode(valid_labels, skip_special_tokens=True)
                else:
                    gt_text = ""
                ground_truths.append(gt_text)
            
            # Generate single response per example for evaluation
            with torch.no_grad():
                generated_texts, _ = self.generate_responses(inputs)
                
            # Compute rewards and save samples during evaluation
            for i in range(batch_size):
                gen_text = generated_texts[i * self.args.grpo_sample_size]  # Take first sample
                gt_text = ground_truths[i]
                
                format_reward = self.compute_format_reward(gen_text)
                accuracy_reward = self.compute_accuracy_reward(gen_text, gt_text)
                
                total_format_reward += format_reward
                total_accuracy_reward += accuracy_reward
                total_samples += 1
                
                # Save evaluation samples (save a few for inspection)
                if step < 5:  # Save first 5 batches during evaluation
                    prompt = self._extract_prompt_from_input(inputs["input_ids"][i])
                    self.save_output_sample(
                        prompt=prompt,
                        generated_text=gen_text,
                        ground_truth=gt_text,
                        format_reward=format_reward,
                        accuracy_reward=accuracy_reward,
                        step=f"eval_{self.state.global_step}_{step}_{i}"
                    )
        
        # Flush any remaining outputs at end of evaluation
        self._flush_outputs_to_file()
        
        # Add metrics to results
        eval_results.metrics[f"{metric_key_prefix}_format_reward"] = total_format_reward / total_samples
        eval_results.metrics[f"{metric_key_prefix}_accuracy_reward"] = total_accuracy_reward / total_samples
        eval_results.metrics[f"{metric_key_prefix}_total_reward"] = (
            self.args.format_reward_weight * (total_format_reward / total_samples) +
            self.args.accuracy_reward_weight * (total_accuracy_reward / total_samples)
        )
        
        # Log evaluation metrics directly to wandb using passed wandb_run instance
        eval_metrics = {
            f"{metric_key_prefix}/format_reward": total_format_reward / total_samples,
            f"{metric_key_prefix}/accuracy_reward": total_accuracy_reward / total_samples,
            f"{metric_key_prefix}/total_reward": (
                self.args.format_reward_weight * (total_format_reward / total_samples) +
                self.args.accuracy_reward_weight * (total_accuracy_reward / total_samples)
            )
        }
        
        try:
            if self.wandb_run is not None:
                self.wandb_run.log(eval_metrics, step=self.state.global_step)
        except Exception as e:
            print(f"Error logging evaluation metrics to wandb: {e}")
        
        return eval_results

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        """Override save_model to also flush any remaining outputs."""
        # Flush remaining outputs to file
        self._flush_outputs_to_file()
        
        # Call parent save_model
        super().save_model(output_dir, _internal_call)
        
        print(f"Model saved. Output samples saved to: {self.output_log_file}") 