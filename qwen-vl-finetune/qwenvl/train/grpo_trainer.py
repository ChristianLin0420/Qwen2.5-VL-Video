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
import contextlib

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
        data_args=None,
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
        dataset_name = getattr(data_args, 'dataset_use', '')
        self.requires_thinking = 'no_think' not in dataset_name.lower()
        
        # Initialize JSON output logging
        self.output_log_file = os.path.join(args.output_dir, f"grpo_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
        self.outputs_buffer = []
        self.max_buffer_size = 200  # Increased buffer size to handle more frequent saves
        
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
        Compute format reward using standard VLM GRPO approach with exact pattern matching.
        Returns 1.0 for correct format, 0.0 otherwise.
        """
        # Clean the generated text
        content = generated_text.strip()
        
        if self.requires_thinking:
            # Dataset requires thinking process: <think>...</think> <answer>...</answer>
            pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
            match = re.fullmatch(pattern, content, re.DOTALL)
            return 1.0 if match else 0.0
        else:
            # Dataset does not require thinking process: <answer>...</answer> only
            pattern = r"<answer>.*?</answer>"
            match = re.fullmatch(pattern, content, re.DOTALL)
            return 1.0 if match else 0.0
    
    def compute_accuracy_reward(self, generated_text: str, ground_truth: str) -> float:
        """
        Compute accuracy reward using standard VLM GRPO approach with exact matching only.
        Returns 1.0 for exact match, 0.0 otherwise (no partial rewards).
        """
        content = generated_text
        sol = ground_truth
        reward = 0.0
        
        # First try action-based verification (our domain-specific logic)
        try:
            # Extract answer from generated text
            content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
            student_answer = content_match.group(1).strip() if content_match else content.strip()
            
            # Extract answer from ground truth 
            sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
            ground_truth_answer = sol_match.group(1).strip() if sol_match else sol.strip()
            
            # Extract action numbers for exact matching only
            def extract_actions(text):
                actions = re.findall(r'\((\d+)\)', text)
                return sorted(actions)
            
            generated_actions = extract_actions(student_answer)
            gt_actions = extract_actions(ground_truth_answer)
            
            # Check for EXACT action match only (no partial rewards)
            if generated_actions == gt_actions and len(generated_actions) > 0:
                reward = 1.0
            # Remove partial matching - only exact match gets reward
            
        except Exception:
            pass  # Continue to next verification method if this fails
        
        # If action-based verification failed, try normalized string matching (standard VLM approach)
        if reward == 0.0:
            try:
                # Extract answer from solution if it has think/answer tags
                sol_match = re.search(r'<answer>(.*?)</answer>', sol, re.DOTALL)
                ground_truth_answer = sol_match.group(1).strip() if sol_match else sol.strip()
                
                # Extract answer from content if it has think/answer tags  
                content_match = re.search(r'<answer>(.*?)</answer>', content, re.DOTALL)
                student_answer = content_match.group(1).strip() if content_match else content.strip()
                
                # Normalize both answers (standard VLM GRPO approach)
                ground_truth_norm = ground_truth_answer.replace(' ', '').replace('_', '').lower()
                student_answer_norm = student_answer.replace(' ', '').replace('_', '').lower()
                
                # Exact match only (no partial rewards)
                if ground_truth_norm == student_answer_norm and len(ground_truth_norm) > 0:
                    reward = 1.0
                # Remove bidirectional substring matching - only exact match gets reward
                
            except Exception:
                pass  # Keep reward as 0.0 if both methods fail
        
        # Optional debug logging (following standard implementation)
        if os.getenv("DEBUG_MODE") == "true":
            log_path = os.getenv("LOG_PATH", "grpo_debug.log")
            current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
            with open(log_path, "a") as f:
                f.write(f"------------- {current_time} Accuracy reward: {reward} -------------\n")
                f.write(f"content: {content}\n")
                f.write(f"sol: {sol}\n")
        
        return reward
    
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
    
    def _get_per_token_logps(self, model, input_ids, attention_mask, pixel_values=None, image_grid_thw=None, pixel_values_videos=None, video_grid_thw=None):
        """
        Compute per-token log probabilities for the given input sequence.
        Returns log probabilities shifted for causal language modeling.
        """
        model_inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        
        # Add visual inputs if provided
        if pixel_values is not None:
            model_inputs["pixel_values"] = pixel_values
        if image_grid_thw is not None:
            model_inputs["image_grid_thw"] = image_grid_thw
        if pixel_values_videos is not None:
            model_inputs["pixel_values_videos"] = pixel_values_videos
        if video_grid_thw is not None:
            model_inputs["video_grid_thw"] = video_grid_thw
            
        with torch.no_grad() if model != self.model else contextlib.nullcontext():
            outputs = model(**model_inputs)
        
        logits = outputs.logits  # (batch_size, seq_len, vocab_size)
        
        # Shift logits and input_ids for causal LM
        shift_logits = logits[:, :-1, :].contiguous()  # (batch_size, seq_len-1, vocab_size)
        shift_labels = input_ids[:, 1:].contiguous()   # (batch_size, seq_len-1)
        
        # Compute log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)  # (batch_size, seq_len-1, vocab_size)
        per_token_logps = log_probs.gather(dim=-1, index=shift_labels.unsqueeze(-1)).squeeze(-1)
        
        return per_token_logps  # (batch_size, seq_len-1)

    def generate_responses_batch(self, inputs: Dict[str, torch.Tensor]) -> Tuple[List[str], torch.Tensor, torch.Tensor]:
        """
        Generate multiple responses for each input in batch, following standard GRPO approach.
        Returns: (generated_texts, completion_ids, prompt_lengths)
        """
        self.model.eval()
        
        batch_size = inputs["input_ids"].shape[0]
        input_ids = inputs["input_ids"]
        
        # Find prompt lengths (before assistant response)
        assistant_token = "<|im_start|>assistant\n"
        assistant_ids = self.processing_class.encode(assistant_token, add_special_tokens=False)
        
        prompt_lengths = []
        for i in range(batch_size):
            seq = input_ids[i].cpu().tolist()
            prompt_length = len(seq)
            
            # Find assistant marker
            for j in range(len(seq) - len(assistant_ids), -1, -1):
                if j >= 0 and seq[j:j+len(assistant_ids)] == assistant_ids:
                    prompt_length = j + len(assistant_ids)
                    break
            prompt_lengths.append(prompt_length)
        
        # Generate responses one by one to avoid video tensor issues
        all_generated_texts = []
        all_completion_ids = []
        
        with torch.no_grad():
            for i in range(batch_size):
                # Prepare inputs for single sample
                single_inputs = {}
                prompt_len = prompt_lengths[i]
                
                for key, value in inputs.items():
                    if value is not None:
                        if key in ["input_ids", "attention_mask"]:
                            # Handle different tensor dimensions
                            if value.dim() == 1:
                                # 1D tensor (flattened format) - skip for generation
                                continue
                            elif value.dim() == 2:
                                # 2D tensor (standard format) - truncate to prompt
                                single_inputs[key] = value[i:i+1, :prompt_len]
                            else:
                                continue
                        elif key in ["pixel_values", "image_grid_thw"]:
                            # Use original visual inputs for this sample
                            if value.dim() > 1 and value.shape[0] == batch_size:
                                single_inputs[key] = value[i:i+1]
                            else:
                                single_inputs[key] = value
                        elif key in ["pixel_values_videos", "video_grid_thw"]:
                            # Keep video inputs as-is (don't repeat/modify them)
                            single_inputs[key] = value
                
                # Create proper attention masks if not present
                if "attention_mask" not in single_inputs and "input_ids" in single_inputs:
                    single_inputs["attention_mask"] = torch.ones_like(single_inputs["input_ids"], dtype=torch.bool)
                
                # Generate multiple samples for this input
                for _ in range(self.args.grpo_sample_size):
                    sample_outputs = self.model.generate(
                        **single_inputs,
                        max_new_tokens=self.args.generation_max_length,
                        temperature=self.args.generation_temperature,
                        top_p=self.args.generation_top_p,
                        num_beams=self.args.generation_num_beams,
                        do_sample=True,
                        pad_token_id=self.processing_class.pad_token_id,
                        eos_token_id=self.processing_class.eos_token_id,
                        return_dict_in_generate=False,
                    )
                    
                    # Extract only the generated part
                    completion_ids = sample_outputs[0, prompt_len:]
                    completion_text = self.processing_class.decode(completion_ids, skip_special_tokens=True)
                    
                    all_generated_texts.append(completion_text)
                    all_completion_ids.append(completion_ids)
        
        # Pad completion ids to same length
        max_completion_len = max(ids.shape[0] for ids in all_completion_ids)
        padded_completions = []
        
        for ids in all_completion_ids:
            if ids.shape[0] < max_completion_len:
                padding = torch.full(
                    (max_completion_len - ids.shape[0],),
                    self.processing_class.pad_token_id,
                    device=ids.device
                )
                ids = torch.cat([ids, padding])
            padded_completions.append(ids)
        
        completion_ids = torch.stack(padded_completions, dim=0)  # (B*G, C)
        prompt_lengths_tensor = torch.tensor(prompt_lengths, device=self.model.device)
        
        return all_generated_texts, completion_ids, prompt_lengths_tensor

    def generate_responses_eval(self, inputs: Dict[str, torch.Tensor]) -> List[str]:
        """
        Generate single response per input for evaluation (more efficient).
        Returns: generated_texts
        """
        self.model.eval()
        
        batch_size = inputs["input_ids"].shape[0]
        input_ids = inputs["input_ids"]
        
        # Find prompt lengths (before assistant response)
        assistant_token = "<|im_start|>assistant\n"
        assistant_ids = self.processing_class.encode(assistant_token, add_special_tokens=False)
        
        # Generate responses one by one to avoid video tensor issues
        all_generated_texts = []
        
        with torch.no_grad():
            for i in range(batch_size):
                # Prepare inputs for single sample
                single_inputs = {}
                
                # Find prompt length for this sample
                seq = input_ids[i].cpu().tolist()
                prompt_length = len(seq)
                for j in range(len(seq) - len(assistant_ids), -1, -1):
                    if j >= 0 and seq[j:j+len(assistant_ids)] == assistant_ids:
                        prompt_length = j + len(assistant_ids)
                        break
                
                for key, value in inputs.items():
                    if value is not None and key in ["input_ids", "attention_mask"]:
                        # Handle different tensor dimensions
                        if value.dim() == 1:
                            # 1D tensor (flattened format) - skip for generation
                            continue
                        elif value.dim() == 2:
                            # 2D tensor (standard format) - truncate to prompt
                            single_inputs[key] = value[i:i+1, :prompt_length]
                        else:
                            # Skip other dimensions
                            continue
                    elif value is not None and key in ["pixel_values", "image_grid_thw"]:
                        # Use original visual inputs for this sample
                        if value.dim() > 1 and value.shape[0] == batch_size:
                            single_inputs[key] = value[i:i+1]
                        else:
                            single_inputs[key] = value
                    elif value is not None and key in ["pixel_values_videos", "video_grid_thw"]:
                        # Keep video inputs as-is (don't repeat/modify them)
                        single_inputs[key] = value
                
                # Create proper attention masks if not present
                if "attention_mask" not in single_inputs and "input_ids" in single_inputs:
                    single_inputs["attention_mask"] = torch.ones_like(single_inputs["input_ids"], dtype=torch.bool)
                
                # Generate single response for this input
                outputs = self.model.generate(
                    **single_inputs,
                    max_new_tokens=self.args.generation_max_length,
                    temperature=0.1,  # Lower temperature for more deterministic eval
                    top_p=0.9,
                    num_beams=1,
                    do_sample=True,
                    pad_token_id=self.processing_class.pad_token_id,
                    eos_token_id=self.processing_class.eos_token_id,
                    return_dict_in_generate=False,
                )
                
                # Extract generated completion
                completion_ids = outputs[0, prompt_length:]
                completion_text = self.processing_class.decode(completion_ids, skip_special_tokens=True)
                all_generated_texts.append(completion_text)
        
        return all_generated_texts

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute GRPO loss using efficient vectorized operations following standard approach."""
        
        # Get ground truth and generate responses
        labels = inputs.get("labels")
        batch_size = inputs["input_ids"].shape[0]
        device = self.model.device
        
        ground_truths = self._extract_ground_truths(labels)
        
        # Apply GRPO flash attention override for generation
        from qwenvl.train.grpo_flash_attn_override import replace_qwen2_vl_attention_class_grpo
        replace_qwen2_vl_attention_class_grpo()
        
        # Generate responses using efficient batch approach
        generated_texts, completion_ids, prompt_lengths = self.generate_responses_batch(inputs)
        
        # Compute rewards
        rewards = self.compute_rewards(generated_texts, ground_truths * self.args.grpo_sample_size)
        
        # Mask everything after the first EOS token for each completion
        is_eos = completion_ids == self.processing_class.eos_token_id
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()
        
        # Compute grouped-wise rewards and advantages
        rewards_grouped = rewards.view(batch_size, self.args.grpo_sample_size)  # (B, G)
        mean_grouped_rewards = rewards_grouped.mean(dim=1)  # (B,)
        std_grouped_rewards = rewards_grouped.std(dim=1)  # (B,)
        
        # Normalize rewards to compute advantages
        mean_expanded = mean_grouped_rewards.repeat_interleave(self.args.grpo_sample_size, dim=0)  # (B*G,)
        std_expanded = std_grouped_rewards.repeat_interleave(self.args.grpo_sample_size, dim=0)  # (B*G,)
        advantages = (rewards - mean_expanded) / (std_expanded + 1e-4)  # (B*G,)
        
        # Compute per-token log probabilities by replacing assistant responses in original sequences
        total_loss = 0.0
        
        # Collect metrics for logging
        batch_policy_log_probs = []
        batch_kl_divs = []
        batch_advantages = []
        
        for i in range(batch_size):
            for j in range(self.args.grpo_sample_size):
                sample_idx = i * self.args.grpo_sample_size + j
                
                # Create modified sequence with generated completion
                modified_input_ids, modified_labels = self._create_modified_sequence(
                    inputs["input_ids"][i], labels[i], completion_ids[sample_idx], prompt_lengths[i]
                )
                
                # Create inputs for model forward pass (keep original format)
                model_inputs = {
                    "input_ids": modified_input_ids.unsqueeze(0),
                    "attention_mask": torch.ones_like(modified_input_ids, dtype=torch.bool).unsqueeze(0),
                }
                
                # Add visual inputs (use original ones - don't modify)
                if "pixel_values" in inputs and inputs["pixel_values"] is not None:
                    model_inputs["pixel_values"] = inputs["pixel_values"][i:i+1]
                if "image_grid_thw" in inputs and inputs["image_grid_thw"] is not None:
                    model_inputs["image_grid_thw"] = inputs["image_grid_thw"][i:i+1]
                if "pixel_values_videos" in inputs and inputs["pixel_values_videos"] is not None:
                    model_inputs["pixel_values_videos"] = inputs["pixel_values_videos"]
                if "video_grid_thw" in inputs and inputs["video_grid_thw"] is not None:
                    model_inputs["video_grid_thw"] = inputs["video_grid_thw"]
                
                # Forward pass through policy model
                outputs = model(**model_inputs)
                policy_logits = outputs.logits[0]  # Remove batch dimension
                
                # Forward pass through reference model
                with torch.inference_mode():
                    ref_outputs = self.ref_model(**model_inputs)
                    ref_logits = ref_outputs.logits[0]  # Remove batch dimension
                
                # Find completion token positions
                completion_start = prompt_lengths[i].item()
                completion_end = completion_start + completion_mask[sample_idx].sum().item()
                
                if completion_start >= completion_end:
                    continue  # Skip if no completion tokens
                
                # Extract logits for completion tokens (shift for causal LM)
                policy_completion_logits = policy_logits[completion_start-1:completion_end-1, :]
                ref_completion_logits = ref_logits[completion_start-1:completion_end-1, :]
                
                # Get target tokens
                target_tokens = modified_input_ids[completion_start:completion_end]
                
                # Compute log probabilities
                policy_log_probs = F.log_softmax(policy_completion_logits, dim=-1)
                ref_log_probs = F.log_softmax(ref_completion_logits, dim=-1)
                
                # Get log probabilities for actual tokens
                policy_token_log_probs = policy_log_probs.gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)
                ref_token_log_probs = ref_log_probs.gather(1, target_tokens.unsqueeze(-1)).squeeze(-1)
                
                # Apply completion mask to valid tokens only
                valid_mask = completion_mask[sample_idx][:len(policy_token_log_probs)].float()
                
                # Compute sequence log probabilities (sum over valid tokens)
                policy_seq_log_prob = (policy_token_log_probs * valid_mask).sum()
                ref_seq_log_prob = (ref_token_log_probs * valid_mask).sum()
                
                # Compute per-token KL divergence 
                per_token_kl = torch.exp(ref_token_log_probs - policy_token_log_probs) - \
                              (ref_token_log_probs - policy_token_log_probs) - 1
                
                # GRPO loss computation (following standard implementation)
                advantage = advantages[sample_idx]
                
                # Per-token loss with advantage weighting
                per_token_loss = torch.exp(policy_token_log_probs - policy_token_log_probs.detach()) * advantage
                per_token_loss = -(per_token_loss - self.args.grpo_beta * per_token_kl)
                
                # Average loss over valid tokens
                sample_loss = (per_token_loss * valid_mask).sum() / valid_mask.sum()
                total_loss += sample_loss
                
                # Collect metrics
                batch_policy_log_probs.append(policy_seq_log_prob.item())
                batch_kl_divs.append((per_token_kl * valid_mask).sum().item() / valid_mask.sum().item())
                batch_advantages.append(advantage.item())
        
        # Average loss across all samples
        loss = total_loss / (batch_size * self.args.grpo_sample_size)
        
        # Add reward metrics
        batch_format_rewards = []
        batch_accuracy_rewards = []
        for gen_text, gt_text in zip(generated_texts, ground_truths * self.args.grpo_sample_size):
            format_reward = self.compute_format_reward(gen_text)
            accuracy_reward = self.compute_accuracy_reward(gen_text, gt_text)
            batch_format_rewards.append(format_reward)
            batch_accuracy_rewards.append(accuracy_reward)
        
        # Log GRPO metrics for this batch
        self.grpo_metrics['policy_seq_log_prob'].extend(batch_policy_log_probs)
        self.grpo_metrics['kl_div'].extend(batch_kl_divs)
        self.grpo_metrics['format_reward'].extend(batch_format_rewards)
        self.grpo_metrics['accuracy_reward'].extend(batch_accuracy_rewards)
        self.grpo_metrics['total_reward'].extend(rewards.cpu().tolist())
        self.grpo_metrics['advantage'].extend(batch_advantages)
        
        # Save all sample outputs for complete tracking
        for i in range(batch_size):
            sample_idx = i * self.args.grpo_sample_size
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
        
        model.train()
        
        return (loss, None) if return_outputs else loss

    def _create_modified_sequence(self, original_input_ids, original_labels, completion_ids, prompt_length):
        """
        Create a modified sequence by replacing the assistant response with generated completion.
        Maintains original sequence structure for video compatibility.
        """
        prompt_len = prompt_length.item()
        
        # Find original assistant response boundaries
        assistant_start, assistant_end = self._find_assistant_boundaries(original_input_ids)
        
        # Create new sequence: prompt + generated completion + padding/truncation as needed
        if assistant_start == len(original_input_ids):
            # No assistant response found, append completion
            modified_input_ids = torch.cat([
                original_input_ids[:prompt_len],
                completion_ids
            ])
        else:
            # Replace existing assistant response
            max_completion_len = assistant_end - assistant_start
            if len(completion_ids) <= max_completion_len:
                # Completion fits in original space
                modified_input_ids = original_input_ids.clone()
                modified_input_ids[assistant_start:assistant_start + len(completion_ids)] = completion_ids
                # Pad remaining space if needed
                if len(completion_ids) < max_completion_len:
                    modified_input_ids[assistant_start + len(completion_ids):assistant_end] = self.processing_class.pad_token_id
            else:
                # Truncate completion to fit
                modified_input_ids = original_input_ids.clone()
                modified_input_ids[assistant_start:assistant_end] = completion_ids[:max_completion_len]
        
        # Create labels for loss computation (only on generated tokens)
        modified_labels = torch.full_like(modified_input_ids, -100)
        actual_completion_len = min(len(completion_ids), assistant_end - assistant_start if assistant_start < len(original_input_ids) else len(completion_ids))
        modified_labels[assistant_start:assistant_start + actual_completion_len] = \
            modified_input_ids[assistant_start:assistant_start + actual_completion_len]
        
        return modified_input_ids, modified_labels

    def _find_assistant_boundaries(self, input_ids):
        """Find start and end positions of assistant response."""
        assistant_token = "<|im_start|>assistant\n"
        assistant_ids = self.processing_class.encode(assistant_token, add_special_tokens=False)
        eos_id = self.processing_class.eos_token_id
        
        seq = input_ids.cpu().tolist()
        
        # Find assistant start
        assistant_start = None
        for j in range(len(seq) - len(assistant_ids) + 1):
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

    def _extract_ground_truths(self, labels):
        """Extract ground truth texts from labels and clean up concatenated answers."""
        ground_truths = []
        for i in range(labels.shape[0]):
            label_seq = labels[i]
            valid_labels = label_seq[label_seq != -100]
            if len(valid_labels) > 0:
                gt_text = self.processing_class.decode(valid_labels, skip_special_tokens=True)
                
                # Clean up concatenated answers - extract the first complete answer
                gt_text = self._clean_ground_truth(gt_text)
            else:
                gt_text = ""
            ground_truths.append(gt_text)
        return ground_truths

    def _clean_ground_truth(self, gt_text: str) -> str:
        """
        Clean ground truth by extracting the first complete answer from concatenated answers.
        """
        # Strip whitespace
        gt_text = gt_text.strip()
        
        # Find all answer tags
        answer_pattern = r'<answer>(.*?)</answer>'
        matches = re.findall(answer_pattern, gt_text, re.DOTALL)
        
        if matches:
            # Return the first complete answer
            first_answer = matches[0].strip()
            return f"<answer>{first_answer}</answer>"
        else:
            # If no answer tags found, return as is
            return gt_text

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
                generated_texts = self.generate_responses_eval(inputs)
                
            # Compute rewards and save samples during evaluation
            for i in range(batch_size):
                gen_text = generated_texts[i]  # Single sample per input in eval
                gt_text = ground_truths[i]
                
                format_reward = self.compute_format_reward(gen_text)
                accuracy_reward = self.compute_accuracy_reward(gen_text, gt_text)
                
                total_format_reward += format_reward
                total_accuracy_reward += accuracy_reward
                total_samples += 1
                
                # Save all evaluation samples for complete tracking
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