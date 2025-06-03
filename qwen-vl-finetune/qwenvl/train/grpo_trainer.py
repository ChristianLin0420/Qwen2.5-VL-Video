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

from qwenvl.train.trainer import replace_qwen2_vl_attention_class
from qwenvl.train.grpo_flash_attn_override import replace_qwen2_vl_attention_class_grpo


@dataclass
class GRPOTrainingArguments(TrainingArguments):
    """Training arguments for GRPO training."""
    # GRPO specific arguments
    grpo_alpha: float = field(default=0.5, metadata={"help": "GRPO alpha parameter for reward weighting"})
    grpo_beta: float = field(default=0.1, metadata={"help": "GRPO beta parameter for KL penalty"})
    format_reward_weight: float = field(default=0.3, metadata={"help": "Weight for format reward"})
    accuracy_reward_weight: float = field(default=0.7, metadata={"help": "Weight for accuracy reward"})
    generation_max_length: int = field(default=512, metadata={"help": "Maximum length for generation"})
    generation_temperature: float = field(default=0.7, metadata={"help": "Temperature for generation"})
    generation_top_p: float = field(default=0.9, metadata={"help": "Top-p for generation"})
    generation_num_beams: int = field(default=1, metadata={"help": "Number of beams for generation"})
    grpo_sample_size: int = field(default=4, metadata={"help": "Number of samples to generate per example"})
    
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
        
        # Initialize reference model for KL computation
        self.ref_model = copy.deepcopy(self.model)
        self.ref_model.eval()
        for param in self.ref_model.parameters():
            param.requires_grad = False
        
        # DON'T replace attention here - only during training
        
    def compute_format_reward(self, generated_text: str) -> float:
        """
        Compute format reward based on whether the output follows the required format.
        Expected format: <think> ... </think> <answer>actions</answer>
        """
        # Check for think tags
        think_pattern = r'<think>.*?</think>'
        has_think = bool(re.search(think_pattern, generated_text, re.DOTALL))
        
        # Check for answer tags
        answer_pattern = r'<answer>.*?</answer>'
        has_answer = bool(re.search(answer_pattern, generated_text, re.DOTALL))
        
        # Check if answer appears after think
        if has_think and has_answer:
            think_match = re.search(think_pattern, generated_text, re.DOTALL)
            answer_match = re.search(answer_pattern, generated_text, re.DOTALL)
            if think_match and answer_match:
                correct_order = think_match.end() <= answer_match.start()
            else:
                correct_order = False
        else:
            correct_order = False
        
        # Calculate format reward
        format_score = 0.0
        if has_think:
            format_score += 0.3
        if has_answer:
            format_score += 0.3
        if correct_order:
            format_score += 0.4
            
        return format_score
    
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

    def compute_kl_penalty(
        self, 
        input_ids: torch.Tensor,
        generated_ids: torch.Tensor,
        batch: Dict[str, torch.Tensor]
    ) -> torch.Tensor:
        """Compute KL divergence between policy and reference model."""
        self.model.eval()
        self.ref_model.eval()
        
        with torch.no_grad():
            # Get reference model logits
            ref_outputs = self.ref_model(
                input_ids=input_ids,
                attention_mask=batch.get("attention_mask"),
                position_ids=batch.get("position_ids"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                pixel_values_videos=batch.get("pixel_values_videos"),
                video_grid_thw=batch.get("video_grid_thw"),
            )
            ref_logits = ref_outputs.logits
            
            # Get policy model logits
            policy_outputs = self.model(
                input_ids=input_ids,
                attention_mask=batch.get("attention_mask"),
                position_ids=batch.get("position_ids"),
                pixel_values=batch.get("pixel_values"),
                image_grid_thw=batch.get("image_grid_thw"),
                pixel_values_videos=batch.get("pixel_values_videos"),
                video_grid_thw=batch.get("video_grid_thw"),
            )
            policy_logits = policy_outputs.logits
        
        # Compute KL divergence for generated tokens only
        gen_start = input_ids.shape[1] - generated_ids.shape[1]
        ref_logits = ref_logits[:, gen_start:, :]
        policy_logits = policy_logits[:, gen_start:, :]
        
        # Convert to probabilities
        ref_probs = F.softmax(ref_logits, dim=-1)
        policy_probs = F.softmax(policy_logits, dim=-1)
        
        # Compute KL divergence
        kl = torch.sum(policy_probs * (torch.log(policy_probs + 1e-8) - torch.log(ref_probs + 1e-8)), dim=-1)
        kl = kl.mean(dim=1)  # Average over sequence length
        
        return kl
    
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
        
        total_loss = 0.0
        model.train()
        
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
                kl_div = policy_seq_log_prob - ref_seq_log_prob
                
                # GRPO Loss: -advantage * log_prob + beta * KL
                advantage = reward_advantages[i, j]
                sample_loss = -advantage * policy_seq_log_prob + self.args.grpo_beta * kl_div
                
                total_loss += sample_loss
        
        # Average loss across all samples
        loss = total_loss / (batch_size * self.args.grpo_sample_size)
        
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
                
            # Compute rewards
            for i in range(batch_size):
                gen_text = generated_texts[i * self.args.grpo_sample_size]  # Take first sample
                gt_text = ground_truths[i]
                
                format_reward = self.compute_format_reward(gen_text)
                accuracy_reward = self.compute_accuracy_reward(gen_text, gt_text)
                
                total_format_reward += format_reward
                total_accuracy_reward += accuracy_reward
                total_samples += 1
        
        # Add metrics to results
        eval_results.metrics[f"{metric_key_prefix}_format_reward"] = total_format_reward / total_samples
        eval_results.metrics[f"{metric_key_prefix}_accuracy_reward"] = total_accuracy_reward / total_samples
        eval_results.metrics[f"{metric_key_prefix}_total_reward"] = (
            self.args.format_reward_weight * (total_format_reward / total_samples) +
            self.args.accuracy_reward_weight * (total_accuracy_reward / total_samples)
        )
        
        return eval_results 