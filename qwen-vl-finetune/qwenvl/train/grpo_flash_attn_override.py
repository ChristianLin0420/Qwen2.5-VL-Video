# import torch
# from typing import Optional
# from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_func


# def _flash_attention_forward_grpo(
#     query_states: torch.Tensor,
#     key_states: torch.Tensor,
#     value_states: torch.Tensor,
#     attention_mask: torch.Tensor,
#     query_length: int,
#     is_causal: bool,
#     dropout: float = 0.0,
#     position_ids: Optional[torch.Tensor] = None,
#     softmax_scale: Optional[float] = None,
#     sliding_window: Optional[int] = None,
#     use_top_left_mask: bool = False,
#     softcap: Optional[float] = None,
#     deterministic: bool = None,
#     cu_seq_lens_q: Optional[torch.LongTensor] = None,
#     cu_seq_lens_k: Optional[torch.LongTensor] = None,
#     max_length_q: Optional[int] = None,
#     max_length_k: Optional[int] = None,
#     target_dtype: Optional[torch.dtype] = None,
#     **kwargs,
# ):
#     """
#     Flash attention forward that handles both training (with cu_seqlens) and generation.
#     """
#     # Check if we're in generation mode (standard attention mask) or training mode (cu_seqlens)
#     print(f"attention_mask: {attention_mask}")
#     print(f"attention_mask.dtype: {attention_mask.dtype}")
#     if attention_mask is not None and attention_mask.dtype == torch.bool:
#         # Generation mode - use standard flash attention
#         batch_size, num_heads, seq_length, head_dim = query_states.shape
        
#         # Reshape for flash attention
#         query_states = query_states.transpose(1, 2).reshape(batch_size * seq_length, num_heads, head_dim)
#         key_states = key_states.transpose(1, 2).reshape(batch_size * seq_length, num_heads, head_dim)
#         value_states = value_states.transpose(1, 2).reshape(batch_size * seq_length, num_heads, head_dim)
        
#         # Convert boolean mask to flash attention format if needed
#         if attention_mask is not None:
#             # Flash attention expects attention mask in a specific format
#             # For generation, we typically use causal mask
#             attention_mask = None  # Let flash attention handle causal masking

#         print(f"query_states: {query_states.shape}")
#         print(f"key_states: {key_states.shape}")
#         print(f"value_states: {value_states.shape}")
#         print(f"is_causal: {is_causal}")
#         print(f"dropout: {dropout}")
#         print(f"softmax_scale: {softmax_scale}")
#         print(f"sliding_window: {sliding_window}")
#         print(f"use_top_left_mask: {use_top_left_mask}")
#         print(f"softcap: {softcap}")
#         print(f"deterministic: {deterministic}")
#         print(f"cu_seq_lens_q: {cu_seq_lens_q}")
#         print(f"cu_seq_lens_k: {cu_seq_lens_k}")
#         print(f"max_length_q: {max_length_q}")
#         print(f"max_length_k: {max_length_k}")
#         print(f"target_dtype: {target_dtype}")
#         print(f"kwargs: {kwargs}")

#         # Use standard flash attention for generation
#         attn_output = flash_attn_func(
#             query_states,
#             key_states,
#             value_states,
#             dropout_p=dropout,
#             softmax_scale=softmax_scale,
#             causal=is_causal,
#             window_size=(sliding_window, sliding_window) if sliding_window is not None else (-1, -1),
#             alibi_slopes=None,
#             deterministic=deterministic,
#             return_attn_probs=False,
#         )
        
#         # Reshape back
#         attn_output = attn_output.reshape(batch_size, seq_length, num_heads, head_dim)
#         attn_output = attn_output.transpose(1, 2)
        
#         return attn_output
    
#     else:
#         # Training mode with cu_seqlens
#         assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
#         query_states = query_states.squeeze(0)
#         key_states = key_states.squeeze(0)
#         value_states = value_states.squeeze(0)
#         cu_seqlens = attention_mask

#         with torch.no_grad():
#             max_seqlen = max(
#                 [
#                     cu_seqlens[idx + 1] - cu_seqlens[idx]
#                     for idx in range(cu_seqlens.size(0) - 1)
#                 ]
#             ).item()
#             print(f"max_seqlen: {max_seqlen}")

#         if not use_top_left_mask:
#             causal = is_causal
#         else:
#             # TODO: Remove the `query_length != 1` check once Flash Attention for RoCm is bumped to 2.1.
#             causal = is_causal and query_length != 1

#         # Assuming 4D tensors, key_states.shape[1] is the key/value sequence length (source length).
#         flash_kwargs = {}

#         if softcap is not None:
#             flash_kwargs["softcap"] = softcap

#         attn_output = flash_attn_varlen_func(
#             query_states,
#             key_states,
#             value_states,
#             cu_seqlens_q=cu_seqlens,
#             cu_seqlens_k=cu_seqlens,
#             max_seqlen_q=max_seqlen,
#             max_seqlen_k=max_seqlen,
#             dropout_p=dropout,
#             softmax_scale=softmax_scale,
#             causal=causal,
#             **flash_kwargs,
#         )

#         attn_output = attn_output.unsqueeze(0)
        
#         return attn_output


# def replace_qwen2_vl_attention_class_grpo():
#     """Replace flash attention with GRPO-compatible version."""
#     import transformers
#     import transformers.modeling_flash_attention_utils

#     transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
#         _flash_attention_forward_grpo
#     )
#     transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
#         _flash_attention_forward_grpo
#     ) 

# import torch
# from typing import Optional
# from flash_attn.flash_attn_interface import flash_attn_varlen_func, flash_attn_func


# def _flash_attention_forward_grpo(
#     query_states: torch.Tensor,
#     key_states: torch.Tensor,
#     value_states: torch.Tensor,
#     attention_mask: torch.Tensor,
#     query_length: int,
#     is_causal: bool,
#     dropout: float = 0.0,
#     position_ids: Optional[torch.Tensor] = None,
#     softmax_scale: Optional[float] = None,
#     sliding_window: Optional[int] = None,
#     use_top_left_mask: bool = False,
#     softcap: Optional[float] = None,
#     deterministic: bool = None,
#     cu_seq_lens_q: Optional[torch.LongTensor] = None,
#     cu_seq_lens_k: Optional[torch.LongTensor] = None,
#     max_length_q: Optional[int] = None,
#     max_length_k: Optional[int] = None,
#     target_dtype: Optional[torch.dtype] = None,
#     **kwargs,
# ):
#     """
#     Flash attention forward that handles both training (with cu_seqlens) and generation.
#     """

#     # Check if we're in generation mode (standard attention mask) or training mode (cu_seqlens)
#     if attention_mask is not None and attention_mask.dtype == torch.bool:
#         # Generation mode - handle both 3D and 4D tensor formats
#         if query_states.dim() == 3:
#             # 3D format: [num_heads, seq_length, head_dim]
#             num_heads, seq_length, head_dim = query_states.shape
#             batch_size = 1
            
#             # Reshape to [batch_size * seq_length, num_heads, head_dim] for flash_attn_func
#             query_states = query_states.transpose(0, 1).contiguous()  # [seq_length, num_heads, head_dim]
#             key_states = key_states.transpose(0, 1).contiguous()
#             value_states = value_states.transpose(0, 1).contiguous()
            
#         elif query_states.dim() == 4:
#             # 4D format: [batch_size, num_heads, seq_length, head_dim]
#             batch_size, num_heads, seq_length, head_dim = query_states.shape
            
#             # Reshape to [batch_size * seq_length, num_heads, head_dim] for flash_attn_func
#             query_states = query_states.transpose(1, 2).reshape(batch_size * seq_length, num_heads, head_dim)
#             key_states = key_states.transpose(1, 2).reshape(batch_size * seq_length, num_heads, head_dim)
#             value_states = value_states.transpose(1, 2).reshape(batch_size * seq_length, num_heads, head_dim)
#         else:
#             raise ValueError(f"Unexpected query_states shape: {query_states.shape}")
        
#         # Use standard flash attention for generation
#         attn_output = flash_attn_func(
#             query_states,
#             key_states,
#             value_states,
#             dropout_p=dropout,
#             softmax_scale=softmax_scale,
#             causal=is_causal,
#             window_size=(sliding_window, sliding_window) if sliding_window is not None else (-1, -1),
#             alibi_slopes=None,
#             deterministic=deterministic,
#             return_attn_probs=False,
#         )
        
#         # Reshape back to original format
#         if query_states.dim() == 2:  # After reshaping from 3D
#             # attn_output is [seq_length, num_heads, head_dim], reshape back to [num_heads, seq_length, head_dim]
#             attn_output = attn_output.transpose(0, 1).contiguous()
#         else:
#             # Reshape back to [batch_size, num_heads, seq_length, head_dim]
#             attn_output = attn_output.reshape(batch_size, seq_length, num_heads, head_dim)
#             attn_output = attn_output.transpose(1, 2)
        
#         return attn_output
    
#     else:
#         # Training mode with cu_seqlens - use original logic
#         assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
#         query_states = query_states.squeeze(0)
#         key_states = key_states.squeeze(0)
#         value_states = value_states.squeeze(0)
#         cu_seqlens = attention_mask

#         with torch.no_grad():
#             max_seqlen = max(
#                 [
#                     cu_seqlens[idx + 1] - cu_seqlens[idx]
#                     for idx in range(cu_seqlens.size(0) - 1)
#                 ]
#             ).item()

#         if not use_top_left_mask:
#             causal = is_causal
#         else:
#             causal = is_causal and query_length != 1

#         flash_kwargs = {}
#         if softcap is not None:
#             flash_kwargs["softcap"] = softcap

#         attn_output = flash_attn_varlen_func(
#             query_states,
#             key_states,
#             value_states,
#             cu_seqlens_q=cu_seqlens,
#             cu_seqlens_k=cu_seqlens,
#             max_seqlen_q=max_seqlen,
#             max_seqlen_k=max_seqlen,
#             dropout_p=dropout,
#             softmax_scale=softmax_scale,
#             causal=causal,
#             **flash_kwargs,
#         )

#         attn_output = attn_output.unsqueeze(0)
        
#         return attn_output


# def replace_qwen2_vl_attention_class_grpo():
#     """Replace flash attention with GRPO-compatible version."""
#     import transformers
#     import transformers.modeling_flash_attention_utils

#     transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
#         _flash_attention_forward_grpo
#     )
#     transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
#         _flash_attention_forward_grpo
#     )

import torch
from typing import Optional
from flash_attn.flash_attn_interface import flash_attn_varlen_func


def _flash_attention_forward_grpo(
    query_states: torch.Tensor,
    key_states: torch.Tensor,
    value_states: torch.Tensor,
    attention_mask: torch.Tensor,
    query_length: int,
    is_causal: bool,
    dropout: float = 0.0,
    position_ids: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    sliding_window: Optional[int] = None,
    use_top_left_mask: bool = False,
    softcap: Optional[float] = None,
    deterministic: bool = None,
    cu_seq_lens_q: Optional[torch.LongTensor] = None,
    cu_seq_lens_k: Optional[torch.LongTensor] = None,
    max_length_q: Optional[int] = None,
    max_length_k: Optional[int] = None,
    target_dtype: Optional[torch.dtype] = None,
    **kwargs,
):
    """
    Flash attention forward that ONLY handles training mode with cu_seqlens.
    For everything else, fall back to the transformers default implementation.
    """
    # Only handle the very specific case of training with flattened data
    is_training_with_cu_seqlens = (
        attention_mask is not None and 
        attention_mask.dtype != torch.bool and
        attention_mask.dim() == 1 and  # cu_seqlens is 1D
        query_states.size(0) == 1  # batch size 1 typical for flattened
    )
    
    if is_training_with_cu_seqlens:
        # print(f"GRPO: Using cu_seqlens training mode")
        
        # Training mode with cu_seqlens (flattened data format)
        assert query_states.size(0) == key_states.size(0) == value_states.size(0) == 1
        query_states = query_states.squeeze(0)
        key_states = key_states.squeeze(0)
        value_states = value_states.squeeze(0)
        cu_seqlens = attention_mask

        with torch.no_grad():
            max_seqlen = max(
                [
                    cu_seqlens[idx + 1] - cu_seqlens[idx]
                    for idx in range(cu_seqlens.size(0) - 1)
                ]
            ).item()

        if not use_top_left_mask:
            causal = is_causal
        else:
            causal = is_causal and query_length != 1

        flash_kwargs = {}
        if softcap is not None:
            flash_kwargs["softcap"] = softcap

        attn_output = flash_attn_varlen_func(
            query_states,
            key_states,
            value_states,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            dropout_p=dropout,
            softmax_scale=softmax_scale,
            causal=causal,
            **flash_kwargs,
        )

        attn_output = attn_output.unsqueeze(0)
        return attn_output
    
    else:
        # print(f"GRPO: Falling back to original transformers implementation")
        
        # For everything else (including generation), use the original transformers implementation
        # We need to temporarily restore the original function to avoid infinite recursion
        
        # Get the original function from transformers
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl as qwen_module
        
        # This is a bit tricky - we need to get the original implementation
        # The safest way is to import it fresh from transformers without our override
        
        # Temporarily disable our override by importing fresh
        import importlib
        import transformers.models.qwen2_5_vl.modeling_qwen2_5_vl
        importlib.reload(transformers.models.qwen2_5_vl.modeling_qwen2_5_vl)
        
        # Get the original function
        original_func = transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward
        
        # Call it
        result = original_func(
            query_states=query_states,
            key_states=key_states,
            value_states=value_states,
            attention_mask=attention_mask,
            query_length=query_length,
            is_causal=is_causal,
            dropout=dropout,
            position_ids=position_ids,
            softmax_scale=softmax_scale,
            sliding_window=sliding_window,
            use_top_left_mask=use_top_left_mask,
            softcap=softcap,
            deterministic=deterministic,
            cu_seq_lens_q=cu_seq_lens_q,
            cu_seq_lens_k=cu_seq_lens_k,
            max_length_q=max_length_q,
            max_length_k=max_length_k,
            target_dtype=target_dtype,
            **kwargs,
        )
        
        # Re-apply our override for future calls
        replace_qwen2_vl_attention_class_grpo()
        
        return result


def replace_qwen2_vl_attention_class_grpo():
    """Replace flash attention with GRPO-compatible version."""
    import transformers
    
    transformers.models.qwen2_vl.modeling_qwen2_vl._flash_attention_forward = (
        _flash_attention_forward_grpo
    )
    transformers.models.qwen2_5_vl.modeling_qwen2_5_vl._flash_attention_forward = (
        _flash_attention_forward_grpo
    )