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