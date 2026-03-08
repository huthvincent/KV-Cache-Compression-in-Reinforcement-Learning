"""Shadow Attention: Attention-level injection of KV compression masks.

This module provides utilities for applying shadow masks to attention computations.
A shadow mask is a boolean tensor that specifies which key-value positions should
be visible to each query position, effectively simulating KV cache compression
within the attention mechanism.

There are two primary use cases:

1. **Eager Attention Path (shadow_masked_attention):** When FlashAttention cannot
   be used (because it doesn't support arbitrary custom masks), this module provides
   a standard QKV matmul + softmax path with shadow mask injection.

2. **Attention Bias Path (create_shadow_attention_bias):** For FlashAttention
   variants that support attention bias tensors, this converts the boolean shadow
   mask into a float bias (0 for visible, -inf for masked).

Why not just use standard attention masking?
    Standard causal masks only enforce left-to-right ordering. Shadow masks are a
    *subset* of the causal mask — they additionally remove specific KV positions
    that would be evicted by a compression algorithm like SnapKV. This creates an
    information bottleneck that acts as a powerful regularizer during RL training.
"""

import logging
from typing import Optional

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)


def apply_shadow_mask_to_scores(
    attn_scores: torch.Tensor,
    shadow_mask: torch.Tensor,
    causal: bool = True,
) -> torch.Tensor:
    """Apply a shadow mask to pre-softmax attention scores.

    Positions where shadow_mask is False are filled with -inf, preventing the
    model from attending to KV entries that would have been evicted by compression.

    Args:
        attn_scores: Attention scores of shape (batch, heads, seq_len, seq_len).
        shadow_mask: Boolean mask where True = visible, False = masked.
            Accepted shapes: (seq_len, seq_len) or (batch, seq_len, seq_len).
        causal: If True, also enforce standard causal (lower-triangular) masking.

    Returns:
        Masked attention scores with -inf at invisible positions.
    """
    if shadow_mask.dim() == 2:
        shadow_mask = shadow_mask.unsqueeze(0).unsqueeze(0)  # (1, 1, Q, K)
    elif shadow_mask.dim() == 3:
        shadow_mask = shadow_mask.unsqueeze(1)  # (B, 1, Q, K)

    q_len = attn_scores.shape[2]
    k_len = attn_scores.shape[3]
    mask_q = shadow_mask.shape[2]
    mask_k = shadow_mask.shape[3]

    # Truncate mask if it's larger than the scores (can happen with padding)
    if mask_q > q_len or mask_k > k_len:
        shadow_mask = shadow_mask[:, :, :q_len, :k_len]

    # Combine shadow mask with causal constraint
    if causal:
        causal_mask = torch.tril(
            torch.ones(q_len, k_len, dtype=torch.bool, device=attn_scores.device)
        )
        shadow_mask = shadow_mask & causal_mask.unsqueeze(0).unsqueeze(0)

    attn_scores = attn_scores.masked_fill(~shadow_mask, float("-inf"))
    return attn_scores


def shadow_masked_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    shadow_mask: Optional[torch.Tensor] = None,
    causal: bool = True,
    dropout_p: float = 0.0,
    scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute attention with shadow mask using eager (non-Flash) implementation.

    This is the fallback attention path used when shadow masks are active.
    FlashAttention does not support arbitrary per-position boolean masks, so we
    fall back to the standard O(n²) QKV matmul + softmax + dropout pipeline.

    When shadow_mask is None, this degenerates to standard causal attention.

    Args:
        query: Query tensor of shape (batch, heads, seq_len, head_dim).
        key: Key tensor of shape (batch, heads, seq_len, head_dim).
        value: Value tensor of shape (batch, heads, seq_len, head_dim).
        shadow_mask: Optional boolean mask (batch, seq_len, seq_len) or None.
        causal: Whether to apply causal masking.
        dropout_p: Attention dropout probability.
        scale: Attention scale factor. Defaults to 1/√head_dim.

    Returns:
        Attention output of shape (batch, heads, seq_len, head_dim).
    """
    head_dim = query.shape[-1]
    if scale is None:
        scale = head_dim ** -0.5

    attn_scores = torch.matmul(query, key.transpose(-2, -1)) * scale

    if shadow_mask is not None:
        attn_scores = apply_shadow_mask_to_scores(attn_scores, shadow_mask, causal=causal)
    elif causal:
        q_len, k_len = attn_scores.shape[-2], attn_scores.shape[-1]
        causal_mask = torch.tril(
            torch.ones(q_len, k_len, dtype=torch.bool, device=attn_scores.device)
        )
        attn_scores = attn_scores.masked_fill(~causal_mask, float("-inf"))

    # Compute in float32 for numerical stability, then cast back
    attn_weights = F.softmax(attn_scores, dim=-1, dtype=torch.float32).to(query.dtype)
    if dropout_p > 0.0:
        attn_weights = F.dropout(attn_weights, p=dropout_p, training=True)

    output = torch.matmul(attn_weights, value)
    return output


def create_shadow_attention_bias(
    shadow_mask: torch.Tensor,
    dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Convert a boolean shadow mask into a float attention bias tensor.

    This is useful for FlashAttention variants that accept an additive attention
    bias rather than a boolean mask. Visible positions get bias 0.0 (no effect),
    masked positions get -inf (effectively zero attention weight after softmax).

    Args:
        shadow_mask: Boolean mask of shape (batch, seq_len, seq_len).
        dtype: Output dtype for the bias tensor.

    Returns:
        Attention bias of shape (batch, 1, seq_len, seq_len).
    """
    attn_bias = torch.zeros_like(shadow_mask, dtype=dtype)
    attn_bias = attn_bias.masked_fill(~shadow_mask, float("-inf"))
    return attn_bias.unsqueeze(1)  # Add head dimension
