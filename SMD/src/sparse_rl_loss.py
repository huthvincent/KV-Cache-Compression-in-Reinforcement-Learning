"""Sparse-RL Baseline: Post-hoc statistical correction for off-policy RL.

This module implements the Sparse-RL method (arXiv:2601.10079) as a baseline
for comparison with Shadow Mask Distillation. Unlike SMD, which proactively
aligns the training signal with the compressed policy, Sparse-RL attempts to
fix the off-policy bias retroactively using two statistical mechanisms:

    Mechanism A — Rejection Sampling:
        Computes the sequence-level divergence between the dense policy π_dense
        and the sparse rollout policy π_sparse. Rollouts with the highest
        divergence (top 20% by default) are discarded entirely, under the
        assumption that they are too far off-policy to be useful.

    Mechanism B — Importance Reweighting:
        For retained rollouts, computes per-token importance ratios
        ρ(t) = π_dense(t) / π_sparse(t) and uses clipped ρ to reweight
        the GRPO advantages. This partially corrects the distribution
        mismatch at the token level.

    Total Loss = GRPO_loss(π_dense, ρ-weighted advantages, rejection mask)

Key Limitation:
    Rejection sampling discards 20% of expensive rollout compute. In contrast,
    SMD uses ALL rollouts by proactively aligning the training signal, achieving
    better sample efficiency and lower variance (see Exp_02 results).

Configurable Hyperparameters:
    REJECTION_RATIO:     Fraction of most-divergent rollouts to discard (default: 0.20)
    IMPORTANCE_CLIP_LOW: Lower bound for importance ratio clipping (default: 0.8)
    IMPORTANCE_CLIP_HIGH: Upper bound for importance ratio clipping (default: 1.2)
"""

import logging
from argparse import Namespace
from collections.abc import Callable

import torch
import torch.nn.functional as F

logger = logging.getLogger(__name__)

# ── Configurable Hyperparameters ────────────────────────────────────────
REJECTION_RATIO = 0.20       # Discard the most divergent 20% of rollouts
IMPORTANCE_CLIP_LOW = 0.8    # Lower bound for importance ratio clipping
IMPORTANCE_CLIP_HIGH = 1.2   # Upper bound for importance ratio clipping


def sparse_rl_loss_function(
    args: Namespace,
    batch: dict,
    logits: torch.Tensor,
    sum_of_sample_mean: Callable[[torch.Tensor], torch.Tensor],
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compute the Sparse-RL GRPO loss with rejection sampling and importance reweighting.

    This function serves as a drop-in replacement for the standard GRPO policy loss,
    using the same interface as shadow_distillation_loss_function.

    Algorithm:
        1. Compute log-probs under the current dense policy π_dense.
        2. For each rollout sample, measure the average |log π_dense - log π_sparse|
           as a proxy for off-policy divergence.
        3. Reject the top REJECTION_RATIO fraction of most-divergent samples.
        4. For retained samples, compute importance ratios ρ = π_dense / π_sparse,
           clip them to [IMPORTANCE_CLIP_LOW, IMPORTANCE_CLIP_HIGH], and use them
           to reweight the GRPO advantages.
        5. Compute the standard PPO-style clipped surrogate objective with the
           reweighted advantages and rejection masks.

    Args:
        args: Training configuration namespace.
        batch: Rollout batch dictionary (same format as standard GRPO).
        logits: Dense forward pass logits, shape (1, T, V).
        sum_of_sample_mean: Aggregation function for distributed training.

    Returns:
        Tuple of (total_loss, metrics_dict) containing Sparse-RL specific metrics.
    """
    # Framework-specific imports (replace for non-Slime frameworks)
    from slime.backends.megatron_utils.loss import (
        get_log_probs_and_entropy,
    )

    response_lengths = batch["response_lengths"]
    total_lengths = batch["total_lengths"]
    max_seq_lens = batch.get("max_seq_lens", None)

    # ── Step 1: Extract log-probs from dense forward pass ───────────────
    _, log_probs_and_entropy = get_log_probs_and_entropy(
        logits,
        args=args,
        unconcat_tokens=batch["unconcat_tokens"],
        total_lengths=total_lengths,
        response_lengths=response_lengths,
        with_entropy=True,
        max_seq_lens=max_seq_lens,
    )

    log_probs = log_probs_and_entropy["log_probs"]
    entropy_list = log_probs_and_entropy["entropy"]

    old_log_probs_list = (
        batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]
    )
    advantages_list = batch["advantages"]
    loss_masks_list = batch["loss_masks"]
    num_samples = len(log_probs)

    # ── Step 2: Mechanism A — Rejection Sampling ────────────────────────
    rollout_lp_list = batch.get("rollout_log_probs", old_log_probs_list)

    sample_divergences = []
    for i in range(num_samples):
        dense_lp = log_probs[i].detach()
        sparse_lp = rollout_lp_list[i].to(dense_lp.device)
        mask = loss_masks_list[i].float().to(dense_lp.device)

        # Sequence-level mean absolute log-prob divergence
        diff = (dense_lp - sparse_lp).abs() * mask
        valid_tokens = mask.sum().clamp(min=1)
        avg_div = diff.sum() / valid_tokens
        sample_divergences.append(avg_div)

    divergences = torch.stack(sample_divergences)

    # Sort by divergence and reject the worst samples
    num_reject = max(1, int(num_samples * REJECTION_RATIO))
    num_keep = num_samples - num_reject

    if num_samples > 1:
        _, sorted_indices = torch.sort(divergences)
        keep_set = set(sorted_indices[:num_keep].tolist())
        reject_mask = torch.tensor(
            [1.0 if i in keep_set else 0.0 for i in range(num_samples)],
            device=logits.device
        )
    else:
        reject_mask = torch.ones(num_samples, device=logits.device)

    actual_rejection_rate = 1.0 - reject_mask.mean().item()

    # ── Step 3: Mechanism B — Importance Reweighting ────────────────────
    pg_losses = []
    pg_clipfracs = []
    all_entropies = []
    importance_ratios_all = []

    for i in range(num_samples):
        lp = log_probs[i]
        old_lp = old_log_probs_list[i].to(lp.device)
        sparse_lp = rollout_lp_list[i].to(lp.device)
        adv = advantages_list[i].to(lp.device)
        lm = loss_masks_list[i].float().to(lp.device)

        sample_weight = reject_mask[i]  # 0.0 for rejected samples

        # Importance ratio: ρ = π_dense / π_sparse = exp(log π_dense - log π_sparse)
        importance_ratio = torch.exp(lp.detach() - sparse_lp)
        clipped_rho = torch.clamp(importance_ratio, IMPORTANCE_CLIP_LOW, IMPORTANCE_CLIP_HIGH)
        importance_ratios_all.append(importance_ratio.detach())

        # Reweight advantages with clipped importance ratio
        reweighted_adv = adv * clipped_rho

        # Standard PPO clipped surrogate
        ratio = torch.exp(lp - old_lp)
        pg_loss1 = -reweighted_adv * ratio
        pg_loss2 = -reweighted_adv * torch.clamp(ratio, 1.0 - args.eps_clip, 1.0 + args.eps_clip)
        pg_loss = torch.max(pg_loss1, pg_loss2)

        pg_loss = pg_loss * lm * sample_weight
        clipfrac = ((ratio - 1.0).abs() > args.eps_clip).float() * lm * sample_weight

        pg_losses.append(pg_loss)
        pg_clipfracs.append(clipfrac)

        if i < len(entropy_list):
            all_entropies.append(entropy_list[i] * lm * sample_weight)

    # ── Step 4: Aggregate losses ────────────────────────────────────────
    pg_loss_cat = torch.cat(pg_losses, dim=0)
    pg_clipfrac_cat = torch.cat(pg_clipfracs, dim=0)

    pg_loss = sum_of_sample_mean(pg_loss_cat)
    pg_clipfrac = sum_of_sample_mean(pg_clipfrac_cat)

    entropy_loss = torch.tensor(0.0, device=logits.device)
    if all_entropies:
        entropy_loss = sum_of_sample_mean(torch.cat(all_entropies, dim=0))

    total_loss = pg_loss - args.entropy_coef * entropy_loss

    if torch.cat(log_probs, dim=0).numel() == 0:
        total_loss += 0 * logits.sum()

    # ── Step 5: Metrics ─────────────────────────────────────────────────
    all_rho = torch.cat(importance_ratios_all)
    metrics = {
        "loss": total_loss.clone().detach(),
        "sparse_rl/pg_loss": pg_loss.clone().detach(),
        "sparse_rl/entropy": entropy_loss.clone().detach(),
        "sparse_rl/clipfrac": pg_clipfrac.clone().detach(),
        "sparse_rl/rejection_rate": torch.tensor(actual_rejection_rate, device=logits.device),
        "sparse_rl/importance_ratio_mean": all_rho.mean(),
        "sparse_rl/importance_ratio_std": all_rho.std() if all_rho.numel() > 1 else torch.tensor(0.0, device=logits.device),
        "sparse_rl/avg_divergence": divergences.mean().detach(),
    }

    return total_loss, metrics
