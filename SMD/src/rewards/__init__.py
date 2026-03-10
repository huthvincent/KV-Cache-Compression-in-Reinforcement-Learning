"""Unified reward functions for all SMD datasets.

All reward functions are registered here and dispatched by rm_type string.
This module is designed to be used BOTH standalone (for testing) and
integrated into Slime's rm_hub via the registration hook below.

Reward functions:
  - "rouge"     : ROUGE-L F1 (TL;DR, calibrated with low length penalty)
  - "govreport" : ROUGE-L F1 (GovReport, no length penalty — long summaries)
  - "math"      : Exact match on \\boxed{} answer (GSM8K)
  - "hotpotqa"  : Fuzzy exact match (HotpotQA)
"""
import re
import string

# ═══════════════════════════════════════════════════════════════════════
# GovReport Reward (ROUGE-L, no length penalty)
# ═══════════════════════════════════════════════════════════════════════

def compute_govreport_reward(response: str, label: str) -> float:
    """ROUGE-L F1 reward for government report summarization.

    No length penalty — GovReport summaries are naturally long (200-600 words).
    Returns a float in [0, 1].
    """
    from rouge_score import rouge_scorer
    if not response or not response.strip():
        return 0.0
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(label.strip(), response.strip())
    return scores["rougeL"].fmeasure


# ═══════════════════════════════════════════════════════════════════════
# HotpotQA Reward (Fuzzy Exact Match)
# ═══════════════════════════════════════════════════════════════════════

def _normalize_answer(s: str) -> str:
    """Normalize answer string for comparison.

    Lowercases, removes punctuation/articles/extra whitespace.
    Follows the SQuAD evaluation script convention.
    """
    s = s.lower()
    # Remove punctuation
    s = "".join(ch for ch in s if ch not in string.punctuation)
    # Remove articles
    s = re.sub(r"\b(a|an|the)\b", " ", s)
    # Collapse whitespace
    s = " ".join(s.split())
    return s


def compute_hotpotqa_reward(response: str, label: str) -> float:
    """Sparse reward for HotpotQA: 1.0 if answer matches, 0.0 otherwise.

    Uses normalized exact match with fallback to substring containment.
    This handles cases like:
      - label="Paris", response="The answer is Paris."  → 1.0
      - label="Marie Curie", response="marie curie"     → 1.0
    """
    if not response or not response.strip():
        return 0.0

    norm_response = _normalize_answer(response)
    norm_label = _normalize_answer(label)

    if not norm_label:
        return 0.0

    # Exact match after normalization
    if norm_response == norm_label:
        return 1.0

    # Substring containment (label appears in response)
    if norm_label in norm_response:
        return 1.0

    # Token-level F1 as tiebreaker (partial credit for multi-word answers)
    pred_tokens = norm_response.split()
    gold_tokens = norm_label.split()
    common = set(pred_tokens) & set(gold_tokens)
    if not common:
        return 0.0

    precision = len(common) / len(pred_tokens) if pred_tokens else 0
    recall = len(common) / len(gold_tokens) if gold_tokens else 0
    if precision + recall == 0:
        return 0.0
    f1 = 2 * precision * recall / (precision + recall)

    # Threshold: if F1 >= 0.8, consider it a match
    return 1.0 if f1 >= 0.8 else 0.0


# ═══════════════════════════════════════════════════════════════════════
# Registry (for standalone testing)
# ═══════════════════════════════════════════════════════════════════════

REWARD_REGISTRY = {
    "govreport": compute_govreport_reward,
    "hotpotqa": compute_hotpotqa_reward,
}


def compute_reward(rm_type: str, response: str, label: str) -> float:
    """Dispatch reward computation by type."""
    if rm_type not in REWARD_REGISTRY:
        raise ValueError(f"Unknown rm_type: {rm_type}. Available: {list(REWARD_REGISTRY.keys())}")
    return REWARD_REGISTRY[rm_type](response, label)
