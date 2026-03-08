"""Native HuggingFace Rollout Engine with True KV Cache Eviction.

This module provides a custom rollout function that bypasses inference engines
like SGLang/vLLM and uses raw HuggingFace `transformers` for autoregressive
generation. The key feature is **true physical KV cache eviction**: during
generation, the `past_key_values` tensor is physically sliced to remove
evicted tokens, creating genuine off-policy rollout data.

Why is this needed?
    Production inference engines (SGLang, vLLM) manage KV cache internally using
    paged memory and CUDA kernels. There is no Python-level API to selectively
    evict specific KV cache entries. This native engine gives us full control
    over the KV cache tensor, enabling true eviction experiments (e.g., Exp_01
    and Exp_04 in the paper).

Important VRAM Caveat (Exp_04 finding):
    PyTorch tensor slicing is NOT an in-place operation. When we slice
    `past_key_values` from 100% to 50%, PyTorch allocates a NEW tensor for the
    50% view before the GC can free the original 100% tensor. This causes a
    transient ~1.5x memory spike. See `apply_kv_compression()` for details.

Usage:
    This module is intended to be loaded dynamically via:
    ```bash
    --rollout-function-path smd.native_hf_rollout.generate_rollout
    ```

    The function signature follows the Slime rollout function interface:
    `generate_rollout(args, rollout_id, data_buffer, evaluation=False)`
"""

import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

logger = logging.getLogger(__name__)

# Module-level model cache (singleton pattern for efficiency)
_MODEL = None
_TOKENIZER = None


def get_model(args):
    """Lazily load the HuggingFace model and tokenizer.

    The model is loaded on first call and cached for subsequent rollouts.
    This avoids the overhead of reloading weights for every rollout batch.

    Note: The model is loaded onto GPU:0 regardless of torch.cuda.current_device(),
    because Ray workers may not have CUDA properly initialized at import time.

    Args:
        args: Namespace with `hf_checkpoint` attribute pointing to model directory.

    Returns:
        Tuple of (model, tokenizer).
    """
    global _MODEL, _TOKENIZER
    if _MODEL is None:
        import os
        if "CUDA_VISIBLE_DEVICES" in os.environ and not os.environ["CUDA_VISIBLE_DEVICES"]:
            os.environ["CUDA_VISIBLE_DEVICES"] = "0"
            logger.info("Forced CUDA_VISIBLE_DEVICES=0 for RolloutManager (colocate workaround)")

        logger.info(f"Loading native HF Model from {args.hf_checkpoint}...")
        _TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
        device = torch.device("cuda:0")
        _MODEL = AutoModelForCausalLM.from_pretrained(
            args.hf_checkpoint,
            torch_dtype=torch.bfloat16,
            device_map={"": device},
            trust_remote_code=True
        )
        _MODEL.eval()
    return _MODEL, _TOKENIZER


def apply_kv_compression(past_key_values, retention_ratio, strategy="snapkv"):
    """Physically slice the KV cache to simulate true cache eviction.

    This is where the actual memory benefit (or lack thereof — see Exp_04) happens.
    The function converts the DynamicCache to legacy format, slices each layer's K/V
    tensors to retain only the selected positions, and converts back.

    IMPORTANT — Not-In-Place Allocation Spike:
        When we do `k_kept = k[:, :, all_kept, :]`, PyTorch creates a NEW contiguous
        tensor. The original tensor `k` is NOT freed until after this line completes.
        This means peak memory is briefly ~1.5x the KV cache size. This is the root
        cause of why naive KV eviction doesn't save VRAM at the Python level — you
        need CUDA kernel-level block unmapping (like vLLM's prefix caching) to achieve
        true memory savings.

    The token selection logic:
        - Head tokens (first 10%) and tail tokens (last 20%) are always protected.
        - Middle tokens are selected based on the strategy:
            * "snapkv": Scored by harmonic-mean distance to head/tail boundaries.
            * "random": Uniformly random selection.
            * "recent": Keep the most recent middle tokens.

    Args:
        past_key_values: DynamicCache from the prefill pass.
        retention_ratio: Fraction of KV positions to retain (e.g., 0.5).
        strategy: Selection strategy ("snapkv", "random", or "recent").

    Returns:
        New DynamicCache with physically smaller K/V tensors.
    """
    legacy = past_key_values.to_legacy_cache()
    b, h, n, d = legacy[0][0].shape

    num_keep = int(n * retention_ratio)
    head_size = max(int(n * 0.10), 4)
    tail_size = max(int(n * 0.20), 4)

    if head_size + tail_size >= num_keep:
        head_size = num_keep // 2
        tail_size = num_keep - head_size

    protected_indices = set(range(head_size)) | set(range(n - tail_size, n))
    middle_indices = [i for i in range(n) if i not in protected_indices]
    num_middle_keep = max(0, num_keep - len(protected_indices))

    if num_middle_keep < len(middle_indices):
        if strategy == "snapkv":
            scored = []
            for pos in middle_indices:
                dist_head = pos - head_size + 1
                dist_tail = (n - tail_size) - pos
                score = 1.0 / (dist_head + 1) + 1.0 / (dist_tail + 1)
                scored.append((score, pos))
            scored.sort(reverse=True)
            kept_middle = sorted([pos for _, pos in scored[:num_middle_keep]])
        elif strategy == "random":
            import random
            kept_middle = sorted(random.sample(middle_indices, num_middle_keep))
        elif strategy == "recent":
            kept_middle = middle_indices[-num_middle_keep:]
        else:
            kept_middle = middle_indices[:num_middle_keep]
    else:
        kept_middle = middle_indices

    all_kept = sorted(list(protected_indices | set(kept_middle)))

    new_legacy = []
    for k, v in legacy:
        k_kept = k[:, :, all_kept, :]
        v_kept = v[:, :, all_kept, :]
        new_legacy.append((k_kept, v_kept))

    return DynamicCache.from_legacy_cache(tuple(new_legacy))


def generate_rollout(args, rollout_id: int, data_buffer, evaluation=False):
    """Generate rollout samples using the native HuggingFace model.

    This function follows the Slime rollout function interface and can be
    loaded dynamically via --rollout-function-path.

    The generation process:
        1. Encode the prompt and run the prefill pass to get initial KV cache.
        2. If shadow mask is enabled, apply KV compression (true physical eviction).
        3. Run autoregressive generation with the (optionally compressed) KV cache.
        4. Collect generated tokens and per-token log-probabilities.

    Args:
        args: Training configuration namespace.
        rollout_id: Current rollout iteration index.
        data_buffer: Data buffer providing prompt samples.
        evaluation: Must be False (evaluation not supported in native mode).

    Returns:
        RolloutFnTrainOutput containing the generated samples with tokens,
        responses, log-probs, and metadata.
    """
    from slime.utils.types import Sample
    from slime.rollout.base_types import RolloutFnTrainOutput

    assert not evaluation, "Eval not supported in native_hf_rollout"
    model, tokenizer = get_model(args)

    raw_samples = data_buffer.get_samples(args.rollout_batch_size)

    out_samples = []

    shadow_ratio = args.shadow_retention_ratio if getattr(args, "use_shadow_mask", False) else 1.0
    shadow_strategy = getattr(args, "shadow_strategy", "snapkv")

    for prompt_tuple in raw_samples:
        prompt_group = []
        batch_size = len(prompt_tuple)

        prompt = prompt_tuple[0].prompt
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        input_ids = torch.tensor([prompt_ids] * batch_size, device=model.device)

        # Step 1: Prefill — compute initial KV cache
        with torch.no_grad():
            outputs = model(input_ids=input_ids, use_cache=True)

        past_key_values = outputs.past_key_values
        next_logits = outputs.logits[:, -1, :]

        temp = getattr(args, "rollout_temperature", 1.0)

        def sample_tokens(logits, temp):
            if temp < 1e-4:
                return torch.argmax(logits, dim=-1).unsqueeze(-1)
            probs = torch.softmax(logits / temp, dim=-1)
            return torch.multinomial(probs, num_samples=1)

        next_tokens = sample_tokens(next_logits, temp)

        log_probs_list = [[] for _ in range(batch_size)]

        def append_log_probs(logits, tokens, lp_list):
            logs = torch.log_softmax(logits, dim=-1)
            for b in range(logits.shape[0]):
                tok = tokens[b, 0].item()
                lp_list[b].append(logs[b, tok].item())

        append_log_probs(next_logits, next_tokens, log_probs_list)

        generated_tokens = [[] for _ in range(batch_size)]
        for b in range(batch_size):
            generated_tokens[b].append(next_tokens[b, 0].item())

        # Step 2: True KV Eviction (if compression is enabled)
        if shadow_ratio < 1.0:
            past_key_values = apply_kv_compression(past_key_values, shadow_ratio, shadow_strategy)

        cur_pos = len(prompt_ids)
        max_new_tokens = args.rollout_max_response_len

        unfinished = [True] * batch_size
        eos_token_id = tokenizer.eos_token_id

        # Step 3: Autoregressive generation with (compressed) KV cache
        for _ in range(max_new_tokens - 1):
            if not any(unfinished):
                break

            position_ids = torch.tensor([[cur_pos]] * batch_size, device=model.device)
            past_len = past_key_values.get_seq_length()
            att_mask = torch.ones((batch_size, past_len + 1), device=model.device, dtype=torch.long)

            with torch.no_grad():
                outputs = model(
                    input_ids=next_tokens,
                    past_key_values=past_key_values,
                    use_cache=True,
                    position_ids=position_ids,
                    attention_mask=att_mask
                )

            past_key_values = outputs.past_key_values
            next_logits = outputs.logits[:, -1, :]
            next_tokens = sample_tokens(next_logits, temp)

            append_log_probs(next_logits, next_tokens, log_probs_list)

            for b in range(batch_size):
                if unfinished[b]:
                    tok = next_tokens[b, 0].item()
                    generated_tokens[b].append(tok)
                    if tok == eos_token_id or tok in [151645, 151643]:
                        unfinished[b] = False

            cur_pos += 1

        # Step 4: Package results into Sample objects
        for b in range(batch_size):
            s = prompt_tuple[b]
            s.tokens = prompt_ids + generated_tokens[b]
            s.response = tokenizer.decode(generated_tokens[b], skip_special_tokens=True)
            s.response_length = len(generated_tokens[b])
            s.status = Sample.Status.TRUNCATED if unfinished[b] else Sample.Status.COMPLETED
            s.reward = 0
            s.loss_mask = [1] * s.response_length
            s.rollout_log_probs = log_probs_list[b][:s.response_length]
            s.rollout_routed_experts = None
            s.teacher_log_probs = None
            s.multimodal_train_inputs = None
            s.train_metadata = None

            prompt_group.append(s)

        out_samples.append(prompt_group)

    peak_mem = torch.cuda.max_memory_allocated(model.device) / (1024**3)
    logger.info(f"PEAK MEMORY DURING ROLLOUT: {peak_mem:.3f} GB")
    torch.cuda.reset_peak_memory_stats(model.device)

    return RolloutFnTrainOutput(samples=out_samples)
