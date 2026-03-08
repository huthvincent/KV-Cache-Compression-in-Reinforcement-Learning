# RLKV: Reinforcement Learning with KV Cache Compression

A research monorepo for developing and evaluating **KV cache compression methods** integrated with **Reinforcement Learning post-training** (GRPO/PPO).

## Vision

Large Language Models require enormous GPU memory for their KV caches during inference. While compression methods like SnapKV can reduce this footprint, naively applying them during RL training causes **catastrophic policy collapse** due to off-policy mismatch. This repository hosts methods that bridge this gap.

## Projects

| Project | Status | Description |
|---------|--------|-------------|
| **[SMD](./SMD/)** | ✅ Complete | Shadow Mask Distillation — Dual-track GRPO loss for KV-compressed RL |

## Repository Structure

```
RLKV_github/
├── shared_resources/     # Local models & datasets (git-ignored)
│   ├── models/           # HuggingFace model weights
│   └── datasets/         # Training/eval data files
└── SMD/                  # Shadow Mask Distillation project
    ├── src/              # Core Python modules
    ├── scripts/          # Experiment launch scripts
    ├── figures/          # Publication-ready plots
    └── EXPERIMENT_RESULTS.md
```

## Getting Started

See the [SMD README](./SMD/README.md) for installation and usage instructions.

## Citation

```bibtex
@article{smd2026,
  title={Shadow Mask Distillation: Breaking the Memory Wall for RL Post-Training with KV Cache Compression},
  year={2026}
}
```

## License

Apache 2.0
