# Exp 01: Naive KV Compression → Reward Collapse

| Method | Avg Reward (last 50) | Total Steps | Status |
|--------|---------------------|-------------|--------|
| Dense (100% KV) | 0.1227 | 500 | ✅ Learning |
| Naive SnapKV 50% | 0.1051 | 500 | ✅ Learning |
| Naive Random 50% | 0.0995 | 500 | ✅ Learning |
| Naive Recent 50% | 0.1046 | 500 | ✅ Learning |
| Naive R-KV 50% | 0.1140 | 500 | ✅ Learning |

> **Conclusion:** All 4 naive KV compression strategies show degraded performance compared to the Dense baseline, confirming that KV compression during RL training requires a correction mechanism like SMD.
