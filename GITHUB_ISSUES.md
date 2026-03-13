# GitHub Issues 模板

> **使用方法**: 
> 1. 打开 https://github.com/huthvincent/RL-Post-Training-with-KV-Cache-Compression/issues/new
> 2. 复制下面每个 Issue 的内容（从 `## Title:` 到下一个 Issue 之前）
> 3. 粘贴到 GitHub Issue 编辑器中
> 4. 添加对应的 Labels

---

# 🔴 严重问题 (Critical)

---

## Title: [BUG] KL散度计算错误 - 使用绝对值差异而非真正的KL散度

**Labels**: `bug`, `critical`, `loss-function`

### 问题描述
`SMD/src/shadow_distillation_loss.py` 中 Track-2 的 KL 蒸馏损失使用了绝对值差异而非真正的 KL 散度，这在数学上是不正确的。

### 当前代码
```python
# 文件: SMD/src/shadow_distillation_loss.py
# 行号: 约 170-175 行

kl_per_token = (dense_lp - shadow_target_lp).abs()
kl_distill_loss = sum_of_sample_mean(kl_per_token)
```

### 问题分析
- `|log π_dense - log π_sparse|` 不是 KL 散度
- 真正的 KL 散度: `D_KL(P||Q) = Σ P(x) * log(P(x)/Q(x))`
- 当前实现只是 log-prob 差异的绝对值

### 建议修复
```python
# 方案A: 使用真正的 Forward KL (推荐)
# D_KL(π_shadow || π_dense) = E_shadow[log π_shadow - log π_dense]
kl_per_token = shadow_target_lp - dense_lp  # shadow 视角的 KL
kl_distill_loss = sum_of_sample_mean(kl_per_token.clamp(min=0))  # 只惩罚正向偏离

# 方案B: 使用 Reverse KL
# D_KL(π_dense || π_shadow) = E_dense[log π_dense - log π_shadow]
kl_per_token = dense_lp - shadow_target_lp
kl_distill_loss = sum_of_sample_mean(torch.exp(dense_lp) * kl_per_token)
```

### 影响范围
- 影响所有使用 SMD 方法的训练
- 可能导致训练不稳定或收敛到次优解

---

## Title: [BUG] Shadow Token Mask 逻辑可能反转

**Labels**: `bug`, `critical`, `loss-function`

### 问题描述
`SMD/src/shadow_distillation_loss.py` 中 `visibility_ratio < 1.0` 的判断逻辑可能与预期相反。

### 当前代码
```python
# 文件: SMD/src/shadow_distillation_loss.py
# 行号: 约 130-140 行

visibility_ratio = resp_mask.float().sum(dim=-1) / total_len
shadow_token_masks.append(
    (visibility_ratio < 1.0).to(logits.device)  # 可见度 < 100% 的 token
)
```

### 问题分析
- 注释说"如果 visibility_ratio < 1.0, 说明有 KV 被压缩，token 在稀疏条件下生成"
- 但这意味着**所有**在压缩后生成的 token 都被标记为"可信"
- 实际上应该是：只有那些**完全**在保留的 KV 上下文中生成的 token 才是"on-policy faithful"

### 建议修复
```python
# 修复：检查每个 token 是否只依赖于保留的 KV 位置
retention_ratio = getattr(args, "shadow_retention_ratio", 0.5)
# 只有当可见度接近 retention_ratio 时，token 才是在压缩条件下生成的
shadow_token_masks.append(
    (visibility_ratio <= retention_ratio + 0.1).to(logits.device)
)
```

### 影响范围
- 影响 shadow mask 的正确性
- 可能导致错误的 token 被标记为"可信"

---

## Title: [BUG] 除零风险 - prompt_length=0 时会导致失败

**Labels**: `bug`, `critical`, `edge-case`

### 问题描述
`SMD/src/shadow_mask_interceptor.py` 中 `_select_by_position_heuristic` 方法在 `prompt_length=0` 时会导致后续 `topk` 操作失败。

### 当前代码
```python
# 文件: SMD/src/shadow_mask_interceptor.py
# 行号: 约 200-210 行

def _select_by_position_heuristic(self, prompt_length: int, num_keep: int) -> torch.Tensor:
    # 缺少 prompt_length=0 的检查
    importance = torch.zeros(prompt_length, dtype=torch.float32)
    for i in range(prompt_length):
        # ...
    _, top_indices = importance.topk(num_keep)  # 如果 prompt_length=0，这里会失败
```

### 建议修复
```python
def _select_by_position_heuristic(self, prompt_length: int, num_keep: int) -> torch.Tensor:
    if prompt_length == 0:
        return torch.tensor([], dtype=torch.long)
    if num_keep >= prompt_length:
        return torch.arange(prompt_length)
    
    # ... 原有逻辑
```

### 影响范围
- 边界条件处理
- 可能在特殊输入时导致程序崩溃

---

## Title: [BUG] 测试文件中 assert_ 函数重复定义

**Labels**: `bug`, `critical`, `tests`

### 问题描述
`baselines/test_baselines.py` 中 `assert_` 函数被定义了两次。

### 当前代码
```python
# 文件: baselines/test_baselines.py

# 文件开头 (第一次定义)
def assert_(cond):
    if not cond:
        raise AssertionError("Assertion failed")

# ... 中间代码 ...

# 文件末尾 (重复定义)
def assert_(cond):
    if not cond:
        raise AssertionError("Assertion failed")
```

### 建议修复
删除文件末尾的重复定义。

### 影响范围
- 代码冗余
- 可能导致混淆

---

# 🟠 中等问题 (Medium)

---

## Title: [IMPROVEMENT] 硬编码路径应使用环境变量

**Labels**: `enhancement`, `medium`, `config`

### 问题描述
`SMD/experiments/qwen3_0.6b/run_grpo_training.py` 中模型和数据集路径硬编码。

### 当前代码
```python
# 文件: SMD/experiments/qwen3_0.6b/run_grpo_training.py
# 行号: 约 30-40 行

MODEL_PATH = "/workspace/RLKV/shared_resources/models/Qwen3-0.6B"

DATASET_CONFIGS = {
    "tldr": {"data_file": "/workspace/RLKV/shared_resources/datasets/tldr/train.jsonl"},
    # ...
}
```

### 建议修复
```python
import os

# 从环境变量或相对路径获取
REPO_ROOT = os.environ.get("RLKV_ROOT", "/workspace/RLKV")
MODEL_PATH = os.path.join(REPO_ROOT, "shared_resources/models/Qwen3-0.6B")

DATASET_CONFIGS = {
    "tldr": {"data_file": os.path.join(REPO_ROOT, "shared_resources/datasets/tldr/train.jsonl")},
    # ...
}
```

### 影响范围
- 可移植性
- 不同环境部署困难

---

## Title: [IMPROVEMENT] 缺少空值检查 - attention buffer

**Labels**: `enhancement`, `medium`, `robustness`

### 问题描述
`SMD/src/attention_extraction.py` 中 `get_per_key_importance` 函数对 `_ATTENTION_BUFFER` 的访问缺少完整的空值检查。

### 当前代码
```python
def get_per_key_importance(num_recent_queries: int = 64) -> torch.Tensor | None:
    if not _ATTENTION_BUFFER:
        return None

    layers = list(_ATTENTION_BUFFER.values())
    if not layers:
        return None

    attn = layers[-1]  # 可能是空 tensor 或 None
```

### 建议修复
```python
def get_per_key_importance(num_recent_queries: int = 64) -> torch.Tensor | None:
    if not _ATTENTION_BUFFER:
        return None

    layers = list(_ATTENTION_BUFFER.values())
    if not layers:
        return None

    attn = layers[-1]
    if attn is None or attn.numel() == 0:
        return None
    
    # ... 继续处理
```

---

## Title: [IMPROVEMENT] 异常处理过于宽泛

**Labels**: `enhancement`, `medium`, `error-handling`

### 问题描述
`SMD/src/shadow_distillation_loss.py` 中使用 `except Exception` 捕获所有异常，可能隐藏真正的错误。

### 当前代码
```python
try:
    from SMD.src.attention_extraction import get_per_key_importance
    # ...
except ImportError:
    pass
except Exception as e:
    logger.warning(f"Failed to regenerate shadow masks with real attention: {e}")
```

### 建议修复
```python
try:
    from SMD.src.attention_extraction import get_per_key_importance
    # ...
except ImportError:
    logger.debug("attention_extraction module not available, using position-based masks")
except (RuntimeError, ValueError, IndexError) as e:
    logger.warning(f"Failed to regenerate shadow masks with real attention: {e}")
except Exception as e:
    logger.error(f"Unexpected error in shadow mask regeneration: {e}", exc_info=True)
    raise  # 重新抛出未知异常
```

---

## Title: [IMPROVEMENT] 内存泄漏风险 - 全局 attention buffer

**Labels**: `enhancement`, `medium`, `memory`

### 问题描述
`SMD/src/attention_extraction.py` 中全局 `_ATTENTION_BUFFER` 可能在长时间训练中累积大量数据。

### 当前代码
```python
_ATTENTION_BUFFER: OrderedDict[int, torch.Tensor] = OrderedDict()
```

### 建议修复
```python
# 添加最大容量限制
MAX_BUFFER_SIZE = 100  # 最多保存 100 层的 attention

def _make_attention_hook(layer_idx: int):
    def hook_fn(module, args, output):
        global _ATTENTION_BUFFER
        # 限制 buffer 大小
        if len(_ATTENTION_BUFFER) >= MAX_BUFFER_SIZE:
            _ATTENTION_BUFFER.popitem(last=False)  # 移除最旧的
        
        if hasattr(module, "attention_probs") and module.attention_probs is not None:
            _ATTENTION_BUFFER[layer_idx] = module.attention_probs.detach()
        # ...
    return hook_fn
```

---

## Title: [IMPROVEMENT] 不一致的设备处理

**Labels**: `enhancement`, `medium`, `device`

### 问题描述
`baselines/kv_compression/r_kv.py` 中 `importance` 和 `redundancy` 可能在不同设备上。

### 当前代码
```python
z_scores = self.lam * importance - (1 - self.lam) * redundancy.to(importance.device)
```

### 建议修复
```python
def compute_eviction(self, key_states: torch.Tensor, attn_weights: torch.Tensor) -> torch.Tensor:
    device = key_states.device
    num_keys = key_states.shape[0]
    
    # 确保所有输入在同一设备
    key_states = key_states.to(device)
    attn_weights = attn_weights.to(device)
    
    # ... 后续计算
```

---

## Title: [IMPROVEMENT] 缺少 label 参数空值检查

**Labels**: `enhancement`, `medium`, `validation`

### 问题描述
`SMD/src/rewards/__init__.py` 中各 reward 函数缺少对 `label` 参数的空值检查。

### 当前代码
```python
def compute_rouge_reward(response: str, label: str) -> float:
    if not response or not response.strip():
        return 0.0
    # 缺少对 label 的检查
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    scores = scorer.score(label.strip(), response.strip())  # label 为空会出错
```

### 建议修复
```python
def compute_rouge_reward(response: str, label: str) -> float:
    if not response or not response.strip():
        return 0.0
    if not label or not label.strip():
        logger.warning("Empty label provided for ROUGE reward computation")
        return 0.0
    # ...
```

---

## Title: [IMPROVEMENT] 随机种子设置不完整

**Labels**: `enhancement`, `medium`, `reproducibility`

### 问题描述
`SMD/experiments/exp01_qwen3_1.7b/run_exp01.py` 中缺少 CUDA 随机种子设置。

### 当前代码
```python
torch.manual_seed(args.seed)
np.random.seed(args.seed)
stdlib_random.seed(args.seed)
# 缺少 CUDA 种子
```

### 建议修复
```python
torch.manual_seed(args.seed)
np.random.seed(args.seed)
stdlib_random.seed(args.seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)  # 多 GPU
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
```

---

## Title: [IMPROVEMENT] 潜在的索引越界

**Labels**: `enhancement`, `medium`, `edge-case`

### 问题描述
`SMD/src/shadow_mask_interceptor.py` 中 `_select_by_real_attention` 方法当 `attention_scores` 维度不匹配时可能越界。

### 当前代码
```python
if attention_scores.dim() == 1:
    importance = attention_scores[:prompt_length].float().clone()
```

### 建议修复
```python
if attention_scores.dim() == 1:
    if attention_scores.shape[0] < prompt_length:
        logger.warning(
            f"Attention scores length ({attention_scores.shape[0]}) < prompt_length ({prompt_length}), "
            "padding with zeros"
        )
        padded = torch.zeros(prompt_length, dtype=attention_scores.dtype, device=attention_scores.device)
        padded[:attention_scores.shape[0]] = attention_scores
        importance = padded.float()
    else:
        importance = attention_scores[:prompt_length].float().clone()
```

---

# 🟡 轻微问题 (Low)

---

## Title: [CLEANUP] 未使用的导入

**Labels**: `cleanup`, `low`

### 问题描述
`SMD/src/shadow_distillation_loss.py` 中导入了 `torch.nn.functional as F` 但未使用。

### 建议修复
删除未使用的导入：
```python
# 删除这行
import torch.nn.functional as F
```

---

## Title: [CLEANUP] 魔法数字应定义为常量

**Labels**: `cleanup`, `low`, `readability`

### 问题描述
`SMD/src/shadow_mask_interceptor.py` 中多处硬编码数字缺少解释。

### 当前代码
```python
importance[i] += prompt_length  # 为什么是 prompt_length?
importance[i] += prompt_length * 0.5  # 为什么是 0.5?
max_imp * 10  # 为什么是 10?
max_imp * 5   # 为什么是 5?
```

### 建议修复
```python
# 定义常量并添加注释
SINK_TOKEN_BOOST_FACTOR = 10.0  # Sink tokens 重要性提升倍数
RECENT_TOKEN_BOOST_FACTOR = 5.0  # 最近 tokens 重要性提升倍数

importance[:config.sink_tokens] += max_imp * SINK_TOKEN_BOOST_FACTOR
importance[-config.observation_window:] += max_imp * RECENT_TOKEN_BOOST_FACTOR
```

---

## Title: [CLEANUP] 日志级别不一致

**Labels**: `cleanup`, `low`, `logging`

### 问题描述
多个文件中日志级别使用不一致，有些地方用 `logger.debug`，有些用 `logger.info`。

### 建议修复
制定日志级别规范：
- `DEBUG`: 详细的调试信息（如每步的 tensor shape）
- `INFO`: 重要的状态变化（如模型加载完成）
- `WARNING`: 可恢复的问题（如回退到备用策略）
- `ERROR`: 严重错误（如数据损坏）

---

## Title: [CLEANUP] 缺少模块级文档字符串

**Labels**: `cleanup`, `low`, `documentation`

### 问题描述
`baselines/kv_compression/random_eviction.py` 缺少模块级文档字符串。

### 建议修复
添加与其他文件一致的文档字符串格式。

---

## Title: [CLEANUP] 代码重复 - 实验脚本

**Labels**: `cleanup`, `low`, `refactor`

### 问题描述
`SMD/experiments/exp01_qwen3_1.7b/run_exp01.py` 和 `SMD/experiments/qwen3_0.6b/run_grpo_training.py` 有大量重复代码。

### 建议修复
提取公共代码到 `SMD/src/training_utils.py`。

---

## Title: [CLEANUP] 缺少 __all__ 导出定义

**Labels**: `cleanup`, `low`, `api`

### 问题描述
`baselines/__init__.py` 缺少 `__all__` 定义，不清楚哪些是公开 API。

### 建议修复
```python
__all__ = [
    "BASELINE_REGISTRY",
    "sparse_rl_loss_function",
    "qurl_loss_function",
]
```

---

# 📋 Labels 说明

在 GitHub 仓库中创建以下 Labels：

| Label | 颜色 | 说明 |
|:---|:---|:---|
| `bug` | `#d73a4a` (红色) | 代码错误 |
| `enhancement` | `#a2eeef` (青色) | 改进建议 |
| `cleanup` | `#fef2c0` (黄色) | 代码清理 |
| `critical` | `#b60205` (深红) | 严重程度：严重 |
| `medium` | `#fbca04` (橙色) | 严重程度：中等 |
| `low` | `#0e8a16` (绿色) | 严重程度：轻微 |
| `loss-function` | `#5319e7` (紫色) | 涉及损失函数 |
| `tests` | `#1d76db` (蓝色) | 涉及测试 |
| `config` | `#c5def5` (浅蓝) | 涉及配置 |
| `memory` | `#d4c5f9` (浅紫) | 涉及内存 |
| `edge-case` | `#bfdadc` (浅青) | 边界条件 |

---

*共 18 个 Issues，按严重程度分类*