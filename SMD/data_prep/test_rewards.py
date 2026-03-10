"""Sanity check: load 1 sample from each dataset and test reward functions."""
import json
import sys

sys.path.insert(0, "/home/zhu11/RLKV/RLKV_github/SMD")
from src.rewards import compute_govreport_reward, compute_hotpotqa_reward

# === GovReport ===
print("=== GovReport ===")
with open("/home/zhu11/RLKV/manifold/data/gov_report/train.jsonl") as f:
    sample = json.loads(f.readline())
print(f"  Prompt: {len(sample['prompt'].split())} words")
print(f"  Label:  {len(sample['label'].split())} words")

partial = " ".join(sample["label"].split()[:100])
r1 = compute_govreport_reward(partial, sample["label"])
r2 = compute_govreport_reward("Irrelevant response.", sample["label"])
r3 = compute_govreport_reward(sample["label"], sample["label"])
print(f"  Partial: {r1:.4f}, Irrelevant: {r2:.4f}, Exact: {r3:.4f}")
assert 0 < r1 < 1 and r2 < 0.1 and r3 > 0.99
print("  ✅ PASSED")

# === HotpotQA ===
print("\n=== HotpotQA ===")
with open("/home/zhu11/RLKV/manifold/data/hotpot_qa/train.jsonl") as f:
    sample = json.loads(f.readline())
print(f"  Prompt: {len(sample['prompt'].split())} words")
print(f"  Label:  \"{sample['label']}\"")

r1 = compute_hotpotqa_reward(sample["label"], sample["label"])
r2 = compute_hotpotqa_reward(f"The answer is {sample['label']}.", sample["label"])
r3 = compute_hotpotqa_reward("Wrong answer entirely", sample["label"])
print(f"  Exact: {r1:.1f}, Embedded: {r2:.1f}, Wrong: {r3:.1f}")
assert r1 == 1.0 and r2 == 1.0 and r3 == 0.0
print("  ✅ PASSED")

print("\n🎉 ALL CHECKS PASSED!")
