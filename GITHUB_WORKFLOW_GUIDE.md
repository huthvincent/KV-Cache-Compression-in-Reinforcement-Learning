# GitHub 协作开发指南：Code Review 与 Pull Requests

> **适用场景**: 在不同服务器上协作开发 RLKV 项目
> **目标读者**: 项目开发者

---

## 📚 基础概念

### 什么是 Pull Request (PR)?

**Pull Request** 是 GitHub 的核心协作功能，用于：
- 提议将你的代码变更合并到主分支
- 在合并前让其他人审查你的代码
- 讨论和改进代码

**工作流程**:
```
你的分支 (feature-branch) ──PR──> 主分支 (main)
                            ↑
                      Code Review
```

### 什么是 Code Review?

**Code Review** 是在 PR 中进行的代码审查过程：
- 审查者可以对具体代码行添加评论
- 可以请求修改 (Request Changes)
- 可以批准合并 (Approve)
- 可以只是评论 (Comment)

---

## 🔄 多服务器协作开发流程

假设你有两台服务器：
- **Server A** (sev-cxl): 主要开发环境，有 GPU
- **Server B** (另一台): 辅助开发/测试环境

### 推荐的协作模式

```
┌─────────────────────────────────────────────────────────────┐
│                      GitHub (远程仓库)                        │
│                                                             │
│   main ─────────────────────────────────────────────────    │
│          ↑                    ↑                    ↑        │
│          │ PR                 │ PR                 │ PR     │
│          │                    │                    │        │
│   feature/fix-kl-loss    feature/add-tests    bugfix/xxx   │
└─────────────────────────────────────────────────────────────┘
           ↑                    ↑
           │ push               │ push
           │                    │
    ┌──────┴──────┐      ┌──────┴──────┐
    │  Server A   │      │  Server B   │
    │  (开发)     │      │  (测试/修复) │
    └─────────────┘      └─────────────┘
```

---

## 🛠️ 实战教程：在 RLKV 项目中使用 PR 和 Code Review

### Step 1: 初始设置（每台服务器都要做）

```bash
# 1. 克隆仓库（如果还没有）
git clone https://github.com/huthvincent/RL-Post-Training-with-KV-Cache-Compression.git
cd RL-Post-Training-with-KV-Cache-Compression

# 2. 配置 Git 用户信息
git config user.name "Your Name"
git config user.email "your.email@example.com"

# 3. 添加远程仓库（通常已自动配置）
git remote -v
# 应该显示:
# origin  https://github.com/huthvincent/RL-Post-Training-with-KV-Cache-Compression.git (fetch)
# origin  https://github.com/huthvincent/RL-Post-Training-with-KV-Cache-Compression.git (push)
```

### Step 2: 创建功能分支并开发

```bash
# 1. 确保在最新的 main 分支
git checkout main
git pull origin main

# 2. 创建新的功能分支
# 命名规范: feature/功能名, bugfix/问题名, docs/文档名
git checkout -b bugfix/fix-kl-divergence

# 3. 进行代码修改
# 例如修复 KL 散度计算问题
vim SMD/src/shadow_distillation_loss.py

# 4. 查看修改
git status
git diff

# 5. 提交修改
git add SMD/src/shadow_distillation_loss.py
git commit -m "fix: 修复 KL 散度计算错误

- 将绝对值差异改为真正的 Forward KL
- 添加 clamp(min=0) 防止负值
- 更新相关注释

Fixes #1"

# 6. 推送到 GitHub
git push origin bugfix/fix-kl-divergence
```

### Step 3: 在 GitHub 上创建 Pull Request

1. **打开 GitHub 仓库页面**
   - https://github.com/huthvincent/RL-Post-Training-with-KV-Cache-Compression

2. **点击 "Compare & pull request" 按钮**
   - 推送后 GitHub 会自动显示这个按钮

3. **填写 PR 信息**

```markdown
## PR 标题
fix: 修复 KL 散度计算错误 (#1)

## 描述
### 问题
Track-2 的 KL 蒸馏损失使用了绝对值差异而非真正的 KL 散度。

### 解决方案
- 将 `(dense_lp - shadow_target_lp).abs()` 改为真正的 Forward KL
- 使用 `shadow_target_lp - dense_lp` 并 clamp 到非负值

### 测试
- [ ] 本地运行 `python baselines/test_baselines.py` 通过
- [ ] 在 TL;DR 数据集上验证训练收敛

### 相关 Issue
Closes #1
```

4. **选择 Reviewers（审查者）**
   - 在右侧 "Reviewers" 中添加协作者

5. **点击 "Create pull request"**

### Step 4: 进行 Code Review

#### 作为审查者 (Reviewer)

1. **打开 PR 页面**
2. **点击 "Files changed" 标签**
3. **对具体代码行添加评论**
   - 将鼠标悬停在行号上，点击 "+" 按钮
   - 写下你的评论
   - 可以选择 "Add single comment" 或 "Start a review"

4. **提交审查结果**
   - 点击右上角 "Review changes"
   - 选择：
     - **Comment**: 只是评论，不阻止合并
     - **Approve**: 批准合并
     - **Request changes**: 请求修改，阻止合并

#### 审查评论示例

```markdown
# 在代码行上的评论

## 建议修改
```suggestion
kl_per_token = (shadow_target_lp - dense_lp).clamp(min=0)
```

## 问题
这里的 clamp 是否会导致梯度消失？建议添加一个小的 epsilon。

## 赞同
👍 这个修复很好，数学上正确了。
```

### Step 5: 根据 Review 修改代码

```bash
# 在原分支上继续修改
git checkout bugfix/fix-kl-divergence

# 修改代码
vim SMD/src/shadow_distillation_loss.py

# 提交修改
git add .
git commit -m "fix: 根据 review 意见添加 epsilon 防止梯度消失"

# 推送更新（PR 会自动更新）
git push origin bugfix/fix-kl-divergence
```

### Step 6: 合并 PR

当所有审查者批准后：

1. **点击 "Merge pull request"**
2. **选择合并方式**：
   - **Create a merge commit**: 保留所有提交历史
   - **Squash and merge**: 将所有提交压缩为一个（推荐）
   - **Rebase and merge**: 变基合并

3. **删除分支**（可选）
   - 合并后 GitHub 会提示删除分支

### Step 7: 同步到其他服务器

```bash
# 在 Server B 上
git checkout main
git pull origin main

# 删除本地的旧分支（如果有）
git branch -d bugfix/fix-kl-divergence
```

---

## 📋 RLKV 项目的具体工作流示例

### 场景：修复 CODE_REVIEW_ISSUES.md 中的问题

假设你要修复 Issue #1（KL散度计算错误）：

#### Server A（主开发）

```bash
# 1. 创建分支
git checkout -b bugfix/issue-1-kl-divergence

# 2. 修改文件
cat > /tmp/fix.py << 'EOF'
# 在 SMD/src/shadow_distillation_loss.py 中
# 找到约 170 行的代码，修改为：

# 原代码:
# kl_per_token = (dense_lp - shadow_target_lp).abs()

# 新代码:
kl_per_token = (shadow_target_lp - dense_lp).clamp(min=0)
EOF

# 3. 提交并推送
git add SMD/src/shadow_distillation_loss.py
git commit -m "fix(loss): 修复 KL 散度计算 - 使用 Forward KL

- 将绝对值差异改为 Forward KL: shadow_target_lp - dense_lp
- 添加 clamp(min=0) 确保非负
- 符合 D_KL(π_shadow || π_dense) 的数学定义

Fixes #1"
git push origin bugfix/issue-1-kl-divergence
```

#### GitHub 上创建 PR

```markdown
## fix(loss): 修复 KL 散度计算错误

### 变更内容
修复 `shadow_distillation_loss.py` 中 Track-2 KL 蒸馏损失的计算错误。

### 问题描述
原代码使用 `|log π_dense - log π_sparse|`，这不是真正的 KL 散度。

### 解决方案
改为 Forward KL: `D_KL(π_shadow || π_dense) = E_shadow[log π_shadow - log π_dense]`

### 代码变更
```diff
- kl_per_token = (dense_lp - shadow_target_lp).abs()
+ kl_per_token = (shadow_target_lp - dense_lp).clamp(min=0)
```

### 测试清单
- [ ] 单元测试通过
- [ ] TL;DR 训练收敛正常
- [ ] GSM8K 准确率无下降

### 相关 Issue
Closes #1
```

#### Server B（审查和测试）

```bash
# 1. 获取 PR 分支
git fetch origin bugfix/issue-1-kl-divergence
git checkout bugfix/issue-1-kl-divergence

# 2. 运行测试
python baselines/test_baselines.py

# 3. 运行训练验证
python SMD/experiments/qwen3_0.6b/run_grpo_training.py \
    --dataset tldr --method smd --num-rollouts 50 \
    --output-dir /tmp/test_kl_fix

# 4. 如果测试通过，在 GitHub 上 Approve PR
```

---

## 🏷️ 分支命名规范

| 前缀 | 用途 | 示例 |
|:---|:---|:---|
| `feature/` | 新功能 | `feature/add-r-kv-strategy` |
| `bugfix/` | Bug 修复 | `bugfix/fix-kl-divergence` |
| `hotfix/` | 紧急修复 | `hotfix/memory-leak` |
| `docs/` | 文档更新 | `docs/update-readme` |
| `refactor/` | 代码重构 | `refactor/extract-training-utils` |
| `test/` | 测试相关 | `test/add-shadow-mask-tests` |

---

## 📝 Commit Message 规范

```
<type>(<scope>): <subject>

<body>

<footer>
```

### Type 类型
- `feat`: 新功能
- `fix`: Bug 修复
- `docs`: 文档
- `style`: 格式（不影响代码运行）
- `refactor`: 重构
- `test`: 测试
- `chore`: 构建/工具

### 示例

```bash
git commit -m "fix(loss): 修复 KL 散度计算错误

将绝对值差异改为真正的 Forward KL 散度。
原实现 |log π_dense - log π_sparse| 在数学上不正确。

Fixes #1"
```

---

## 🔗 与 Issues 联动

### 在 Commit 中关联 Issue

```bash
# 引用 Issue（不关闭）
git commit -m "fix: 部分修复 KL 计算 (ref #1)"

# 关闭 Issue
git commit -m "fix: 修复 KL 计算 (closes #1)"
git commit -m "fix: 修复 KL 计算 (fixes #1)"
```

### 在 PR 中关联 Issue

在 PR 描述中写：
```markdown
Closes #1
Fixes #2, #3
Related to #4
```

---

## 🚀 快速参考卡片

### 日常开发流程

```bash
# 1. 更新 main
git checkout main && git pull

# 2. 创建分支
git checkout -b feature/xxx

# 3. 开发 & 提交
git add . && git commit -m "feat: xxx"

# 4. 推送 & 创建 PR
git push origin feature/xxx
# 然后在 GitHub 上创建 PR

# 5. Review 后合并
# 在 GitHub 上点击 Merge

# 6. 清理
git checkout main && git pull
git branch -d feature/xxx
```

### 常用 Git 命令

```bash
# 查看状态
git status

# 查看分支
git branch -a

# 切换分支
git checkout <branch>

# 查看提交历史
git log --oneline -10

# 查看远程 PR 分支
git fetch origin pull/<PR_NUMBER>/head:<LOCAL_BRANCH>

# 撤销未提交的修改
git checkout -- <file>

# 撤销最后一次提交（保留修改）
git reset --soft HEAD~1
```

---

## ❓ 常见问题

### Q: PR 冲突怎么办？

```bash
# 1. 更新 main
git checkout main && git pull

# 2. 切回你的分支
git checkout your-branch

# 3. 合并 main（或 rebase）
git merge main
# 或
git rebase main

# 4. 解决冲突
# 编辑冲突文件，然后
git add .
git commit -m "resolve conflicts"

# 5. 推送
git push origin your-branch
```

### Q: 如何撤销已推送的 commit？

```bash
# 创建一个新的 commit 来撤销
git revert <commit-hash>
git push
```

### Q: 如何修改最后一次 commit message？

```bash
git commit --amend -m "新的 commit message"
git push --force  # 注意：只在个人分支上使用 force push
```

---

*此指南专为 RLKV 项目的多服务器协作开发编写*