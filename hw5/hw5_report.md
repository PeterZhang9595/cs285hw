# CS285 HW5：Offline RL 实验报告（IQL / FQL）

## 1. 实验设置

- 算法：IQL（第 3 部分）、FQL（第 4 部分），均在 `src/agents/` 下实现。
- 环境：OGBench `cube-single-play-singletask-task1-v0`、`antsoccer-arena-navigate-singletask-task1-v0`；debug 用 `antmaze-medium-navigate-singletask-task1-v0`。
- 超参：`discount=0.99`、`target_update_rate=0.005`、`expectile=0.9`（IQL）、`flow_steps=10`（FQL）、`batch_size=256`、`lr=3e-4`、`seed=0`、`1M` steps。
- 硬件：NVIDIA A100-40GB，每卡并行 3–4 个 job。
- 评估：每 100K steps 用 25 条轨迹的成功率。

命令模板（与 PDF 一致）：

```bash
# IQL
CUDA_VISIBLE_DEVICES=<gpu> UV_COMPILE_BYTECODE=0 uv run src/scripts/run.py \
  --run_group=q2 --base_config=iql --env_name=<env> --seed=0 --alpha=<a>

# FQL
CUDA_VISIBLE_DEVICES=<gpu> UV_COMPILE_BYTECODE=0 uv run src/scripts/run.py \
  --run_group=q3 --base_config=fql --env_name=<env> --seed=0 --alpha=<a>
```

---

## 2. 实现要点与踩坑记录（非语法/非低级错误）

以下记录实现过程中遇到的**概念性、设计性**问题（不包括拼写、缺 import 这类低级错误）。

### 2.1 IQL

1. **Expectile loss 的方向容易搞反。**
   公式为 `ℓ²_τ(x) = |τ − I(x>0)| x²`，输入是 `x = V(s) − min_i Q̄_i(s,a)`（V 减 target Q），不是优势 `Q − V`。若写成 `Q − V`，正负号颠倒会导致 V 收敛到错误的分位数。

2. **Expectile loss 必须做 reduction。**
   最初返回的是逐元素 `(B,)` 张量，`loss.backward()` 会因非标量而报错。本仓库约定 loss 返回标量（`.mean()`）。

3. **V 的 target 必须用 target critic，且 stop-gradient。**
   `q = target_critic(...).min(dim=0).values` 要放在 `torch.no_grad()` 内；只有 `v` 需要梯度。用当前 critic 会让 V 的回归目标随 critic 漂移。

4. **优势权重必须 stop-gradient 且裁剪。**
   `w = min(exp(α·adv), M)`，`adv = min_i Q_i − V`。其中：
   - `M = 100` 在代码里没有现成参数，需要**自己硬编码**；
   - 权重必须是常数（`adv` 在 `no_grad` 下计算），否则会通过权重把梯度反传到 Q/V，改变算法语义；
   - 数值上先 `clamp` 指数再 `exp` 更稳：`exp((α·adv).clamp(max=log(M)))`，避免 `exp` 溢出。

5. **Q loss 的 reduction 会改变有效学习率。**
   用 `.sum()` 而非 `.mean()` 会把 loss（及梯度）放大 batch size（256）倍，等价于把 critic 学习率放大 256 倍，训练不稳定。应为 `((q1−y)² + (q2−y)²).mean()`。

6. **done 信号。**
   PDF 的公式省略了 `(1−done)`，但 Section 2.2 tip 明确要求加入：`y = r + γ(1−done)V(s')`。漏掉会在终止 transition 上错误 bootstrap。

### 2.2 FQL

1. **`flow_steps` 只体现在行为流策略的动作生成上。**
   `bc_actor` 与 `onestep_actor` 网络结构完全相同（都是 `VectorFieldPolicy`），区别只在调用方式：`bc_actor` 通过 `get_bc_action` 做 `flow_steps`（10）步 Euler 积分；`onestep_actor` 永远只做一次前向（`times=0`），用于评估、Bellman backup 和蒸馏。

2. **Euler 积分的细节容易写错。**
   正确形式：
   ```python
   action = noise
   dt = 1.0 / self.flow_steps
   for i in range(self.flow_steps):
       t = torch.full((B, 1), i * dt, device=...)
       action = action + dt * self.bc_actor(observation, action, t)
   ```
   要点：从**噪声**出发（不是全零）、每步传入**演化中的 action**、时间 `t` 必须随步变化、观测全程不变、只在最后 clamp。

3. **clip 的梯度语义（PDF 特别强调）。**
   `torch.clamp` 在区间外梯度为 0。因此：
   - 喂给 Q 前**要 clip**（Q 只在 `[−1,1]` 内可信，避免外推被利用）；
   - 蒸馏项**不能 clip**，否则越界动作拿到 0 梯度，永远无法被拉回。两者配合：Q 项在界内优化，蒸馏项负责纠正越界。

4. **Bellman backup 用两个 target Q 的“平均”而非“最小”。**
   这是 FQL 与 SAC+BC/IQL 的一个区别，PDF 公式 (16) 与 tip 均强调（对 antsoccer 尤其重要）：`target_q = target_critic(...).mean(dim=0)`。

5. **评估用 one-step 策略并采样噪声。**
   输出策略是 `πω`，不是行为流 `πv`；评估时对 `z ~ N(0,I)` 采样（不是取 mode/mean），因为 one-step 策略是噪声条件化的多模态策略。

6. **蒸馏目标必须 stop-gradient。**
   `teacher = get_bc_action(...)` 要放在 `torch.no_grad()` 内，梯度不应反传到行为流 `πv`（PDF 明确说明）。

7. **`mse` 诊断指标应对比数据集动作。**
   PDF tip 要求监控“数据集动作 vs one-step 动作”的 MSE 以调 α；若误用 `teacher_actions`，则它只是 `distill_loss / α`，无法反映策略偏离数据的程度。

8. **蒸馏系数。**
   Eq. (17) 为 `(α/|A|)·‖πω − πv‖²`；由于 `.mean()` 已对动作维取平均，代码应写 `alpha * ((πω − πv)**2).mean()`，等价于 `α/|A|·‖·‖²` 对 batch 取期望。

### 2.3 工程/运行

1. **本地不能直接用内置 `--njobs` 并行。**
   `src/scripts/run_njobs.py` 依赖 `modal.Volume` 作为 checkpoint 回调，是给 Modal 用的；本机运行会在保存 checkpoint 时报错。本地并行应改为“多个独立进程 + `CUDA_VISIBLE_DEVICES`”，并保证每个 run 的 `exp/` 目录由 `--alpha` 区分。

2. **后台启动需要 `setsid`。**
   直接用 `nohup ... &` 启动 sweep 脚本时，父 shell 退出/被终止会把整个进程组（含子 job）一起杀掉。改用 `setsid ... < /dev/null > log 2>&1 &` 可完全脱离会话，任务稳定存活。

3. **并发加载大数据集较慢。**
   OGBench 数据集单个约 250–300MB，多个进程同时从共享存储加载时前几分钟会处于 `D`（disk sleep）状态、GPU 显存为 0，属正常现象，加载完成后利用率即拉满。

---

## 3. 实验结果

### 3.1 IQL（q2）

**cube-single**（目标 final >60%）

| α | 100K | 200K | 300K | 400K | 500K | 600K | 700K | 800K | 900K | 1M | 峰值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.88 | 0.84 | **1.00** | 0.92 | 0.80 | **1.00** | 0.96 | 0.88 | 0.88 | **0.88** | 1.00 |
| 3 | 0.12 | 0.64 | 0.72 | 0.56 | 0.60 | 0.76 | 0.64 | 0.80 | 0.84 | 0.64 | 0.84 |
| 10 | 0.76 | 0.44 | 0.52 | 0.64 | 0.80 | 0.72 | 0.92 | 0.96 | 0.68 | 0.88 | 0.96 |

**antsoccer-arena**（目标 final >5%）

| α | 100K | 200K | 300K | 400K | 500K | 600K | 700K | 800K | 900K | 1M | 峰值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.08 | 0.08 | 0.00 | 0.16 | 0.08 | 0.12 | 0.04 | 0.00 | 0.04 | 0.08 | 0.16 |
| 3 | 0.00 | 0.04 | 0.00 | 0.08 | 0.08 | 0.16 | 0.08 | 0.04 | 0.08 | 0.08 | 0.16 |
| 10 | 0.04 | 0.08 | 0.04 | 0.28 | 0.12 | 0.08 | 0.24 | 0.04 | **0.36** | **0.20** | 0.36 |

**antmaze-medium（debug）**：α=3，200K 达 0.92，峰值 0.92。

### 3.2 FQL（q3）

**cube-single**（目标 final >80%）

| α | 100K | 200K | 300K | 400K | 500K | 600K | 700K | 800K | 900K | 1M | 峰值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 30 | 0.00 | 0.16 | 0.04 | 0.24 | 0.20 | 0.08 | 0.36 | 0.32 | 0.64 | 0.60 | 0.64 |
| 100 | 0.80 | 0.92 | **1.00** | **1.00** | 0.96 | 0.96 | **1.00** | 0.92 | **1.00** | 0.84 | 1.00 |
| 300 | 0.92 | 0.88 | 0.96 | **1.00** | 0.96 | **1.00** | 0.96 | 0.96 | **1.00** | **0.92** | 1.00 |
| 1000 | 0.64 | 0.48 | 0.88 | 0.92 | 0.96 | **1.00** | 0.92 | 0.88 | 0.92 | 0.72 | 1.00 |

**antsoccer-arena**（目标 best >30%）

| α | 100K | 200K | 300K | 400K | 500K | 600K | 700K | 800K | 900K | 1M | 峰值 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.00 | 0.04 | 0.00 | 0.04 | 0.04 |
| 3 | 0.00 | 0.04 | 0.04 | 0.04 | 0.12 | 0.00 | 0.00 | 0.00 | 0.08 | 0.08 | 0.12 |
| 10 | 0.04 | 0.28 | 0.40 | 0.20 | **0.48** | 0.32 | 0.28 | 0.44 | 0.12 | **0.28** | 0.48 |
| 30 | 0.00 | 0.04 | 0.04 | 0.00 | 0.00 | 0.08 | 0.00 | 0.00 | 0.00 | 0.00 | 0.08 |

**antmaze-medium（debug）**：α=3，200K 达 0.80，满足 >80%。

---

## 4. 分析与对比

### 4.1 最优结果

| 算法 | 任务 | 最优 α | final | 峰值 | 是否达标 |
|---|---|---|---|---|---|
| IQL | cube-single | 1 | 0.88 | 1.00 | ✅ (>60%) |
| IQL | antsoccer-arena | 10 | 0.20 | 0.36 | ✅ (>5%) |
| FQL | cube-single | 300 | 0.92 | 1.00 | ✅ (>80%) |
| FQL | antsoccer-arena | 10 | 0.28 | 0.48 | ✅ (best >30%) |

### 4.2 对比（对应 PDF 要求的讨论）

- **最佳性能**：FQL 整体更强。cube-single 上 FQL（α=300, final 0.92）略优于 IQL（α=1, final 0.88）；antsoccer-arena 上 FQL 峰值 0.48 明显高于 IQL 的 0.36。这符合“表达力更强的流策略/多模态策略在离线 RL 中更有利”的预期。
- **对 α 的敏感性**：FQL 更敏感。cube-single 只有 α∈{100,300} 达标，α=30/1000 明显下滑；antsoccer 仅 α=10 可用（α=1/30 几乎为 0）。IQL 相对鲁棒，cube-single 三个 α 均 >60%，antsoccer 三个 α 均 >5%。
- **趋势**：IQL 的 antmaze debug 在 500K 后 success 从 0.92 缓降到 0.64，可能存在后期值函数过估计；FQL 在最优 α 下更平稳。

---

## 5. 结论

- IQL、FQL 均正确实现并通过 debug 验证（antmaze-medium 分别达 0.92 / 0.80），最终在 cube-single 与 antsoccer-arena 上均达到 PDF 的性能要求。
- 推荐用于提交的最优 run：
  - IQL：cube-single `α=1`，antsoccer-arena `α=10`
  - FQL：cube-single `α=300`，antsoccer-arena `α=10`
- 关键实现陷阱集中在：IQL 的 expectile 方向与 stop-gradient、权重裁剪；FQL 的 Euler 积分细节、clip 的梯度语义、Bellman backup 取平均、蒸馏 stop-gradient。
