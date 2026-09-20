# CS 285/185 HW4 — LLM RL Report

Project: `llm-rl-hw4` — https://wandb.ai/3621048471-peking-university/llm-rl-hw4

All experiments were run locally on A100-40GB GPUs with LoRA (`r=16`, `alpha=32`) on
`Qwen/Qwen2.5-Math-1.5B-Instruct`, using the provided single-GPU training entry point
(`uv run --no-sync python -m hw4.train`). The reference policy is the base model obtained via
`disable_adapter()`.

## Run summary

| Run | Task / algorithm | Steps | Final eval (main metric) | Notes |
|---|---|---|---|---|
| 1 | Format Copy + GRPO | 51 | exact match **1.00** | required |
| 2 | Format Copy + GR-REINFORCE | 51 | exact match **1.00** | required |
| 3 | Math Hard + GR-REINFORCE | 201 | boxed exact match **0.277** (relaxed 0.311) | required |
| 4 | Math Hard + GRPO | 501 | boxed exact match **0.375** (relaxed 0.379) | required |
| — | GRPO hyperparameter study on format copy | 34 runs | exact match 1.00 (saturated) | Section 4 |

The two format-copy runs reach the maximum reward (1.3) and 100% eval exact match within a few
minutes. The math-hard runs are the interesting ones (Section 3). Math-hard baseline before any
RL update is 0.2266 boxed exact match (train reward ≈ 0.22), matching the value stated in the
assignment (~0.23).

---

## 1. Approximate KL

We use the sampled-token estimator

```
KL_hat = E_{a ~ p_new}[ exp(Δ) − Δ − 1 ],   Δ = log p_ref(a) − log p_new(a),
```

computed over the sampled completion tokens (masked mean). With `Δ = log(p_ref(a)/p_new(a))` and
the sample drawn from `p_new`,

```
E_{a~p_new}[ exp(Δ) ] = Σ_a p_new(a) · p_ref(a)/p_new(a) = Σ_a p_ref(a) = 1.
```

Therefore

```
E[ exp(Δ) − Δ − 1 ] = 1 − E[Δ] − 1 = −E[Δ]
                     = E[ log p_new(a) − log p_ref(a) ]
                     = KL(p_new ‖ p_ref).
```

So `e^Δ − Δ − 1` is a valid (single-sample, Monte-Carlo) estimator of `KL(p_new ‖ p_ref)` that only
requires the log-probabilities of the **sampled** tokens.

Why the exact full-vocabulary KL would be far more expensive: the exact KL at one position is

```
KL(p_new(·|x) ‖ p_ref(·|x)) = Σ_{v=1}^{V} p_new(v) [ log p_new(v) − log p_ref(v) ],
```

which requires a full softmax over the vocabulary (`V ≈ 151k`) at **every** token position for
**both** policies. That means materializing `[B, L, V]` logits / log-softmax tensors for the policy
and the reference — `O(B·L·V)` compute and memory — whereas the sampled estimator only needs the
`[B, L]` log-probs of the already-sampled tokens. This is exactly why `compute_per_token_logprobs`
uses the fused `F.cross_entropy(..., reduction='none')` (log-softmax + gather of the target token)
instead of an explicit dense log-softmax + gather. The clamp of `Δ` to `[-20, 20]` is only for
numerical stability / variance control; it does not change the estimator's expectation.

---

## 2. Implementation

### 2.1 Order of implementation

I implemented the TODOs bottom-up so each later piece could be tested against a working earlier one:

1. `compute_per_token_logprobs` — next-token log-probs aligned to target tokens (`[B, L-1]`).
2. `build_completion_mask` — the `[B, L-1]` mask selecting completion (non-prompt, non-pad) tokens.
3. `approx_kl_from_logprobs` — the sampled KL estimator above.
4. `iter_minibatches` — lazy minibatch generator over a `RolloutBatch`.
5. `Reinforce.update` — single-pass on-policy policy-gradient update.
6. `GRPO.update` — PPO-style clipped surrogate with importance reweighting.
7. `compute_group_advantages` — group-relative normalization.
8. `maybe_normalize_advantages` — optional global z-score.

The single most instructive bug/confusion point was the **one-position alignment** between
per-token log-probs and tokens: logprob index `t` scores token `t+1`, so the completion mask has
to be built from `attention_mask[:, 1:]` and the prompt region must be zeroed up to (but **not
including**) column `prompt_input_len − 1`, which is the first generated token.

### 2.2 Non-trivial bugs found and fixed during review

These are conceptual/logical mistakes (typos and pure variable-name slips are omitted):

- **`compute_per_token_logprobs` used the wrong logit slice.** `out.logits[:, -1, :]` (last
  position, shape `[B, V]`) instead of `out.logits[:, :-1, :]` (`[B, L-1, V]`). This broke the
  next-token alignment and the shape contract; fixed to `[:, :-1, :]`.
- **`build_completion_mask` off-by-one.** Zeroed `[:, :prompt_input_len]` instead of
  `[:, :prompt_input_len - 1]`, silently **dropping the first generated token** from the loss/KL.
- **`Reinforce.update` used the wrong mask.** Passed `mb.attention_mask` (`[B, L]`) where
  `mb.completion_mask` (`[B, L-1]`) was required — a shape mismatch that would also have averaged
  over prompt tokens instead of only completion tokens.
- **`GRPO.update` missing the loss sign.** `pg_loss = +mean(seq_obj)` instead of `-mean(seq_obj)`,
  which reverses the optimization direction (minimizing instead of maximizing the clipped
  surrogate). Also passed `torch.clamp(x, [-20, 20])` (a Python list) instead of two scalar
  arguments, which raises `TypeError`.
- **`compute_group_advantages` used true division.** `batch_size = rewards.shape[0] / group_size`
  produces a float, and `reshape` with a float shape fails; fixed to `//`.
- **`maybe_normalize_advantages` used the sample standard deviation.** `unbiased=True` instead of
  the required population std `unbiased=False`.
- **`iter_minibatches` did not actually shuffle or slice correctly.** It computed `indices` but
  never used them (so `shuffle` had no effect), yielded a raw `input_ids` tensor instead of a
  `RolloutBatch`, sliced only one field, and used `range(N // minibatch_size + 1)`, which emits an
  empty trailing minibatch when `N` is divisible by `minibatch_size`.

### 2.3 Conceptual questions / confusions resolved

A summary of the points I had to clarify (these drove several of the fixes above):

- **Padding and alignment.** What `pad_token_id` is for; whether the left-padding columns are
  counted inside `prompt_input_len` (they are — `prompt_input_len` is the padded prompt width, so
  completions start at token index `prompt_input_len` for every row); and the meaning of
  "`mask[:, t]` is 1 iff token `t+1` is a non-pad completion token". Also how to vectorize the
  mask (slice `attention_mask[:, 1:]`, then zero the prompt columns).
- **Roles of the three log-prob tensors.** Whether the two inputs to `approx_kl_from_logprobs` are
  already computed by the new and reference policies (yes); where `old_logprobs` is updated in
  GRPO (nowhere — it is frozen at rollout time and only used as the importance-ratio denominator);
  and the difference between `old_logprobs` (behavior policy, ratio denominator) and
  `ref_logprobs` (frozen base model, KL reference).
- **Loss/entropy/clipfrac semantics.** Why the logged GRPO loss is ≈ 0 (group-normalized
  advantages have exactly zero global mean and the logged loss is a mean over minibatches — a
  logging artifact, not a training failure; per-minibatch losses are `O(0.1–1)`); whether "entropy"
  here is a loss term (it is not — logging only) and why `entropy = −mean(log p)` is numerically the
  negative of the sequence log-prob; what `clipfrac` measures; and why `adv` is `detach()`-ed
  (it is a constant coefficient of the policy gradient).
- **Batch/minibatch mechanics.** Why `minibatch_size`/`grad_accum_steps` still matter for
  single-pass REINFORCE (they control splitting/accumulation, not data reuse); what `yield` and
  the `generator`/`device` arguments of `iter_minibatches` mean; and how the effective batch relates
  to the accumulation group.
- **Numerical artifacts.** Why the sampled KL can be slightly negative (`≈ −1e-6`) even though
  `e^Δ − Δ − 1 ≥ 0` — a bf16 catastrophic-cancellation effect when `Δ` is tiny; harmless.
- **Memory.** The math-hard runs initially OOM-ed because computing `old_logprobs`/`ref_logprobs`
  for the whole rollout at once materializes `[N=64, L≈868, V≈151k]` logits (> 40 GB). Resolved by
  chunking the batch inside `compute_per_token_logprobs` (Section 2.4).
- **Tooling.** The uv bytecode-compilation `EMFILE` ("too many open files") caused by
  `UV_COMPILE_BYTECODE=1` on a many-core machine; and why format-copy converges in minutes while
  math-hard needs hours.

### 2.4 Memory fix: chunked per-token log-probs

The provided sampler computes `old_logprobs` and `ref_logprobs` for the entire rollout
(`N = batch_size × group_size = 64` sequences, up to `L ≈ 868`) in one forward pass. On a 40 GB
A100 the resulting `[64, L, 151646]` bf16 logits plus the internal `log_softmax` of
`F.cross_entropy` exceed memory (it tried to allocate an extra 16.8 GB). I made a minimal change to
`compute_per_token_logprobs`: process the batch in chunks of `LOGPROB_CHUNK_SIZE = 8`, compute the
per-token log-probs per chunk, and concatenate. Per-token log-probs are independent, so the result
is numerically identical; only the peak memory changes (≈ 5 GB per chunk instead of > 40 GB). This
fixed both math-hard runs without changing any training hyperparameter.

---

## 3. GR-REINFORCE vs. GRPO on math-hard

The two math commands were chosen to be directly comparable: **identical** rollout batch
(8 prompts × 8 = 64 completions), `minibatch_size = 8`, `grad_accum_steps = 8`, `lr = 3e-5`,
`kl_coef = 0.05`, same sampling (`temperature 0.8`, `top_p 0.95`, `max_new_tokens 512`). The main
algorithmic difference is that **GRPO reuses each rollout for `ppo_epochs = 2`** with a PPO-clipped
importance ratio, while **GR-REINFORCE is single-pass on-policy**. GRPO also runs 501 steps vs. 201.

Reward and eval over the first 200 iterations:

| step | REINFORCE reward | REINFORCE boxedEM | REINFORCE KL | GRPO reward | GRPO boxedEM | GRPO KL |
|---|---|---|---|---|---|---|
| 0 | 0.217 | 0.2266 | 2.6e-6 | 0.217 | 0.2266 | ~0 |
| 50 | 0.397 | — | 6.8e-6 | 0.450 | — | 9.5e-4 |
| 100 | 0.312 | 0.2285 (@99) | 1.5e-4 | 0.370 | 0.2559 (@99) | 8.8e-4 |
| 150 | 0.191 | — | 5.7e-4 | 0.269 | — | 7.7e-3 |
| 200 | 0.488 | 0.2715 (@199) | 2.0e-3 | 0.631 | 0.3262 (@199) | 1.7e-2 |

Mean training reward: REINFORCE `0.281` (0–50), `0.288` (0–100), `0.289` (100–200) — essentially
flat; GRPO `0.282` (0–50), `0.298` (0–100), `0.420` (100–200) — clearly rising.

**Interpretation.** Over the first 200 iterations GRPO improves noticeably faster than
GR-REINFORCE (eval 0.326 vs. 0.272 at step 199; mean reward 0.42 vs. 0.29 over steps 100–200). Both
start from the same baseline (0.2266) and see the same number of *rollouts* per step, so the gap is
attributable to **sample reuse**: GRPO performs two epochs of updates per rollout (and the clipped
objective stabilizes those extra updates), whereas REINFORCE takes a single, high-variance
policy-gradient step per rollout. This is exactly why the commands are interesting: they isolate
the value of off-policy reuse under an otherwise controlled setup. GRPO also ends much higher
(boxedEM 0.375 vs. 0.277), though at the cost of ~5× the wall-clock (501 vs. 201 steps).

Caveats: math-hard train reward is noisy step-to-step (expected; the assignment notes this), and
GRPO showed late-training drift — KL grew from ~0 to ≈ 0.42 and the token entropy rose to ≈ 3.9
around steps 375–425, after which the reward became noisier. The eval curve stayed smooth and
plateaued at ≈ 0.36–0.375, so the run remained usable.

---

## 4. GRPO hyperparameter study on format copy

34 runs total: the baseline plus one-factor-at-a-time sweeps over `ppo_epochs`, `kl_coef`,
`clip_eps`, `minibatch_size × grad_accum_steps`, learning rate, `normalize_advantages`, `adv_clip`,
`group_size`, and `temperature`, plus seed replicates (`seed ∈ {0,1,2}`) for the baseline, `ppo4`,
and `kl0.0`. All runs use the default format-copy + GRPO command (51 steps) unless overridden.

**Important:** every run reaches final eval exact match = **1.00** — format copy is easy enough
that the eval metric saturates. Ablations must therefore be judged by *stability* metrics (KL,
clip-fraction, entropy), *convergence speed* (first step with reward ≥ 1.0), and *cost*.

### Selected results

| run | final reward | best | steps→reward≥1.0 | KL mean | KL max | clipfrac max | entropy final | wall (min) |
|---|---|---|---|---|---|---|---|---|
| base | 1.30 | 1.3 | 8 | 0.21 | 0.34 | 0.031 | 0.14 | 4.2 |
| ppo1 | 1.19 | 1.3 | 11 | 0.20 | 0.35 | 0.005 | 0.36 | 2.8 |
| ppo4 | 1.22 | 1.3 | 6 | 0.33 | 1.27 | 0.057 | 1.25 | 6.8 |
| ppo8 | 1.27 | 1.3 | 4 | 0.46 | 1.50 | 0.14 | 0.53 | 12.3 |
| kl0.0 | 1.30 | 1.3 | 8 | 0.22 | 0.33 | 0.029 | 0.22 | 4.1 |
| kl0.2 | 1.00 | 1.3 | 8 | 0.23 | 0.39 | 0.023 | 1.17 | 4.3 |
| kl0.5 | 1.08 | 1.25 | 12 | 0.17 | 0.29 | 0.022 | 0.67 | 4.5 |
| clip0.05 | 1.25 | 1.3 | 8 | 0.21 | 0.32 | 0.16 | 0.46 | 4.3 |
| clip0.4 | 1.30 | 1.3 | 8 | 0.21 | 0.33 | 0.021 | 0.16 | 4.1 |
| mb48_ga1 | 1.27 | 1.3 | 8 | 0.22 | 0.51 | 0.032 | 0.44 | 4.1 |
| mb8_ga1 | **1.03** | 1.3 | 5 | **0.50** | **1.41** | 0.11 | **3.39** | 4.3 |
| lr1e-5 | 1.22 | 1.3 | 15 | 0.19 | 0.34 | 0.018 | 0.17 | 4.1 |
| lr1e-4 | 1.25 | 1.3 | 4 | 0.28 | 0.59 | 0.070 | 1.01 | 4.2 |
| lr3e-4 | **0.85** | 1.3 | 3 | **0.47** | **3.07** | 0.085 | **5.22** | 4.1 |
| norm_adv | **1.30** | 1.3 | 8 | 0.23 | 0.34 | 0.026 | **0.063** | **3.8** |
| group2 | 1.29 | 1.3 | 16 | 0.21 | 0.78 | 0.008 | 0.13 | 1.9 |
| group8 | 1.22 | 1.3 | 7 | 0.23 | 0.45 | 0.052 | 0.86 | 5.1 |
| temp1.2 | 1.27 | 1.29 | 23 | 0.27 | 0.62 | 0.021 | 0.35 | 4.4 |

### Which hyperparameters mattered most

1. **Learning rate (largest effect).** `lr = 3e-4` is clearly too aggressive: KL explodes to
   **3.07**, entropy to **5.22**, and the final reward drops to **0.85**. `lr = 1e-5` is too slow
   (converges only by step 15). The sweet spot is the default `3e-5` to `1e-4`.
2. **`grad_accum_steps = 1` with `minibatch_size = 8`** (effective batch 8, i.e. many optimizer
   updates per rollout) is the second clear failure: KL max **1.41**, entropy **3.39**, final
   reward **1.03**. Note that `grad_accum_steps = 1` with the **full** minibatch (`mb48_ga1`) is
   perfectly stable (KL max 0.51) — the problem is update *frequency*, not accumulation per se.
3. **`ppo_epochs`.** More epochs reuse each rollout more aggressively: convergence speeds up
   (steps→1.0: 11 → 8 → 6 → 4 for 1/2/4/8) but off-policy drift and cost grow (KL max 0.35 → 1.5,
   clipfrac 0.005 → 0.14, wall 2.8 → 12.3 min).
4. **`kl_coef`.** The task needs very little KL regularization: 0.0/0.01 behave like the default,
   while **larger values hurt** (`kl0.2` ends at reward 1.00; `kl0.5` converges slower and caps at
   1.25). A too-strong trust region just limits learning here.
5. **`temperature`.** Raising to 1.2 slows convergence dramatically (steps→1.0 = 23) because the
   reward signal becomes noisier; 0.5 is comparable to the default.
6. **`group_size`.** Larger groups give a better baseline estimate and faster convergence
   (steps→1.0: 16/9/8/7 for 2/4/6/8) but cost more; `group2` is the cheapest but slowest to
   converge and noisiest.
7. **Little effect:** `clip_eps` (tighter clipping raises clipfrac but barely changes reward),
   `adv_clip` (advantages are group-normalized and rarely large), and `minibatch_size`/`grad_accum`
   when the effective batch is held fixed.

### Competitive or better than the default

`normalize_advantages = True` is competitive with, and arguably slightly better than, the default:
final reward 1.30, the **lowest final entropy** (0.063), and the **shortest wall time** (3.8 min).
`clip_eps = 0.4` and `mb48_ga1` are also on par with the default while being stable.

### Seed variance

| config | final reward | KL max |
|---|---|---|
| base | 1.235 ± 0.047 | 0.418 ± 0.093 |
| ppo4 | 1.263 ± 0.034 | 1.308 ± 0.275 |
| kl0.0 | 1.249 ± 0.070 | 0.552 ± 0.320 |

Seed-to-seed spread is substantial (especially for KL), so small differences between single runs
should not be over-interpreted. The two degradation findings above (`lr3e-4`, `mb8_ga1`) are far
outside this noise and are robust.

---

## 5. Qualitative behavior (math-hard)

Examples pulled from the W&B sampled-generation tables
(`samples/math_hard_prompt_completion_reward_breakdown`).

**Early (step 0, before RL).** The base policy produces well-formatted long chains of thought but
the wrong answer; it receives only the format bonus.

- Question (geometry): square `ABCD` inscribed in a circle, square `EFGH` with `E,F` on `CD` and
  `G,H` on the circle; area of `ABCD` is 1, find `10n + m` for the area of `EFGH = m/n`.
- Ground truth: `251`; the model derives `t = 1` and concludes `\boxed{11}`.
- Reward = 0.1 (contains `\boxed{}` only), advantage = **+2.65** (positive because it is the best
  in its group).

**Late (end of training).** The model solves the problem correctly and emits a well-formed boxed
answer.

- Question: `(log2 x)^4 + (log3 y)^4 + 8 = 8 (log2 x)(log3 y)`, compute `x^√2 + y^√2`.
- Ground truth: `13`; the model sets `a = log2 x`, `b = log3 y`, finds `a = b = √2`, and concludes
  `\boxed{13}`.
- Reward = 1.1 (`format` 0.1 + `exact_match` 1.0), predicted boxed value `13.0`.

**Observations.** (i) The model's completions are long (mean ≈ 330–470 tokens, often close to the
512-token cap), and several early samples are truncated mid-derivation — a natural source of
missing `\boxed{}` and thus zero reward. (ii) The late-training KL/entropy blow-up (Section 3) is
visible qualitatively as the policy becoming less confident: entropy rose to ≈ 3.9 while eval
plateaued, consistent with the policy drifting toward more exploratory (and occasionally
degenerate) long generations after the trust region was effectively left. (iii) Advantages are
frequently zero within a group when all candidates are wrong, which is why group-relative
normalization is essential for this sparse-reward task.

---

## Reproducing

```bash
# required runs
./scripts/run_math_hard_gpu01.sh           # math-hard reinforce (GPU0) + grpo (GPU1)
# ablations (34 runs, GPU1) + auto summary
./scripts/run_ablations_gpu1.sh
uv run --no-sync python scripts/summarize_ablations.py   # runs/ablation_summary.csv
```

Run artifacts live under `runs/local_math_hard_reinforce/`, `runs/local_math_hard_grpo/`,
`runs/local_format_copy_grpo_gpu3/`, `runs/local_format_copy_reinforce_gpu0/`, and
`runs/ablation_fc_grpo_*/`, each containing `config.json`, `metrics.jsonl`, and checkpoints.
