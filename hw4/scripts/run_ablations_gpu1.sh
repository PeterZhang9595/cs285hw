#!/usr/bin/env bash
# CS285 HW4 — GRPO hyperparameter ablation study on format_copy (all sequential on GPU1).
# Baseline = default format_copy + GRPO (Required Run 1). Each `run` overrides exactly
# the hyperparameters listed after its tag (argparse: last occurrence wins).
set -uo pipefail

cd "$(dirname "$0")/.."          # repo root
mkdir -p runs

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
GPU=1

BASE=(uv run --no-sync python -m hw4.train \
  --task format_copy --algo grpo --steps 51 \
  --batch_size 8 --group_size 6 \
  --min_new_tokens 1 --max_new_tokens 24 --lr 3e-5 \
  --ppo_epochs 2 --minibatch_size 8 --grad_accum_steps 6 \
  --clip_eps 0.2 --kl_coef 0.05 --max_grad_norm 0.5 \
  --wandb_enabled --wandb_project llm-rl-hw4 \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 6 \
  --eval_interval 50 --save_interval 50 --warmup_steps 10)

run() {                          # $1=tag, rest = overrides
  local tag="$1"; shift
  echo "=== ablation: $tag ==="
  CUDA_VISIBLE_DEVICES=$GPU "${BASE[@]}" \
    --output_dir "runs/ablation_fc_grpo_$tag" \
    --wandb_name "fc_grpo_ablation_$tag" "$@" \
    > "runs/log_ablation_$tag.log" 2>&1
  echo "  $tag exit=$?"
}

# ---- Group 0: baseline (seed 0) + seed replicates ----
run base
run base_s1        --seed 1
run base_s2        --seed 2

# ---- Group 1: ppo_epochs (default 2) ----
run ppo1           --ppo_epochs 1
run ppo4           --ppo_epochs 4
run ppo4_s1        --ppo_epochs 4 --seed 1
run ppo4_s2        --ppo_epochs 4 --seed 2
run ppo8           --ppo_epochs 8

# ---- Group 2: kl_coef (default 0.05) ----
run kl0.0          --kl_coef 0.0
run kl0.0_s1       --kl_coef 0.0 --seed 1
run kl0.0_s2       --kl_coef 0.0 --seed 2
run kl0.01         --kl_coef 0.01
run kl0.2          --kl_coef 0.2
run kl0.5          --kl_coef 0.5

# ---- Group 3: clip_eps (default 0.2) ----
run clip0.05       --clip_eps 0.05
run clip0.1        --clip_eps 0.1
run clip0.4        --clip_eps 0.4

# ---- Group 4: minibatch_size x grad_accum_steps (effective batch fixed at 48) ----
run mb4_ga12       --minibatch_size 4  --grad_accum_steps 12
run mb12_ga4       --minibatch_size 12 --grad_accum_steps 4
run mb16_ga3       --minibatch_size 16 --grad_accum_steps 3
run mb24_ga2       --minibatch_size 24 --grad_accum_steps 2
run mb48_ga1       --minibatch_size 48 --grad_accum_steps 1

# ---- Group 5: grad_accum_steps = 1 (effective batch = 8) ----
run mb8_ga1        --grad_accum_steps 1

# ---- Group 6: learning rate (default 3e-5) ----
run lr1e-5         --lr 1e-5
run lr1e-4         --lr 1e-4
run lr3e-4         --lr 3e-4

# ---- Group 7: advantage normalization ----
run norm_adv       --normalize_advantages

# ---- Group 8: adv_clip (default 5) ----
run advclip1       --adv_clip 1
run advclip20      --adv_clip 20

# ---- Group 9: group_size (default 6) ----
run group2         --group_size 2
run group4         --group_size 4
run group8         --group_size 8

# ---- Group 10: sampling temperature (default 0.8) ----
run temp0.5        --temperature 0.5
run temp1.2        --temperature 1.2

echo
echo "All ablations finished. Aggregating..."
uv run --no-sync python scripts/summarize_ablations.py
