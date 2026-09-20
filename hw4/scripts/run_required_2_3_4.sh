#!/usr/bin/env bash
# CS285 HW4 Required Runs 2/3/4 — parallel on GPU 0/1/2
set -uo pipefail

cd "$(dirname "$0")/.."          # repo root
mkdir -p runs                    # log dir

TRAIN=(uv run --no-sync python -m hw4.train)

run_job() {                      # $1=gpu  $2=logfile   rest=train args
  local gpu="$1"; shift
  local log="$1"; shift
  CUDA_VISIBLE_DEVICES="$gpu" "${TRAIN[@]}" "$@" > "$log" 2>&1
}

# ---- Run 2: Format Copy + GR-REINFORCE -> GPU0 ----
run_job 0 runs/log_gpu0_fc_reinforce.log \
  --task format_copy --algo reinforce \
  --output_dir runs/local_format_copy_reinforce_gpu0 \
  --steps 51 --batch_size 8 --group_size 6 \
  --min_new_tokens 1 --max_new_tokens 24 \
  --lr 3e-5 --minibatch_size 8 --grad_accum_steps 6 \
  --kl_coef 0.05 --max_grad_norm 0.5 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name format_copy_reinforce \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 6 \
  --eval_interval 50 --save_interval 50 --warmup_steps 10 &
PID0=$!

# ---- Run 3: Math Hard + GR-REINFORCE -> GPU1 ----
run_job 1 runs/log_gpu1_mh_reinforce.log \
  --task math_hard --algo reinforce \
  --output_dir runs/local_math_hard_reinforce_gpu1 \
  --steps 201 --batch_size 8 --group_size 8 \
  --min_new_tokens 8 --max_new_tokens 512 --max_prompt_tokens 512 \
  --temperature 0.8 --top_p 0.95 \
  --lr 3e-5 --minibatch_size 8 --grad_accum_steps 8 \
  --kl_coef 0.05 --max_grad_norm 0.5 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name math_hard_reinforce \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 8 \
  --cuda_empty_cache_interval 50 --eval_interval 100 --save_interval 100 &
PID1=$!

# ---- Run 4: Math Hard + GRPO -> GPU2 ----
run_job 2 runs/log_gpu2_mh_grpo.log \
  --task math_hard --algo grpo \
  --output_dir runs/local_math_hard_grpo_gpu2 \
  --steps 501 --batch_size 8 --group_size 8 \
  --min_new_tokens 8 --max_new_tokens 512 --max_prompt_tokens 512 \
  --temperature 0.8 --top_p 0.95 \
  --lr 3e-5 --ppo_epochs 2 --minibatch_size 8 --grad_accum_steps 8 \
  --clip_eps 0.2 --kl_coef 0.05 --max_grad_norm 0.5 \
  --wandb_enabled --wandb_project llm-rl-hw4 --wandb_name math_hard_grpo \
  --sample_markdown_log_interval 1 --sample_log_interval 10 --sample_log_n 8 \
  --cuda_empty_cache_interval 50 --eval_interval 100 --save_interval 100 &
PID2=$!

echo "Launched: GPU0 pid=$PID0 | GPU1 pid=$PID1 | GPU2 pid=$PID2"
echo "Logs: runs/log_gpu{0,1,2}_*.log  (tail -f 查看)"

wait "$PID0"; RC0=$?
wait "$PID1"; RC1=$?
wait "$PID2"; RC2=$?
echo "Exit codes: gpu0=$RC0 gpu1=$RC1 gpu2=$RC2"
exit $(( RC0 | RC1 | RC2 ))
