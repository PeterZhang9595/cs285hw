#!/usr/bin/env bash
set -uo pipefail
cd "$(dirname "$0")/.."
mkdir -p exp/logs

pids=()

launch() {
  local gpu=$1 alpha=$2 env=$3
  CUDA_VISIBLE_DEVICES=$gpu UV_COMPILE_BYTECODE=0 uv run src/scripts/run.py \
    --run_group=q3 --base_config=fql --env_name="$env" \
    --seed=0 --alpha="$alpha" \
    > "exp/logs/fql_${env}_a${alpha}.log" 2>&1 &
  pids+=($!)
}

for a in 30 100 300 1000; do
  launch 4 "$a" cube-single-play-singletask-task1-v0
done
for a in 1 3 10 30; do
  launch 7 "$a" antsoccer-arena-navigate-singletask-task1-v0
done

echo "Launched ${#pids[@]} jobs: ${pids[*]}"
wait "${pids[@]}"
echo "All FQL jobs finished."
