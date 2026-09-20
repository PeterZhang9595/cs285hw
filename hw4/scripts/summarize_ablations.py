#!/usr/bin/env python3
"""Aggregate GRPO format_copy ablation runs into a comparison table.

Reads runs/ablation_fc_grpo_<tag>/metrics.jsonl for every ablation run and
writes runs/ablation_summary.csv plus a printed table. Intended to be run after
scripts/run_ablations_gpu1.sh.
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics
from typing import Any, Dict, List, Optional

RUN_GLOB = "runs/ablation_fc_grpo_*/metrics.jsonl"
OUT_CSV = "runs/ablation_summary.csv"

EVAL_EM = "eval/format_copy_fraction_predicted_number_matches_target_integer_exactly"
REWARD = "rollout/mean_total_reward_across_all_completions_in_batch_and_groups"
KL = "train/approximate_kl_divergence_policy_vs_reference_mean_over_minibatches"
CLIPFRAC = "train/fraction_of_completion_tokens_where_ppo_ratio_was_clipped_mean_over_minibatches"
ENTROPY = "train/policy_token_entropy_mean_over_minibatches"
WALL = "train/wall_clock_seconds_for_this_training_iteration"

CONVERGE_REWARD = 1.0


def load(path: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def series(rows: List[Dict[str, Any]], key: str) -> List[float]:
    out = []
    for r in rows:
        v = r.get("metrics", {}).get(key)
        if isinstance(v, (int, float)):
            out.append(float(v))
    return out


def last(rows: List[Dict[str, Any]], key: str) -> Optional[float]:
    s = series(rows, key)
    return s[-1] if s else None


def steps_to_reward(rows: List[Dict[str, Any]], threshold: float) -> Optional[int]:
    for r in rows:
        v = r.get("metrics", {}).get(REWARD)
        if isinstance(v, (int, float)) and float(v) >= threshold:
            return int(r.get("step"))
    return None


def summarize(path: str) -> Dict[str, Any]:
    tag = os.path.basename(os.path.dirname(path)).replace("ablation_fc_grpo_", "")
    rows = load(path)
    kl = series(rows, KL)
    clip = series(rows, CLIPFRAC)
    reward = series(rows, REWARD)
    ent = series(rows, ENTROPY)
    wall = series(rows, WALL)
    n_train = sum(1 for r in rows if KL in r.get("metrics", {}))
    return {
        "tag": tag,
        "n_train_steps": n_train,
        "last_step": max((int(r["step"]) for r in rows if r.get("step") is not None), default=None),
        "final_eval_em": last(rows, EVAL_EM),
        "final_reward": reward[-1] if reward else None,
        "best_reward": max(reward) if reward else None,
        "steps_to_reward>=1.0": steps_to_reward(rows, CONVERGE_REWARD),
        "kl_mean": statistics.mean(kl) if kl else None,
        "kl_max": max(kl) if kl else None,
        "clipfrac_max": max(clip) if clip else None,
        "entropy_final": ent[-1] if ent else None,
        "wall_minutes": sum(wall) / 60.0 if wall else None,
    }


def fmt(v: Any) -> str:
    if v is None:
        return "-"
    if isinstance(v, float):
        return f"{v:.4g}"
    return str(v)


def main() -> None:
    paths = sorted(glob.glob(RUN_GLOB))
    if not paths:
        print(f"No ablation runs found matching {RUN_GLOB}")
        return

    summaries = [summarize(p) for p in paths]
    summaries.sort(key=lambda s: s["tag"])

    fields = list(summaries[0].keys())
    with open(OUT_CSV, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for s in summaries:
            writer.writerow(s)

    header = ["tag", "final_eval_em", "final_reward", "best_reward",
              "steps_to_reward>=1.0", "kl_mean", "kl_max", "clipfrac_max",
              "entropy_final", "wall_minutes"]
    widths = {h: max(len(h), *(len(fmt(s.get(h))) for s in summaries)) for h in header}
    line = " | ".join(h.ljust(widths[h]) for h in header)
    print(line)
    print("-" * len(line))
    for s in summaries:
        print(" | ".join(fmt(s.get(h)).ljust(widths[h]) for h in header))

    # Seed-replicate summary (mean +/- std) for configs that have _s1/_s2.
    def group_stats(prefix: str) -> None:
        members = [s for s in summaries if s["tag"] == prefix or s["tag"].startswith(prefix + "_s")]
        if len(members) < 2:
            return
        print(f"\nseed replicates for '{prefix}' (n={len(members)}):")
        for h in ["final_eval_em", "final_reward", "kl_max"]:
            vals = [s[h] for s in members if isinstance(s[h], (int, float))]
            if vals:
                mean = statistics.mean(vals)
                std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
                print(f"  {h}: {mean:.4g} +/- {std:.3g}")

    print()
    for prefix in ["base", "ppo4", "kl0.0"]:
        group_stats(prefix)

    print(f"\nWrote {OUT_CSV}")


if __name__ == "__main__":
    main()
