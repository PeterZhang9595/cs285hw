"""Generate figures for the Chinese HW3 report from local experiment logs."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report_assets"

RUNS = {
    "cartpole": ROOT / "exp/CartPole-v1_dqn_sd1_20260803_123046/log.csv",
    "lunarlander": ROOT / "exp/LunarLander-v2_dqn_sd1_20260803_124025/log.csv",
    "mspacman": ROOT / "exp/MsPacman_dqn_sd1_20260803_133132/log.csv",
    "inverted_fixed": ROOT / "exp/InvertedPendulum-v4_sac_sd1_20260803_155246/log.csv",
    "inverted_auto": ROOT / "exp/InvertedPendulum-v4_sac_autotune_sd1_20260803_162920/log.csv",
    "halfcheetah_fixed": ROOT / "exp/HalfCheetah-v4_sac_sd1_20260803_164009/log.csv",
    "halfcheetah_auto": ROOT / "exp/HalfCheetah-v4_sac_autotune_sd1_20260803_164635/log.csv",
    "hopper_single": ROOT / "exp/Hopper-v4_sac_singleq_sd1_20260803_164844/log.csv",
    "hopper_clip": ROOT / "exp/Hopper-v4_sac_clipq_sd1_20260803_175843/log.csv",
}


def read_series(path: Path, key: str) -> tuple[np.ndarray, np.ndarray]:
    steps: list[float] = []
    values: list[float] = []
    with path.open(newline="") as file:
        for row in csv.DictReader(file):
            if row.get(key):
                steps.append(float(row["step"]))
                values.append(float(row[key]))
    return np.asarray(steps), np.asarray(values)


def rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    if len(values) < window:
        return values.copy()
    kernel = np.ones(window, dtype=np.float64) / window
    valid = np.convolve(values, kernel, mode="valid")
    prefix = np.asarray(
        [values[: index + 1].mean() for index in range(window - 1)],
        dtype=np.float64,
    )
    return np.concatenate([prefix, valid])


def annotate_max(axis: plt.Axes, steps: np.ndarray, values: np.ndarray) -> None:
    index = int(np.argmax(values))
    axis.scatter([steps[index]], [values[index]], color="black", s=18, zorder=4)
    axis.annotate(
        f"max={values[index]:.1f}",
        (steps[index], values[index]),
        xytext=(6, 8),
        textcoords="offset points",
        fontsize=8,
    )


def finish(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=180, bbox_inches="tight")
    plt.close(fig)


plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams.update(
    {
        "figure.figsize": (7.2, 4.2),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 10,
    }
)


# DQN: CartPole.
steps, values = read_series(RUNS["cartpole"], "Eval_AverageReturn")
fig, ax = plt.subplots()
ax.plot(steps, values, marker="o", linewidth=2, label="Eval return")
annotate_max(ax, steps, values)
ax.set(title="DQN on CartPole-v1", xlabel="Environment steps", ylabel="Eval return")
ax.legend()
finish(fig, "dqn_cartpole_eval.png")


# Double DQN: LunarLander.
steps, values = read_series(RUNS["lunarlander"], "Eval_AverageReturn")
fig, ax = plt.subplots()
ax.plot(steps, values, linewidth=2, label="Eval return")
ax.axhline(200, color="gray", linestyle="--", linewidth=1, label="Return 200")
annotate_max(ax, steps, values)
ax.set(
    title="Double DQN on LunarLander-v2",
    xlabel="Environment steps",
    ylabel="Eval return",
)
ax.legend()
finish(fig, "dqn_lunarlander_eval.png")


# Double DQN: MsPacman train/eval comparison.
eval_steps, eval_values = read_series(RUNS["mspacman"], "Eval_AverageReturn")
train_steps, train_values = read_series(RUNS["mspacman"], "Train_EpisodeReturn")
fig, ax = plt.subplots()
ax.plot(
    train_steps,
    rolling_mean(train_values, 30),
    linewidth=1.6,
    label="Train return (30-episode mean)",
)
ax.plot(eval_steps, eval_values, linewidth=2, label="Eval return")
annotate_max(ax, eval_steps, eval_values)
ax.set(
    title="Double DQN on MsPacman",
    xlabel="Environment steps",
    ylabel="Episode return",
)
ax.legend()
finish(fig, "dqn_mspacman_returns.png")


# SAC sanity checks.
fixed_steps, fixed_eval = read_series(RUNS["inverted_fixed"], "Eval_AverageReturn")
auto_steps, auto_eval = read_series(RUNS["inverted_auto"], "Eval_AverageReturn")
fixed_ent_steps, fixed_ent = read_series(RUNS["inverted_fixed"], "entropy")
auto_ent_steps, auto_ent = read_series(RUNS["inverted_auto"], "entropy")
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)
axes[0].plot(fixed_steps, fixed_eval, marker="o", label="Fixed temperature")
axes[0].plot(auto_steps, auto_eval, marker="o", label="Auto temperature")
axes[0].axhline(1000, color="gray", linestyle="--", linewidth=1)
axes[0].set(ylabel="Eval return", title="SAC sanity check on InvertedPendulum-v4")
axes[0].legend()
axes[1].plot(fixed_ent_steps, fixed_ent, label="Fixed temperature")
axes[1].plot(auto_ent_steps, auto_ent, label="Auto temperature")
axes[1].set(xlabel="Environment steps", ylabel="Entropy")
axes[1].legend()
finish(fig, "sac_invertedpendulum_sanity.png")


# SAC fixed-temperature HalfCheetah result.
steps, values = read_series(RUNS["halfcheetah_fixed"], "Eval_AverageReturn")
fig, ax = plt.subplots()
ax.plot(steps, values, linewidth=2, label="Fixed temperature (0.1)")
ax.axhline(6000, color="gray", linestyle="--", linewidth=1, label="Return 6000")
annotate_max(ax, steps, values)
ax.set(
    title="SAC on HalfCheetah-v4",
    xlabel="Environment steps",
    ylabel="Eval return",
)
ax.legend()
finish(fig, "sac_halfcheetah_fixed_eval.png")


# Fixed versus automatically tuned temperature on HalfCheetah.
fixed_steps, fixed_eval = read_series(RUNS["halfcheetah_fixed"], "Eval_AverageReturn")
auto_steps, auto_eval = read_series(RUNS["halfcheetah_auto"], "Eval_AverageReturn")
temp_steps, temperature = read_series(RUNS["halfcheetah_auto"], "temperature")
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)
axes[0].plot(fixed_steps, fixed_eval, linewidth=2, label="Fixed temperature = 0.1")
axes[0].plot(auto_steps, auto_eval, linewidth=2, label="Auto-tuned temperature")
axes[0].set(title="Temperature tuning on HalfCheetah-v4", ylabel="Eval return")
axes[0].legend()
axes[1].plot(temp_steps, temperature, linewidth=2, label="Learned temperature")
axes[1].axhline(0.1, color="gray", linestyle="--", linewidth=1, label="Initial 0.1")
axes[1].set(xlabel="Environment steps", ylabel="Temperature (alpha)")
axes[1].legend()
finish(fig, "sac_halfcheetah_autotune.png")


# Hopper: single-Q versus clipped double-Q.
single_eval_steps, single_eval = read_series(RUNS["hopper_single"], "Eval_AverageReturn")
clip_eval_steps, clip_eval = read_series(RUNS["hopper_clip"], "Eval_AverageReturn")
single_q_steps, single_q = read_series(RUNS["hopper_single"], "q_values")
clip_q_steps, clip_q = read_series(RUNS["hopper_clip"], "q_values")
fig, axes = plt.subplots(2, 1, figsize=(7.2, 6.6), sharex=True)
axes[0].plot(single_eval_steps, single_eval, linewidth=2, label="Single-Q")
axes[0].plot(clip_eval_steps, clip_eval, linewidth=2, label="Clipped double-Q")
axes[0].axhline(1500, color="gray", linestyle="--", linewidth=1, label="Return 1500")
axes[0].set(title="SAC critic comparison on Hopper-v4", ylabel="Eval return")
axes[0].legend()
axes[1].plot(
    single_q_steps,
    rolling_mean(single_q, 10),
    linewidth=2,
    label="Single-Q (smoothed)",
)
axes[1].plot(
    clip_q_steps,
    rolling_mean(clip_q, 10),
    linewidth=2,
    label="Clipped double-Q (smoothed)",
)
axes[1].set(xlabel="Environment steps", ylabel="Logged Q value")
axes[1].legend()
finish(fig, "sac_hopper_q_comparison.png")

print("Generated report figures in", OUT)
