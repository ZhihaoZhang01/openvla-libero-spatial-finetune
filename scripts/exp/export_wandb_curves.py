#!/usr/bin/env python3
"""Export W&B training curves to experiments/assets/training_curves/ for README."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import wandb

AUTODL_ROOT = Path(os.environ.get("AUTODL_ROOT", "/root/autodl-tmp"))
OUT_DIR = AUTODL_ROOT / "experiments/assets/training_curves"
ENTITY = "zhihaozhang321-george-mason-university"

RUNS = {
    "D0": f"{ENTITY}/openvla-LoRA/lsyw8cci",
    "D1": f"{ENTITY}/openvla-LoRA-ablation/5juw27wz",
    "A-F": f"{ENTITY}/openvla-LoRA-ablation/xwa4y0sq",
    "S3k": f"{ENTITY}/openvla-LoRA-ablation/iyuu8nay",
    "S10k": f"{ENTITY}/openvla-LoRA-ablation/wh2355m8",
}

COLORS = {
    "D0": "#1f77b4",
    "D1": "#ff7f0e",
    "A-F": "#2ca02c",
    "S3k": "#d62728",
    "S10k": "#9467bd",
}


def fetch_history(api: wandb.Api, run_path: str):
    run = api.run(run_path)
    df = run.history(samples=50000, pandas=True)
    if df.empty:
        raise RuntimeError(f"No history for {run_path}")
    df = df.dropna(subset=["_step"]).sort_values("_step")
    return run, df


def plot_single(df, title: str, out_path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True)
    metrics = [
        ("train_loss", "Train Loss", axes[0]),
        ("action_accuracy", "Action Token Accuracy", axes[1]),
        ("l1_loss", "L1 Action Loss", axes[2]),
    ]
    for key, label, ax in metrics:
        if key in df.columns:
            ax.plot(df["_step"], df[key], linewidth=1.2, color="#2563eb")
            ax.set_ylabel(label)
            ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Training Step")
    fig.suptitle(title, fontsize=13, fontweight="bold")
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_compare(histories: dict, metric: str, ylabel: str, title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(11, 5))
    for name, df in histories.items():
        if metric not in df.columns:
            continue
        ax.plot(
            df["_step"],
            df[metric],
            label=name,
            color=COLORS.get(name, None),
            linewidth=1.5,
            alpha=0.9,
        )
    ax.set_xlabel("Training Step")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontweight="bold")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_quick_start(df, out_loss: Path, out_acc: Path) -> None:
    """Use first ~200 steps of D0 as environment sanity reference (short-train preview)."""
    short = df[df["_step"] <= 200].copy()
    if short.empty:
        short = df.head(30)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(short["_step"], short["train_loss"], color="#0ea5e9", linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Train Loss")
    ax.set_title("Quick Start — Train Loss (D0, first 200 steps)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_loss, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(short["_step"], short["action_accuracy"], color="#16a34a", linewidth=1.5)
    ax.set_xlabel("Training Step")
    ax.set_ylabel("Action Token Accuracy")
    ax.set_title("Quick Start — Action Accuracy (D0, first 200 steps)", fontweight="bold")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_acc, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    api = wandb.Api()
    histories = {}

    for name, path in RUNS.items():
        print(f"Fetching {name} …")
        run, df = fetch_history(api, path)
        histories[name] = df
        out = OUT_DIR / f"{name}_curves.png"
        plot_single(df, f"{name} — {run.name}", out)
        print(f"  → {out}")

    plot_compare(
        histories,
        "train_loss",
        "Train Loss",
        "Ablation Sweep — Train Loss",
        OUT_DIR / "sweep_train_loss_compare.png",
    )
    plot_compare(
        histories,
        "action_accuracy",
        "Action Token Accuracy",
        "Ablation Sweep — Action Accuracy",
        OUT_DIR / "sweep_action_accuracy_compare.png",
    )
    print(f"  → {OUT_DIR / 'sweep_train_loss_compare.png'}")
    print(f"  → {OUT_DIR / 'sweep_action_accuracy_compare.png'}")

    plot_quick_start(
        histories["D0"],
        OUT_DIR / "quick_start_loss.png",
        OUT_DIR / "quick_start_action_accuracy.png",
    )
    print(f"  → {OUT_DIR / 'quick_start_loss.png'}")
    print(f"  → {OUT_DIR / 'quick_start_action_accuracy.png'}")
    print("Done.")


if __name__ == "__main__":
    main()
