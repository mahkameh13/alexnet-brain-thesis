"""TODO 17/11: accuracy and loss over training, train vs validation, for both training setups.

One figure per (setup, quantity):
  results/imagenet_curves_ours_{acc,loss}.png             week 1: torchvision AlexNet, train.py (seeds 0-2)
  results/imagenet_curves_paper_{acc,loss}.png            week 2: paper code/recipe, BatchNorm as saved (seeds 1-3)
  results/imagenet_curves_paper_bnimagenet_{acc,loss}.png week 2 with BatchNorm re-estimated on ImageNet
Lines = mean across seeds, shaded band = +- 1 std across seeds. Colours: train (blue), official ImageNet
validation (orange), and for the paper setup also the paper's own 20% held-out split (green), when available.
Loss panels: mean cross-entropy (solid). For the paper network the mean is dominated by a few images with huge
logits, so the median per-image loss is drawn as well (dotted).

Reads results/imagenet_eval_<setup>[_bnimagenet][_heldout]_seed<N>.csv (eval_imagenet_checkpoints.py)
Usage: python plot_imagenet_curves.py
"""
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

COLORS = {"train": "tab:blue", "val": "tab:orange", "heldout": "tab:green"}
NAMES = {"train": "train (50k sample of training images)", "val": "validation (official ImageNet val, 50k)",
         "heldout": "paper's 20% held-out split (50k sample)"}
SETUPS = {
    "ours": ("Our setup (week 1): torchvision AlexNet, SGD", "imagenet_eval_ours_seed*.csv", None),
    "paper": ("Paper setup (week 2): CustomCNN, paper code, BatchNorm stats from checkpoint",
              "imagenet_eval_paper_seed*.csv", "imagenet_eval_paper_heldout_seed*.csv"),
    "alexnet_paper": ("New: torchvision AlexNet (week 1 network) trained with the paper's code and recipe",
                      "imagenet_eval_alexnet_paper_seed*.csv", "imagenet_eval_alexnet_paper_heldout_seed*.csv"),
    "paper_bnimagenet": ("Paper setup (week 2), BatchNorm re-estimated on 512k ImageNet images (paper recipe)",
                         "imagenet_eval_paper_bnimagenet_seed*.csv", "imagenet_eval_paper_bnimagenet_heldout_seed*.csv"),
}


def load(main, held):
    files = sorted(glob.glob(f"results/{main}"))
    if not files:
        return None
    df = pd.concat([pd.read_csv(f) for f in files])
    if held and glob.glob(f"results/{held}"):
        h = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"results/{held}"))])
        df = df.merge(h[["seed", "epoch"] + [c for c in h.columns if c.startswith("heldout_")]], on=["seed", "epoch"], how="left")
    return df


def band(ax, df, col, color, label, ls="-", alpha=0.22, lw=1.8):
    s = df.groupby("epoch")[col].agg(["mean", "std"]).fillna(0)
    ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=color, alpha=alpha, lw=0)
    ax.plot(s.index, s["mean"], color=color, ls=ls, lw=lw, marker="o" if ls == "-" else None, ms=3, label=label)


for key, (title, main, held) in SETUPS.items():
    df = load(main, held)
    if df is None:
        print(f"skip {key}: no results yet")
        continue
    splits = [s for s in ("train", "val", "heldout") if f"{s}_acc" in df.columns and df[f"{s}_acc"].notna().all()]
    n_seeds = df.seed.nunique()
    for quantity in ("acc", "loss"):
        fig, ax = plt.subplots(figsize=(7.5, 4.8))
        for s in splits:
            if quantity == "acc":
                d = df.assign(**{f"{s}_acc": 100 * df[f"{s}_acc"]})
                band(ax, d, f"{s}_acc", COLORS[s], NAMES[s])
            else:
                band(ax, df, f"{s}_loss", COLORS[s], NAMES[s] + " - mean")
                band(ax, df, f"{s}_loss_median", COLORS[s], NAMES[s] + " - median", ls=":", alpha=0.10, lw=1.4)
        ax.set_xlabel("epoch")
        ax.set_ylabel("top-1 accuracy (%)" if quantity == "acc" else "cross-entropy loss")
        if quantity == "loss" and df[[f"{s}_loss" for s in splits]].max().max() > 15:
            ax.set_yscale("log")
            ax.set_ylabel("cross-entropy loss (log scale)")
        ax.set_xlim(-0.3, 20.3)
        ax.grid(alpha=0.3)
        ax.set_title(f"{title}\nmean of {n_seeds} seeds, band = +-1 std", fontsize=9.5)
        ax.legend(fontsize=7.5, loc="lower right" if quantity == "acc" else "upper right")
        fig.tight_layout()
        out = f"results/imagenet_curves_{key}_{quantity}.png"
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)
    last = df[df.epoch == df.epoch.max()]
    print(f"  {key} epoch {df.epoch.max():g}: " + ", ".join(
        f"{s} acc {100 * last[f'{s}_acc'].mean():.1f}+-{100 * last[f'{s}_acc'].std():.1f}%" for s in splits))
