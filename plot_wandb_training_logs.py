"""Charts of the metrics logged to W&B DURING training (project alexnet-brain), week 1 and week 2.

Data: results/wandb/training_logs_from_wandb.csv (downloaded from the W&B runs alexnet_seed0-2 and
paper_recipe_seed1-3). Writes results/wandb/wandb_training_logs.png. Mean of 3 seeds, band = +-1 std.
Note: these are the numbers measured while training (augmentation and dropout on; week 2 loss includes label
smoothing 0.1), so they are not directly comparable to the eval-mode curves in imagenet_curves_*.png.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

df = pd.read_csv("results/wandb/training_logs_from_wandb.csv")
w1 = df[(df.setup == "week1") & df["val/acc"].notna()]
w2 = df[(df.setup == "week2") & df["training/loss"].notna()].copy()
w2["bin"] = (np.floor(w2.epoch * 4) / 4 + 0.125).clip(upper=19.875)  # quarter-epoch bins


def band(ax, d, x, y, color, label, scale=1.0):
    s = d.groupby(x)[y].agg(["mean", "std"]).fillna(0) * scale
    ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=color, alpha=0.22, lw=0)
    ax.plot(s.index, s["mean"], color=color, lw=1.8, label=label)


fig, axes = plt.subplots(1, 3, figsize=(17, 4.6))
ax = axes[0]
band(ax, w1, "epoch", "train/loss", "tab:purple", "week 1: train loss (per epoch)")
band(ax, w1, "epoch", "val/loss", "tab:orange", "week 1: val loss, official val (per epoch)")
band(ax, w2, "bin", "training/loss", "tab:green", "week 2: train loss incl. label smoothing (per 1/4 epoch)")
ax.set(title="Loss logged during training", xlabel="epoch", ylabel="cross-entropy loss", xlim=(0, 20.3))
ax.legend(fontsize=7.5)
ax = axes[1]
lr1 = df[(df.setup == "week1") & df["lr"].notna()].copy()
lr1["epoch_f"] = lr1["_step"] / 5004
s1 = lr1.groupby("epoch_f").lr.mean()
s2 = w2.groupby("epoch")["training/learning_rate"].mean()
ax.plot(s1.index, s1.values, color="tab:purple", label="week 1: SGD, lr 0.01, cosine to 0")
ax.plot(s2.index, s2.values, color="tab:green", label="week 2: AdamW, lr 5e-4, 2 warmup epochs, cosine to 5%")
ax.set(title="Learning-rate schedule", xlabel="epoch", ylabel="learning rate", yscale="log", xlim=(0, 20.3))
ax.legend(fontsize=7.5)
ax = axes[2]
band(ax, w1, "epoch", "train/acc", "tab:blue", "train acc (during training, augmented)", 100)
band(ax, w1, "epoch", "val/acc", "tab:orange", "val acc (official val)", 100)
ax.set(title="Week 1 accuracy as logged during training", xlabel="epoch", ylabel="top-1 accuracy (%)", xlim=(0, 20.3))
ax.legend(fontsize=7.5)
for a in axes:
    a.grid(alpha=0.3)
fig.suptitle("Metrics logged to W&B during training (project alexnet-brain), mean of 3 seeds, band = +-1 std", fontsize=10)
fig.tight_layout()
fig.savefig("results/wandb/wandb_training_logs.png", dpi=140)
print("wrote results/wandb/wandb_training_logs.png")
