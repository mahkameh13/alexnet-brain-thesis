"""TODO 2: brain alignment over training for the reference-paper networks, with last week's run as a shadow.

For every region, one figure with one panel per layer:
  * solid line + shaded band  = NEW run (paper's code/recipe, CustomCNN), mean over seeds 1-3 +- 1 std across seeds
  * dashed line + light band  = OLD run (last week, torchvision AlexNet + SGD), mean over seeds 0-2 +- 1 std
  * dotted horizontal line    = fully trained, pretrained torchvision AlexNet (stage 3), for reference
Colors = age groups. x axis = training progress in epochs (old run: step / 5004 steps per epoch, so its early
within-epoch checkpoints appear between 0 and 1).
A second figure shows the last layer (fc2) for the three regions side by side.

Usage: python plot_rsa_paper_vs_old.py [--regions VVC LOC V1] [--no-old]
"""
import argparse
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rsa_lib import AGE_GROUPS, ALEXNET_LAYERS

p = argparse.ArgumentParser()
p.add_argument("--regions", nargs="+", default=["VVC", "LOC", "V1"])
p.add_argument("--no-old", action="store_true", help="do not draw last week's run as a shadow")
args = p.parse_args()

new = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/rsa_paper_seed*.csv"))])
new["x"] = new.epoch.astype(float)
old = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/rsa_alexnet_seed*.csv"))])
old["x"] = old.step / 5004  # 1,281,167 images // batch 256 = 5004 steps per epoch
pre = pd.read_csv("results/stage3_pretrained_rsa.csv")

colors = {"twomonth": "tab:red", "ninemonth": "tab:orange", "adult": "tab:blue"}
names = {"twomonth": "2-month infants", "ninemonth": "9-month infants", "adult": "adults"}
n_new, n_old = new.seed.nunique(), old.seed.nunique()
last_epoch = int(new.epoch.max())
print(f"new run: seeds {sorted(new.seed.unique())}, epochs 0-{last_epoch} | old run: seeds {sorted(old.seed.unique())}")


def band(ax, df, color, solid):
    s = df.groupby("x").rho.agg(["mean", "std"]).fillna(0)
    if solid:
        ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=color, alpha=0.25, lw=0)
        return ax.plot(s.index, s["mean"], color=color, lw=1.8, marker="o", ms=2.5)[0]
    ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=color, alpha=0.08, lw=0)
    return ax.plot(s.index, s["mean"], color=color, lw=1.0, ls="--", alpha=0.55)[0]


def draw(ax, region, layer):
    for g in AGE_GROUPS:
        sel = lambda d: d[(d.region == region) & (d.layer == layer) & (d.group == g)]
        if not args.no_old:
            band(ax, sel(old), colors[g], solid=False)
        band(ax, sel(new), colors[g], solid=True)
        ax.axhline(sel(pre).rho.iloc[0], color=colors[g], ls=":", lw=1, alpha=0.7)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(-0.3, 20.3)


def legend(ax):
    h = [plt.Line2D([], [], color=colors[g], lw=1.8, label=names[g]) for g in AGE_GROUPS]
    h += [plt.Line2D([], [], color="k", lw=1.8, label=f"paper recipe (mean ± std, {n_new} seeds)")]
    if not args.no_old:
        h += [plt.Line2D([], [], color="k", lw=1, ls="--", alpha=0.55, label=f"last week's run ({n_old} seeds)")]
    h += [plt.Line2D([], [], color="k", lw=1, ls=":", label="pretrained torchvision AlexNet")]
    ax.legend(handles=h, fontsize=7, loc="best")


suffix = "" if not args.no_old else "_noold"
for region in args.regions:
    fig, axes = plt.subplots(1, len(ALEXNET_LAYERS), figsize=(3.1 * len(ALEXNET_LAYERS), 3.6), sharey=True)
    for ax, layer in zip(axes, ALEXNET_LAYERS):
        draw(ax, region, layer)
        ax.set_title(layer); ax.set_xlabel("training epoch")
    axes[0].set_ylabel("RSA (Spearman rho)"); legend(axes[0])
    fig.suptitle(f"Brain region {region}: alignment over training. Solid + band = paper's AlexNet-style CNN and recipe "
                 f"(mean ± 1 std across seeds); dashed = last week's torchvision AlexNet run")
    fig.tight_layout(); fig.savefig(f"results/rsa_paper_vs_old_{region}{suffix}.png", dpi=140); plt.close(fig)

fig, axes = plt.subplots(1, len(args.regions), figsize=(5.2 * len(args.regions), 3.9), sharey=True)
for ax, region in zip(axes, args.regions):
    draw(ax, region, "fc2"); ax.set_title(f"{region}, last hidden layer (fc2)"); ax.set_xlabel("training epoch")
axes[0].set_ylabel("RSA (Spearman rho)"); legend(axes[0])
fig.tight_layout(); fig.savefig(f"results/rsa_paper_vs_old_fc2_summary{suffix}.png", dpi=140)
print("saved results/rsa_paper_vs_old_*.png")
