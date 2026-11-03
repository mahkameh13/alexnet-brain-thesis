"""Stage 5c: plot brain alignment across training checkpoints, for all seeds found in results/.

Figure 1 (per region): one panel per AlexNet layer; x = training step (symlog; 0 = untrained network);
                       one line per age group = mean over seeds, thin lines = individual seeds.
Figure 2: alignment of the last layer (fc2) with each age group, all three regions side by side.
Usage: python plot_rsa_checkpoints.py [--regions VVC LOC V1]
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
args = p.parse_args()
df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/rsa_alexnet_seed*.csv"))])
seeds = sorted(df.seed.unique())
colors = {"twomonth": "tab:red", "ninemonth": "tab:orange", "adult": "tab:blue"}
names = {"twomonth": "2-month infants", "ninemonth": "9-month infants", "adult": "adults"}
print("seeds found:", seeds, "| checkpoints:", df.step.nunique())


def draw(ax, region, layer):
    for g in AGE_GROUPS:
        d = df[(df.region == region) & (df.layer == layer) & (df.group == g)]
        for s in seeds:
            ds = d[d.seed == s].sort_values("step")
            ax.plot(ds.step, ds.rho, color=colors[g], alpha=0.25, lw=0.8)
        m = d.groupby("step").rho.mean()
        ax.plot(m.index, m.values, color=colors[g], marker="o", ms=3, label=names[g])
    ax.set_xscale("symlog", linthresh=100)
    ax.axhline(0, color="gray", lw=0.5)


for region in args.regions:
    fig, axes = plt.subplots(1, len(ALEXNET_LAYERS), figsize=(3 * len(ALEXNET_LAYERS), 3.4), sharey=True)
    for ax, layer in zip(axes, ALEXNET_LAYERS):
        draw(ax, region, layer)
        ax.set_title(layer); ax.set_xlabel("training step")
    axes[0].set_ylabel("RSA (Spearman rho)"); axes[0].legend(fontsize=7)
    fig.suptitle(f"Brain region {region}: alignment during training (thick = mean of {len(seeds)} seeds, thin = single seeds)")
    fig.tight_layout(); fig.savefig(f"results/rsa_across_training_{region}.png", dpi=140); plt.close(fig)

fig, axes = plt.subplots(1, len(args.regions), figsize=(5 * len(args.regions), 3.6), sharey=True)
for ax, region in zip(axes, args.regions):
    draw(ax, region, "fc2"); ax.set_title(f"{region}, last layer (fc2)"); ax.set_xlabel("training step")
axes[0].set_ylabel("RSA (Spearman rho)"); axes[0].legend(fontsize=8)
fig.tight_layout(); fig.savefig("results/rsa_across_training_fc2_summary.png", dpi=140)
print("saved results/rsa_across_training_*.png")
