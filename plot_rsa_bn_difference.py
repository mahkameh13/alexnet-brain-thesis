"""How much re-estimating BatchNorm changes brain alignment, per epoch: delta rho = rho(re-estimated) - rho(checkpoint).

Paper-recipe CustomCNN (week 2), mean over seeds 1-3 of the per-seed difference, band = +-1 std across seeds.
Rows = regions, columns = age groups, one line per layer. Zero line = no change.
Reads results/rsa_paper_seed*.csv and results/rsa_paper_bnimagenet_seed*.csv
Writes results/rsa_paper_bn_difference.png
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rsa_lib import AGE_GROUPS, CUSTOMCNN_LAYERS

REGIONS = ["VVC", "LOC", "V1"]
NAMES = {"twomonth": "2-month infants", "ninemonth": "9-month infants", "adult": "adults"}
keys = ["seed", "epoch", "region", "group", "layer"]
old = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/rsa_paper_seed*.csv"))])
new = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/rsa_paper_bnimagenet_seed*.csv"))])
d = new.merge(old, on=keys, suffixes=("_new", "_old"))
d["delta"] = d.rho_new - d.rho_old
colors = plt.cm.viridis([i / (len(CUSTOMCNN_LAYERS) - 1) for i in range(len(CUSTOMCNN_LAYERS))])

fig, axes = plt.subplots(3, 3, figsize=(15, 9.5), sharex=True, sharey=True)
for i, region in enumerate(REGIONS):
    for j, g in enumerate(AGE_GROUPS):
        ax = axes[i, j]
        ax.axhspan(-0.03, 0.03, color="0.9", zorder=0)
        ax.axhline(0, color="k", lw=0.7)
        for layer, c in zip(CUSTOMCNN_LAYERS, colors):
            s = d[(d.region == region) & (d.group == g) & (d.layer == layer)].groupby("epoch").delta.agg(["mean", "std"])
            ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=c, alpha=0.12, lw=0)
            ax.plot(s.index, s["mean"], color=c, lw=1.5, marker="o", ms=2.2, label=layer)
        ax.grid(alpha=0.25)
        if i == 0:
            ax.set_title(NAMES[g])
        if j == 0:
            ax.set_ylabel(f"{region}\nchange in rho")
        if i == 2:
            ax.set_xlabel("epoch")
axes[0, -1].legend(fontsize=8, ncol=2, loc="upper right")
fig.suptitle("Effect of re-estimating BatchNorm on brain alignment (week 2, paper-recipe CustomCNN)\n"
             "change in Spearman rho = re-estimated on ImageNet minus checkpoint statistics; "
             "mean of 3 seeds, band = +-1 std; grey band = +-0.03", fontsize=10.5)
fig.tight_layout()
fig.savefig("results/rsa_paper_bn_difference.png", dpi=130)
print("wrote results/rsa_paper_bn_difference.png")

summary = d.groupby(d.epoch == 0).delta.agg(lambda x: x.abs().mean())
print(f"mean |delta rho|: epoch 0 = {summary[True]:.3f}, epochs 1-20 = {summary[False]:.3f}")
print(f"share of (epoch>=1, region, group, layer) cells with |mean delta| < 0.03: "
      f"{(d[d.epoch > 0].groupby(['epoch','region','group','layer']).delta.mean().abs() < 0.03).mean():.0%}")
