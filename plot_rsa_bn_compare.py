"""TODO 3 (BatchNorm): brain alignment over training for the paper-recipe networks, BatchNorm as saved vs re-estimated.

For every region, one figure with one panel per layer:
  * solid line + shaded band = BatchNorm statistics re-estimated on 512,000 ImageNet images (paper's recipe)
  * dashed line + light band = BatchNorm statistics from the checkpoint (what week 2 reported)
Mean over seeds 1-3, band = +-1 std across seeds. Colours = age groups.
A second figure shows fc2 for the three regions side by side.

Reads results/rsa_paper_seed<N>.csv and results/rsa_paper_bnimagenet_seed<N>.csv
Writes results/rsa_paper_bn_compare_<region>.png and results/rsa_paper_bn_compare_fc2_summary.png
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rsa_lib import AGE_GROUPS, CUSTOMCNN_LAYERS

REGIONS = ["VVC", "LOC", "V1"]
colors = {"twomonth": "tab:red", "ninemonth": "tab:orange", "adult": "tab:blue"}
names = {"twomonth": "2-month infants", "ninemonth": "9-month infants", "adult": "adults"}
old = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/rsa_paper_seed*.csv"))])
new = pd.concat([pd.read_csv(f) for f in sorted(glob.glob("results/rsa_paper_bnimagenet_seed*.csv"))])
print(f"recalibrated: seeds {sorted(new.seed.unique())}, epochs {new.epoch.min()}-{new.epoch.max()}")


def band(ax, df, color, solid):
    s = df.groupby("epoch").rho.agg(["mean", "std"]).fillna(0)
    ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=color, alpha=0.25 if solid else 0.08, lw=0)
    ax.plot(s.index, s["mean"], color=color, lw=1.8 if solid else 1.0, ls="-" if solid else "--",
            alpha=1 if solid else 0.6, marker="o" if solid else None, ms=2.5)


def draw(ax, region, layer):
    for g in AGE_GROUPS:
        sel = lambda d: d[(d.region == region) & (d.layer == layer) & (d.group == g)]
        band(ax, sel(old), colors[g], solid=False)
        band(ax, sel(new), colors[g], solid=True)
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xlim(-0.3, 20.3)


handles = [plt.Line2D([], [], color=colors[g], lw=2, label=names[g]) for g in AGE_GROUPS] + [
    plt.Line2D([], [], color="k", lw=1.8, label="BatchNorm re-estimated on ImageNet"),
    plt.Line2D([], [], color="k", lw=1.0, ls="--", alpha=0.6, label="BatchNorm from checkpoint (week 2)")]

for region in REGIONS:
    fig, axes = plt.subplots(2, 4, figsize=(16, 7), sharex=True, sharey=True)
    for ax, layer in zip(axes.flat, CUSTOMCNN_LAYERS):
        draw(ax, region, layer)
        ax.set_title(layer)
    axes.flat[-1].axis("off")
    axes.flat[-1].legend(handles=handles, loc="center", fontsize=9)
    for ax in axes[1]:
        ax.set_xlabel("epoch")
    for ax in axes[:, 0]:
        ax.set_ylabel("RSA (Spearman rho)")
    fig.suptitle(f"{region}: paper-recipe CNN, BatchNorm from checkpoint vs re-estimated on ImageNet "
                 f"(mean of 3 seeds, band = +-1 std)")
    fig.tight_layout()
    fig.savefig(f"results/rsa_paper_bn_compare_{region}.png", dpi=130)
    plt.close(fig)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2), sharey=True)
for ax, region in zip(axes, REGIONS):
    draw(ax, region, "fc2")
    ax.set_title(f"{region}, fc2")
    ax.set_xlabel("epoch")
axes[0].set_ylabel("RSA (Spearman rho)")
axes[-1].legend(handles=handles, fontsize=8, loc="lower right")
fig.tight_layout()
fig.savefig("results/rsa_paper_bn_compare_fc2_summary.png", dpi=130)
print("wrote results/rsa_paper_bn_compare_*.png")

m = lambda d: d.groupby(["region", "group", "layer", "epoch"]).rho.mean()
diff = (m(new) - m(old)).rename("rho_change").reset_index()
print("mean change in rho (re-estimated minus checkpoint), fc2:")
print(diff[diff.layer == "fc2"].pivot_table(index="epoch", columns=["region", "group"], values="rho_change").round(3)
      .loc[[0, 1, 3, 5, 10, 20]].to_string())
