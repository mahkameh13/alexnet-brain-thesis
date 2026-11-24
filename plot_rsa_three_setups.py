"""Brain alignment over training for the three setups, so the early-peak-then-drop can be attributed:

  week 1   torchvision AlexNet, our code/recipe        results/rsa_alexnet_seed*.csv           (seeds 0-2)
  week 2   paper's CustomCNN, paper's code/recipe       results/rsa_paper_bnimagenet_seed*.csv  (seeds 1-3, BatchNorm
           re-estimated on ImageNet)
  new      torchvision AlexNet, paper's code/recipe     results/rsa_alexnet_paper_recipe_seed*.csv (seeds 1-3)

One figure per region (rows = age groups, columns = layers) and one fc2 summary (rows = age groups, columns =
regions). Colour = setup; line = mean over seeds, band = +-1 std. x = epoch (week 1: step / 5004 steps per epoch).
Writes results/rsa_three_setups_<region>.png and results/rsa_three_setups_fc2_summary.png
"""
import glob

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from rsa_lib import AGE_GROUPS, ALEXNET_LAYERS

REGIONS = ["VVC", "LOC", "V1"]
NAMES = {"twomonth": "2-month infants", "ninemonth": "9-month infants", "adult": "adults"}


def load(pattern, x_from_step=False):
    files = sorted(glob.glob(pattern))
    if not files:
        return None
    d = pd.concat([pd.read_csv(f) for f in files])
    d["x"] = d.step / 5004 if x_from_step else d.epoch.astype(float)
    return d


SETUPS = [
    ("week 1: torchvision AlexNet, our recipe", "tab:gray", load("results/rsa_alexnet_seed*.csv", x_from_step=True)),
    ("week 2: paper's CustomCNN, paper recipe (BN re-estimated)", "tab:green",
     load("results/rsa_paper_bnimagenet_seed*.csv")),
    ("new: torchvision AlexNet, paper recipe", "tab:purple", load("results/rsa_alexnet_paper_recipe_seed*.csv")),
]
SETUPS = [s for s in SETUPS if s[2] is not None]
print("setups:", [s[0] for s in SETUPS])


def draw(ax, region, group, layer):
    for _, color, d in SETUPS:
        sel = d[(d.region == region) & (d.group == group) & (d.layer == layer)]
        s = sel.groupby("x").rho.agg(["mean", "std"]).fillna(0)
        ax.fill_between(s.index, s["mean"] - s["std"], s["mean"] + s["std"], color=color, alpha=0.18, lw=0)
        ax.plot(s.index, s["mean"], color=color, lw=1.6, marker="o", ms=2)
    ax.axhline(0, color="k", lw=0.4)
    ax.set_xlim(-0.3, 20.3)
    ax.grid(alpha=0.25)


handles = [plt.Line2D([], [], color=c, lw=2, label=n) for n, c, _ in SETUPS]
for region in REGIONS:
    fig, axes = plt.subplots(3, len(ALEXNET_LAYERS), figsize=(22, 8.5), sharex=True, sharey=True)
    for i, g in enumerate(AGE_GROUPS):
        for j, layer in enumerate(ALEXNET_LAYERS):
            draw(axes[i, j], region, g, layer)
            if i == 0:
                axes[i, j].set_title(layer)
            if j == 0:
                axes[i, j].set_ylabel(f"{NAMES[g]}\nRSA (Spearman rho)")
            if i == 2:
                axes[i, j].set_xlabel("epoch")
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f"{region}: brain alignment over training, three setups (mean over seeds, band = +-1 std)", y=1.04)
    fig.tight_layout()
    fig.savefig(f"results/rsa_three_setups_{region}.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

fig, axes = plt.subplots(3, 3, figsize=(15, 9), sharex=True, sharey=True)
for i, g in enumerate(AGE_GROUPS):
    for j, region in enumerate(REGIONS):
        draw(axes[i, j], region, g, "fc2")
        if i == 0:
            axes[i, j].set_title(f"{region}, fc2")
        if j == 0:
            axes[i, j].set_ylabel(f"{NAMES[g]}\nRSA (Spearman rho)")
        if i == 2:
            axes[i, j].set_xlabel("epoch")
fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=10, bbox_to_anchor=(0.5, 1.02))
fig.tight_layout()
fig.savefig("results/rsa_three_setups_fc2_summary.png", dpi=130, bbox_inches="tight")
print("wrote results/rsa_three_setups_*.png")
