"""Stage 3: test the RSA pipeline with the PRETRAINED torchvision AlexNet.

Output: results/stage3_pretrained_rsa.csv and results/stage3_pretrained_rsa.png
(layer-by-age-group alignment). Runs on CPU (only 36 images), so no GPU is used.

Optional W&B logging:  --wandb --entity <team> --project <name>
(you must have run `wandb login` yourself; if W&B fails we continue with local logs).
"""
import argparse
import os
import time

os.environ.setdefault("TORCH_HOME", os.path.abspath("cache"))  # keep downloaded weights inside the project

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision

from rsa_lib import (AGE_GROUPS, ALEXNET_LAYERS, extract_activations, load_brain_labels,
                     load_brain_rdms, load_stimuli, model_rdm, rsa, rsa_bootstrap_ci)

p = argparse.ArgumentParser()
p.add_argument("--regions", nargs="+", default=["LOC", "VVC", "V1"])
p.add_argument("--wandb", action="store_true")
p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
p.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
args = p.parse_args()
os.makedirs("results", exist_ok=True)

# --- 1. images and the pretrained model -------------------------------------------------
labels = load_brain_labels()
images = load_stimuli(labels)
print("images:", tuple(images.shape))
model = torchvision.models.alexnet(weights=torchvision.models.AlexNet_Weights.IMAGENET1K_V1)
acts = extract_activations(model, images)
rdms = {layer: model_rdm(a) for layer, a in acts.items()}
print({k: v.shape for k, v in acts.items()})

# --- 2. compare every layer with every age group and region ----------------------------------
brain = load_brain_rdms()
rows = []
for region in args.regions:
    for g in AGE_GROUPS:
        for layer in ALEXNET_LAYERS:
            r = rsa(rdms[layer], brain[g][region])
            lo, hi = rsa_bootstrap_ci(rdms[layer], brain[g][region])
            rows.append(dict(region=region, group=g, layer=layer, rho=r, ci_lo=lo, ci_hi=hi))
df = pd.DataFrame(rows)
df.to_csv("results/stage3_pretrained_rsa.csv", index=False)
print(df.pivot_table(index=["region", "layer"], columns="group", values="rho")[AGE_GROUPS].round(3).to_string())

# --- 3. plot: one panel per region, one line per age group -----------------------------------
colors = {"twomonth": "tab:red", "ninemonth": "tab:orange", "adult": "tab:blue"}
names = {"twomonth": "2-month infants", "ninemonth": "9-month infants", "adult": "adults"}
fig, axes = plt.subplots(1, len(args.regions), figsize=(5 * len(args.regions), 4), sharey=True)
for ax, region in zip(np.atleast_1d(axes), args.regions):
    for g in AGE_GROUPS:
        d = df[(df.region == region) & (df.group == g)]
        ax.errorbar(range(len(d)), d.rho, yerr=[d.rho - d.ci_lo, d.ci_hi - d.rho], marker="o",
                    capsize=3, color=colors[g], label=names[g])
    ax.axhline(0, color="gray", lw=0.5)
    ax.set_xticks(range(len(ALEXNET_LAYERS)))
    ax.set_xticklabels(list(ALEXNET_LAYERS))
    ax.set_title(f"Brain region: {region}")
    ax.set_xlabel("AlexNet layer (early -> late)")
np.atleast_1d(axes)[0].set_ylabel("RSA (Spearman rho)")
np.atleast_1d(axes)[0].legend()
fig.suptitle("Pretrained AlexNet vs infant / adult brain (36 images); bars = 95% bootstrap CI over images")
fig.tight_layout()
fig.savefig("results/stage3_pretrained_rsa.png", dpi=150)
print("saved results/stage3_pretrained_rsa.{csv,png}")

# --- 4. optional W&B logging (falls back to local logs only) ---------------------------------
if args.wandb:
    try:
        import wandb
        run = wandb.init(entity=args.entity, project=args.project, name="stage3_pretrained_alexnet_rsa",
                         config=dict(model="torchvision AlexNet IMAGENET1K_V1", regions=args.regions))
        run.log({"rsa_table": wandb.Table(dataframe=df), "rsa_plot": wandb.Image("results/stage3_pretrained_rsa.png")})
        run.finish()
        print("logged to W&B")
    except Exception as e:  # never let logging break the analysis
        print("W&B logging failed, continuing with local logs only:", repr(e))
