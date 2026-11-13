"""RSA for the networks trained with the reference paper's code (visreps repo, CustomCNN, 1000 classes).

Same pipeline as rsa_checkpoints.py (same 36 images, same brain RDMs, same 7 layers, Spearman), so the
results are directly comparable with last week's torchvision-AlexNet runs.

Usage:  ~/visreps/.venv/bin/python rsa_checkpoints_paper.py --seed 1
Reads   ~/visreps/model_checkpoints/alexnet_paper_recipe/cfg1000<a|b|c>/checkpoint_epoch_<E>.pth
Writes  results/rsa_paper_seed<N>.csv  (one row per checkpoint x region x age group x layer)
Already-processed epochs are skipped, so it can be re-run while training is still going. CPU only.
"""
import argparse
import glob
import os
import re
import sys

import pandas as pd
import torch

VISREPS = os.path.expanduser("~/visreps")
sys.path.insert(0, VISREPS)  # the checkpoints store the whole model object, which needs the visreps package

from rsa_lib import (AGE_GROUPS, CUSTOMCNN_LAYERS, extract_activations, load_brain_labels, load_brain_rdms,
                     load_stimuli, model_rdm, rsa)

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True, help="1, 2 or 3 (the paper's seeds)")
p.add_argument("--regions", nargs="+", default=["VVC", "LOC", "V1"])
args = p.parse_args()
torch.set_num_threads(4)  # shared machine: do not take all cores

out_csv = f"results/rsa_paper_seed{args.seed}.csv"
done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
done_epochs = set(done["epoch"]) if len(done) else set()

images = load_stimuli(load_brain_labels())
brain = load_brain_rdms()
letter = "abc"[args.seed - 1]
ckpt_dir = f"{VISREPS}/model_checkpoints/alexnet_paper_recipe/cfg1000{letter}"

rows = []
paths = sorted(glob.glob(f"{ckpt_dir}/checkpoint_epoch_*.pth"), key=lambda s: int(re.search(r"_(\d+)\.pth", s)[1]))
for path in paths:
    epoch = int(re.search(r"_(\d+)\.pth", path)[1])
    if epoch in done_epochs:
        continue
    # our own checkpoint files (whole pickled model), so weights_only=False is safe here
    model = torch.load(path, map_location="cpu", weights_only=False)["model"]
    rdms = {layer: model_rdm(a) for layer, a in extract_activations(model, images, layers=CUSTOMCNN_LAYERS).items()}
    for region in args.regions:
        for g in AGE_GROUPS:
            for layer in CUSTOMCNN_LAYERS:
                rows.append(dict(seed=args.seed, epoch=epoch, region=region, group=g, layer=layer,
                                 rho=rsa(rdms[layer], brain[g][region])))
    print(f"processed epoch {epoch}", flush=True)

if rows:
    allrows = pd.concat([done, pd.DataFrame(rows)]).sort_values(["epoch", "region", "group", "layer"])
    allrows.to_csv(out_csv, index=False)
print(f"{out_csv}: {len(pd.read_csv(out_csv)) if os.path.exists(out_csv) else 0} rows")
