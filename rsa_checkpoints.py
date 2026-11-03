"""Stage 5b: run every saved checkpoint of one training run through the RSA pipeline.

Usage:  python rsa_checkpoints.py --seed 0 [--wandb --project alexnet-brain]
Reads   checkpoints/alexnet_seed<N>/ckpt_step*_epoch*.pt   (weights-only files, loaded safely)
Writes  results/rsa_alexnet_seed<N>.csv  (one row per checkpoint x region x age group x layer)
Already-processed checkpoints are skipped, so it can be re-run while training is still going.
Runs on the CPU (36 images) - it never touches a GPU.
"""
import argparse
import glob
import os
import re

import pandas as pd
import torch
import torchvision

from rsa_lib import (AGE_GROUPS, ALEXNET_LAYERS, extract_activations, load_brain_labels, load_brain_rdms,
                     load_stimuli, model_rdm, rsa)

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--regions", nargs="+", default=["VVC", "LOC", "V1"])
p.add_argument("--wandb", action="store_true")
p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
p.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
args = p.parse_args()
torch.set_num_threads(4)  # shared machine: do not take all cores

out_csv = f"results/rsa_alexnet_seed{args.seed}.csv"
os.makedirs("results", exist_ok=True)
done = pd.read_csv(out_csv) if os.path.exists(out_csv) else pd.DataFrame()
done_steps = set(done["step"]) if len(done) else set()

images = load_stimuli(load_brain_labels())
brain = load_brain_rdms()

rows = []
paths = sorted(glob.glob(f"checkpoints/alexnet_seed{args.seed}/ckpt_step*_epoch*.pt"))
for path in paths:
    step, epoch = map(int, re.search(r"step(\d+)_epoch(\d+)", path).groups())
    if step in done_steps:
        continue
    ck = torch.load(path, map_location="cpu", weights_only=True)
    model = torchvision.models.alexnet(weights=None, num_classes=1000)
    model.load_state_dict(ck["model"])
    rdms = {layer: model_rdm(a) for layer, a in extract_activations(model, images).items()}  # eval mode inside
    for region in args.regions:
        for g in AGE_GROUPS:
            for layer in ALEXNET_LAYERS:
                rows.append(dict(seed=args.seed, step=step, epoch=epoch, region=region, group=g, layer=layer,
                                 rho=rsa(rdms[layer], brain[g][region])))
    print(f"processed {os.path.basename(path)}", flush=True)

if rows:
    new = pd.DataFrame(rows)
    allrows = pd.concat([done, new]).sort_values(["step", "region", "group", "layer"]).reset_index(drop=True)
    allrows.to_csv(out_csv, index=False)
    if args.wandb:
        try:
            import wandb
            run = wandb.init(entity=args.entity, project=args.project, name=f"alexnet_seed{args.seed}_rsa",
                             id=f"alexnet_seed{args.seed}_rsa_{len(allrows)}", config=dict(seed=args.seed, regions=args.regions))
            for step, d in allrows.groupby("step"):
                run.log({f"rsa/{r.region}/{r.group}/{r.layer}": r.rho for r in d.itertuples()} | {"train_step": int(step)})
            run.log({"rsa_table": wandb.Table(dataframe=allrows)})
            run.finish()
        except Exception as e:
            print("W&B logging failed, continuing with local results only:", repr(e))
print(f"{out_csv}: {len(pd.read_csv(out_csv)) if os.path.exists(out_csv) else 0} rows")
