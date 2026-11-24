"""Upload the 17/11 TODO results (ImageNet accuracy/loss over training, BatchNorm-recalibrated RSA) to W&B
(project alexnet-brain on the CoreWeave W&B server). Local files in results/ stay the source of truth.

Creates, in group "imagenet_eval":
  imagenet_eval_<setup>[_bnimagenet]_seed<N>  one run per CSV: train/val/heldout acc, top-5, loss (mean, median) vs epoch
In group "paper_recipe_rsa_bnimagenet":
  paper_recipe_bnimagenet_seed<N>_rsa  rsa/<region>/<group>/<layer> vs epoch
And one summary run "todo_1711_summary" with all tables and every imagenet_curves_* / rsa_paper_bn_compare_* plot.
"""
import glob
import os
import time

import sys

import pandas as pd
import wandb

# optional: only upload result files whose name contains this text (e.g. "alexnet_paper"), to avoid re-uploading
MATCH = sys.argv[1] if len(sys.argv) > 1 else ""

os.environ.setdefault("WANDB_BASE_URL", "https://api.forge.coreweave.com")
os.environ.setdefault("WANDB_PROJECT", "alexnet-brain")
stamp = time.strftime("%Y%m%d-%H%M")
tables = {}

for path in [f for f in sorted(glob.glob("results/imagenet_eval_*_seed[0-9].csv")) if MATCH in f]:
    name = os.path.basename(path)[:-4]
    df = pd.read_csv(path)
    tables[name] = df
    run = wandb.init(name=name, id=f"{name}_{stamp}", group="imagenet_eval", job_type="eval",
                     config=dict(setup=df.setup.iloc[0], bn=df.bn.iloc[0], seed=int(df.seed.iloc[0]), source=path))
    run.define_metric("epoch")
    for prefix in ("train", "val", "heldout"):
        run.define_metric(f"{prefix}/*", step_metric="epoch")
    for r in df.to_dict("records"):
        log = {"epoch": r["epoch"]}
        for prefix in ("train", "val", "heldout"):
            for m in ("acc", "top5", "loss", "loss_median"):
                if f"{prefix}_{m}" in r and pd.notna(r[f"{prefix}_{m}"]):
                    log[f"{prefix}/{m}"] = r[f"{prefix}_{m}"]
        run.log(log)
    run.finish()
    print("uploaded", name, flush=True)

for path in [f for f in sorted(glob.glob("results/rsa_paper_bnimagenet_seed*.csv")
                               + glob.glob("results/rsa_alexnet_paper_recipe_seed*.csv")) if MATCH in f]:
    df = pd.read_csv(path)
    seed = int(df.seed.iloc[0])
    tables[os.path.basename(path)[:-4]] = df
    base = os.path.basename(path)[:-4].replace("rsa_", "").replace(f"_seed{seed}", "")
    run = wandb.init(name=f"{base}_seed{seed}_rsa", id=f"{base}_seed{seed}_rsa_{stamp}", group=f"{base}_rsa",
                     job_type="rsa", config=dict(seed=seed, source=path))
    run.define_metric("epoch")
    run.define_metric("rsa/*", step_metric="epoch")
    for epoch, d in df.groupby("epoch"):
        run.log({f"rsa/{r.region}/{r.group}/{r.layer}": r.rho for r in d.itertuples()} | {"epoch": int(epoch)})
    run.finish()
    print("uploaded", path, flush=True)

run = wandb.init(name=f"todo_1711_summary{'_' + MATCH if MATCH else ''}", id=f"todo_1711_summary_{stamp}", group="imagenet_eval", job_type="summary")
run.log({f"tables/{k}": wandb.Table(dataframe=v) for k, v in tables.items()})
plots = sorted(glob.glob("results/imagenet_curves_*.png") + glob.glob("results/rsa_paper_bn_compare_*.png")
               + glob.glob("results/rsa_paper_vs_old_*.png")
               + glob.glob("results/rsa_three_setups_*.png"))
run.log({f"plots/{os.path.basename(p)[:-4]}": wandb.Image(p) for p in plots})
run.finish()
print(f"uploaded summary: {len(tables)} tables + {len(plots)} plots")
