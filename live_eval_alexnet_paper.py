"""Live evaluation of the new run (torchvision AlexNet + paper recipe) while it trains.

Watches ~/visreps/model_checkpoints/alexnet_torchvision_paper_recipe/cfg1000<a|b|c>/ and, as soon as a new
checkpoint_epoch_<E>.pth is complete, measures train and official-validation accuracy and loss with
eval_imagenet_checkpoints.py (same images and settings as every other result) and logs them to W&B, one run per
seed: alexnet_paper_recipe_seed<N>_live_eval (metrics train/* and val/* against epoch).
Results also go to results/live/alexnet_paper_seed<N>_epoch<E>.csv, so a restart skips what is done.
The full evaluation after training (chain.sh) still runs and is the final record; this is only for watching.

Usage: CUDA_VISIBLE_DEVICES=0 ~/visreps/.venv/bin/python live_eval_alexnet_paper.py
"""
import os
import subprocess
import time

import pandas as pd
import wandb

os.environ.setdefault("WANDB_BASE_URL", "https://api.forge.coreweave.com")
os.environ.setdefault("WANDB_PROJECT", "alexnet-brain")
CKPT = os.path.expanduser("~/visreps/model_checkpoints/alexnet_torchvision_paper_recipe/cfg1000{}")
PY = os.path.expanduser("~/visreps/.venv/bin/python")
SEEDS, LAST_EPOCH = (1, 2, 3), 20
os.makedirs("results/live", exist_ok=True)

runs = {}
for s in SEEDS:
    runs[s] = wandb.init(name=f"alexnet_paper_recipe_seed{s}_live_eval", id=f"alexnet_paper_recipe_seed{s}_live_eval",
                         resume="allow", group="imagenet_eval_live", job_type="eval", reinit="create_new",
                         config=dict(setup="alexnet_paper", seed=s, note="live, per checkpoint while training"))
    runs[s].define_metric("epoch")
    runs[s].define_metric("train/*", step_metric="epoch")
    runs[s].define_metric("val/*", step_metric="epoch")


def ready(path):
    """A checkpoint counts as written once its size has not changed for 60 s."""
    if not os.path.exists(path):
        return False
    size, age = os.path.getsize(path), time.time() - os.path.getmtime(path)
    return size > 0 and age > 60


done = {(s, e) for s in SEEDS for e in range(LAST_EPOCH + 1) if os.path.exists(f"results/live/alexnet_paper_seed{s}_epoch{e}.csv")}
while len(done) < len(SEEDS) * (LAST_EPOCH + 1):
    todo = [(s, e) for e in range(LAST_EPOCH + 1) for s in SEEDS
            if (s, e) not in done and ready(f"{CKPT.format('abc'[s - 1])}/checkpoint_epoch_{e}.pth")]
    if not todo:
        time.sleep(60)
        continue
    s, e = todo[0]
    out = f"results/live/alexnet_paper_seed{s}_epoch{e}.csv"
    t0 = time.time()
    r = subprocess.run([PY, "eval_imagenet_checkpoints.py", "--setup", "alexnet_paper", "--seed", str(s),
                        "--epochs", str(e), "--out", out, "--workers", "8"], capture_output=True, text=True)
    if r.returncode != 0 or not os.path.exists(out):
        print(f"seed {s} epoch {e} FAILED:\n{r.stderr[-2000:]}", flush=True)
        time.sleep(120)
        continue
    row = pd.read_csv(out).iloc[0]
    runs[s].log({"epoch": int(row.epoch), **{f"{sp}/{m}": float(row[f"{sp}_{m}"]) for sp in ("train", "val")
                                             for m in ("acc", "top5", "loss", "loss_median")}})
    done.add((s, e))
    print(f"{time.strftime('%H:%M')} seed {s} epoch {e}: train {100 * row.train_acc:.1f}%  val {100 * row.val_acc:.1f}%  "
          f"({time.time() - t0:.0f}s)", flush=True)

for run in runs.values():
    run.finish()
print("all checkpoints evaluated")
