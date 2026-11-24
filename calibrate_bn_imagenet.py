"""Re-estimate the BatchNorm statistics of every paper-recipe checkpoint of one seed on ImageNet.

This is the reference paper's own "imagenet" calibration recipe (visreps/models/batchnorm.py, the one the paper
uses for ResNet-50): reset the running statistics, then one pass over a fixed random subset of 512,000 images
from the paper's training partition, in fixed shuffled batches of 256, cumulative average (momentum=None),
only BatchNorm modules in training mode (dropout off), no weights changed, fp32. We call visreps' own dataset,
subset and batch functions, so the images and batches are exactly the ones its code would use.

The only difference from calling visreps once per checkpoint: each batch is decoded once and fed to all 21
checkpoints of the seed, which gives identical statistics but is ~20x faster.

Why: FOUNDCOG has no training images to calibrate on (the paper's default), so this is the paper's fallback.

Usage:  CUDA_VISIBLE_DEVICES=1 ~/visreps/.venv/bin/python calibrate_bn_imagenet.py --seed 1
Writes  cache/bn_imagenet/paper_seed<N>_epoch<E>.pt  ({module name: {running_mean, running_var, num_batches_tracked}})
"""
import argparse
import glob
import os
import re
import sys
import time

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset

VISREPS = os.path.expanduser("~/visreps")
sys.path.insert(0, VISREPS)
from visreps.dataloaders.obj_cls import get_obj_cls_loader  # noqa: E402
from visreps.models.batchnorm import IMAGENET_BATCH_SIZE, IMAGENET_IMAGES, _batches, _norm_layers  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True, help="1, 2 or 3 (the paper's seeds)")
p.add_argument("--workers", type=int, default=14)
p.add_argument("--epochs", type=int, nargs="*", help="only these epochs (default: all checkpoints)")
p.add_argument("--limit-batches", type=int, default=0, help="TEST ONLY: stop after N batches")
args = p.parse_args()

OUT = "cache/bn_imagenet"
os.makedirs(OUT, exist_ok=True)
ckpt_dir = f"{VISREPS}/model_checkpoints/alexnet_paper_recipe/cfg1000{'abc'[args.seed - 1]}"

# ---------------------------------------------------------------- models: reset BN, cumulative average, eval otherwise
models = {}
for path in glob.glob(f"{ckpt_dir}/checkpoint_epoch_*.pth"):
    epoch = int(re.search(r"_(\d+)\.pth", path)[1])
    if args.epochs and epoch not in args.epochs:
        continue
    if os.path.exists(f"{OUT}/paper_seed{args.seed}_epoch{epoch}.pt") and not args.limit_batches:
        continue
    # our own checkpoint files (whole pickled model), so weights_only=False is safe here
    model = torch.load(path, map_location="cpu", weights_only=False)["model"].cuda().eval()
    for m in _norm_layers(model).values():
        assert m.track_running_stats
        m.reset_running_stats()
        m.momentum = None
        m.train()
    models[epoch] = model
print(f"seed {args.seed}: calibrating {len(models)} checkpoints {sorted(models)}", flush=True)
if not models:
    sys.exit(0)

# ---------------------------------------------------------------- data: exactly visreps' _calibrate_on_imagenet
os.chdir(VISREPS)  # visreps reads IMAGENET_DATA_DIR / IMAGENET_LOCAL_DIR from its .env
data_cfg = {"dataset": "imagenet", "pca_labels": False, "data_augment": False,
            "batchsize": IMAGENET_BATCH_SIZE, "num_workers": args.workers}
train = get_obj_cls_loader(data_cfg, shuffle=False)[0]["train"]
os.chdir(os.path.dirname(os.path.abspath(__file__)))
n = min(IMAGENET_IMAGES, len(train))
pick = torch.randperm(len(train), generator=torch.Generator().manual_seed(0))[:n]
subset = Subset(train, sorted(pick.tolist()))
loader = DataLoader(subset, batch_sampler=_batches(n, IMAGENET_BATCH_SIZE), num_workers=args.workers,
                    pin_memory=True, prefetch_factor=4)

t0 = time.time()
with torch.no_grad():
    for b, (images, _) in enumerate(loader):
        images = images.cuda(non_blocking=True)
        for model in models.values():
            model(images)
        if b % 100 == 0:
            print(f"  batch {b}/{len(loader)}  {time.time() - t0:.0f}s", flush=True)
        if args.limit_batches and b + 1 >= args.limit_batches:
            break

for epoch, model in models.items():
    buffers = {name: {k: getattr(m, k).detach().cpu().clone() for k in ("running_mean", "running_var",
                                                                        "num_batches_tracked")}
               for name, m in _norm_layers(model).items()}
    assert all(torch.isfinite(v).all() for d in buffers.values() for v in d.values()), f"non-finite, epoch {epoch}"
    if args.limit_batches:
        print(f"TEST epoch {epoch}: first BN mean {buffers[next(iter(buffers))]['running_mean'][:3].tolist()}")
        continue
    torch.save(buffers, f"{OUT}/paper_seed{args.seed}_epoch{epoch}.pt")
print(f"done in {time.time() - t0:.0f}s", flush=True)
