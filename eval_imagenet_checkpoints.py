"""Accuracy and loss on ImageNet for every saved checkpoint of one seed, for either training setup.

  --setup ours   torchvision AlexNet trained with train.py         (seeds 0-2, checkpoints/alexnet_seed<N>/)
  --setup paper  CustomCNN trained with the reference paper's code (seeds 1-3, ~/visreps/model_checkpoints/...)
  --setup alexnet_paper  torchvision AlexNet trained with the paper's code and recipe (seeds 1-3)

Both setups are measured in exactly the same way, so the curves are comparable:
  * validation = the 50,000 images of ImageNet's OFFICIAL validation set (ILSVRC2012 val)
  * train      = a fixed random sample of 50,000 images from the paper's training partition (its 80% split of
                 the train folder; train.py trained on all of the train folder, so both setups have seen them)
  * heldout    = (--splits heldout, paper setup only) a fixed random sample of 50,000 images from the paper's 20%
                 held-out partition: the split the paper reports its accuracy on. Not held out for train.py.
  * network in eval mode (dropout off, BatchNorm uses the statistics saved in the checkpoint - this is also how
    the paper's own code measures classification accuracy), no augmentation: resize 256, center crop 224
  * loss = plain cross-entropy (no label smoothing): mean over images, and the median as well, because with its
    checkpoint BatchNorm statistics the paper network gives huge logits on a few images, which dominate the mean
  * fp32 throughout (in fp16 those logits overflow to inf)
  * --bn imagenet (paper setup only): BatchNorm statistics re-estimated on ImageNet first (calibrate_bn_imagenet.py)

Every image is decoded once and passed through all checkpoints of the seed, so a run takes a few minutes.

Usage:  CUDA_VISIBLE_DEVICES=1 ~/visreps/.venv/bin/python eval_imagenet_checkpoints.py --setup paper --seed 1
Writes  results/imagenet_eval_<setup>[_bnimagenet]_seed<N>.csv  (one row per checkpoint)
"""
import argparse
import csv
import glob
import os
import re
import sys

import pandas as pd
import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

VISREPS = os.path.expanduser("~/visreps")
TRAIN_DIR = "/home/data/ImageNet/ILSVRC/Data/CLS-LOC/train"
VAL_DIR = "/home/data/ImageNet/ILSVRC/Data/CLS-LOC/val"
VAL_CSV = "/home/data/ImageNet/LOC_val_solution.csv"
SAMPLE_FILE = "cache/imagenet_{}_sample_50k.txt"  # one "<wnid>/<file>" per line, shared by every run
STEPS_PER_EPOCH = 5004  # train.py: 1,281,167 images // batch 256

p = argparse.ArgumentParser()
p.add_argument("--setup", choices=["ours", "paper", "alexnet_paper"], required=True)
p.add_argument("--seed", type=int, required=True, help="ours: 0-2, paper: 1-3")
p.add_argument("--workers", type=int, default=6)
p.add_argument("--batch-size", type=int, default=250)
p.add_argument("--bn", choices=["checkpoint", "imagenet"], default="checkpoint",
               help="paper setup: BatchNorm statistics from the checkpoint, or re-estimated on ImageNet")
p.add_argument("--splits", nargs="+", choices=["val", "train", "heldout", "trainfull"], default=["val", "train"],
               help="trainfull = ALL 1,024,933 images of the paper's 80%% training partition (train = 50k sample of it)")
p.add_argument("--epochs", type=int, nargs="*", help="only these epochs (default: all checkpoints)")
p.add_argument("--out", help="output CSV (default: results/imagenet_eval_<setup>..._seed<N>.csv)")
p.add_argument("--limit", type=int, default=0, help="TEST ONLY: use only the first N images of each split")
args = p.parse_args()

# Class index = position of the wnid in sorted order. train.py uses this directly, and the paper code's
# folder_labels.json (~/visreps/imagenet_meta) is the same mapping (checked), so one label works for both.
classes = sorted(os.listdir(TRAIN_DIR))
class_idx = {c: i for i, c in enumerate(classes)}
assert len(classes) == 1000


def partition_sample(part):
    """Fixed sample of the paper's 80% training ("train") or 20% held-out ("heldout") partition of the train
    folder. Created once, then always read from the file."""
    path = SAMPLE_FILE.format(part)
    if not os.path.exists(path):
        sys.path.insert(0, VISREPS)
        from visreps.dataloaders.obj_cls_folder import ImageNetDataset  # reproduces the paper code's 80/20 split
        samples = ImageNetDataset(TRAIN_DIR, split="train" if part == "train" else "test").samples
        pick = torch.randperm(len(samples), generator=torch.Generator().manual_seed(0))[:50_000].tolist()
        os.makedirs("cache", exist_ok=True)
        with open(path, "w") as f:
            f.writelines(os.path.relpath(samples[i][0], TRAIN_DIR) + "\n" for i in sorted(pick))
    rel = [line.strip() for line in open(path)]
    return [(os.path.join(TRAIN_DIR, r), class_idx[r.split("/")[0]]) for r in rel]


def full_train_partition():
    """Every image of the paper's 80% training partition (every setup trained on all of them)."""
    path = "cache/imagenet_train_full.txt"
    if not os.path.exists(path):
        sys.path.insert(0, VISREPS)
        from visreps.dataloaders.obj_cls_folder import ImageNetDataset  # reproduces the paper code's 80/20 split
        samples = ImageNetDataset(TRAIN_DIR, split="train").samples
        os.makedirs("cache", exist_ok=True)
        with open(path, "w") as f:
            f.writelines(os.path.relpath(s[0], TRAIN_DIR) + "\n" for s in sorted(samples))
    rel = [line.strip() for line in open(path)]
    return [(os.path.join(TRAIN_DIR, r), class_idx[r.split("/")[0]]) for r in rel]


def val_samples():
    rows = sorted(csv.DictReader(open(VAL_CSV)), key=lambda r: r["ImageId"])
    return [(os.path.join(VAL_DIR, r["ImageId"] + ".JPEG"), class_idx[r["PredictionString"].split()[0]]) for r in rows]


class Images(Dataset):
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    def __init__(self, samples):
        self.samples = samples[:args.limit] if args.limit else samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        path, y = self.samples[i]
        with Image.open(path) as im:
            return self.tf(im.convert("RGB")), y


def load_models():
    """Return [(epoch, step, model)] sorted by training progress. `epoch` is fractional for the within-epoch
    checkpoints of train.py (step / steps per epoch)."""
    out = []
    if args.setup == "ours":
        for path in sorted(glob.glob(f"checkpoints/alexnet_seed{args.seed}/ckpt_step*_epoch*.pt")):
            ck = torch.load(path, map_location="cpu")
            model = torchvision.models.alexnet(weights=None, num_classes=1000)
            model.load_state_dict(ck["model"])
            out.append((ck["step"] / STEPS_PER_EPOCH, ck["step"], model))
    else:
        sys.path.insert(0, VISREPS)  # the checkpoints store the whole model object, which needs the visreps package
        run_dir = "alexnet_paper_recipe" if args.setup == "paper" else "alexnet_torchvision_paper_recipe"
        ckpt_dir = f"{VISREPS}/model_checkpoints/{run_dir}/cfg1000{'abc'[args.seed - 1]}"
        for path in glob.glob(f"{ckpt_dir}/checkpoint_epoch_*.pth"):
            epoch = int(re.search(r"_(\d+)\.pth", path)[1])
            if args.epochs is not None and epoch not in args.epochs:
                continue
            # our own checkpoint files (whole pickled model), so weights_only=False is safe here
            model = torch.load(path, map_location="cpu", weights_only=False)["model"]
            if args.bn == "imagenet":
                stats = torch.load(f"cache/bn_imagenet/paper_seed{args.seed}_epoch{epoch}.pt", weights_only=True)
                for name, m in model.named_modules():
                    if name in stats:
                        for key, value in stats[name].items():
                            getattr(m, key).copy_(value)
            out.append((float(epoch), None, model))
    assert out, "no checkpoints found"
    assert args.bn == "checkpoint" or args.setup == "paper", "torchvision AlexNet has no BatchNorm"
    assert "heldout" not in args.splits or args.setup != "ours", "train.py trained on the paper's held-out images"
    return sorted(out, key=lambda t: t[0])


models = [(e, s, m.cuda().eval()) for e, s, m in load_models()]
print(f"{args.setup} seed {args.seed}: {len(models)} checkpoints", flush=True)
stats = {}  # (split, checkpoint index) -> [per-image losses, top-1 correct, top-5 correct, images]
for split in args.splits:
    samples = (val_samples() if split == "val" else full_train_partition() if split == "trainfull"
               else partition_sample(split))
    loader = DataLoader(Images(samples), batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)
    for b, (x, y) in enumerate(loader):
        x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
        for k, (_, _, model) in enumerate(models):
            with torch.no_grad():
                logits = model(x)
            top5 = logits.topk(5, dim=1).indices
            s = stats.setdefault((split, k), [[], 0, 0, 0])
            s[0].append(F.cross_entropy(logits, y, reduction="none").cpu())
            s[1] += (top5[:, 0] == y).sum().item()
            s[2] += (top5 == y[:, None]).any(1).sum().item()
            s[3] += len(y)
        if b % 40 == 0:
            print(f"  {split}: batch {b}/{len(loader)}", flush=True)

rows = []
for k, (epoch, step, _) in enumerate(models):
    row = dict(setup=args.setup, bn=args.bn if args.setup == "paper" else "none", seed=args.seed, epoch=epoch, step=step)
    for split in args.splits:
        losses, top1, top5, n = stats[(split, k)]
        losses = torch.cat(losses).double()
        row.update({f"{split}_loss": losses.mean().item(), f"{split}_loss_median": losses.median().item(), f"{split}_acc": top1 / n, f"{split}_top5": top5 / n, f"n_{split}": n})
    rows.append(row)
tag = ("_bnimagenet" if args.bn == "imagenet" else "") + ("" if args.splits == ["val", "train"] else
                                                          "_" + "_".join(args.splits))
out_csv = args.out or f"results/imagenet_eval_{args.setup}{tag}_seed{args.seed}{'_TEST' if args.limit else ''}.csv"
pd.DataFrame(rows).to_csv(out_csv, index=False)
print(pd.DataFrame(rows)[["epoch"] + [f"{sp}_{m}" for sp in args.splits for m in ("loss", "loss_median", "acc")]]
      .round(4).to_string(index=False))
print(f"wrote {out_csv}")
