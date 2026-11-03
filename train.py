"""Train standard AlexNet (torchvision architecture, random init) on ImageNet-1k.

Example (ALWAYS pick a free GPU with nvidia-smi first and set CUDA_VISIBLE_DEVICES):
    CUDA_VISIBLE_DEVICES=1 python train.py --seed 0 --epochs 20 --project alexnet-brain

What it does
  * trains with SGD + cosine learning-rate decay, mixed precision (fp16)
  * saves WEIGHTS-ONLY checkpoints on a schedule: step 0 (untrained), a few early steps
    inside epoch 1, then chosen epochs (--ckpt-steps / --ckpt-epochs)
  * separately saves 'last.pt' (weights + optimizer + counters) after every epoch so a
    stopped job can resume automatically (re-run the same command)
  * logs per-epoch train/val loss and accuracy to logs/ (always) and to W&B (if available)
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset

p = argparse.ArgumentParser()
p.add_argument("--seed", type=int, required=True)
p.add_argument("--epochs", type=int, default=20)
p.add_argument("--batch-size", type=int, default=256)
p.add_argument("--lr", type=float, default=0.01)
p.add_argument("--momentum", type=float, default=0.9)
p.add_argument("--weight-decay", type=float, default=5e-4)
p.add_argument("--workers", type=int, default=10, help="data-loading CPU workers (shared machine: keep modest)")
p.add_argument("--data-root", default="/home/data/ImageNet/ILSVRC/Data/CLS-LOC")  # ORIGINAL full-resolution ImageNet; read-only, never modified
p.add_argument("--val-csv", default="/home/data/ImageNet/LOC_val_solution.csv", help="val labels for the flat val folder")
p.add_argument("--ckpt-dir", default="checkpoints")
p.add_argument("--ckpt-steps", default="0,100,500,2000", help="global training steps to save (0 = untrained init)")
p.add_argument("--ckpt-epochs", default="1,2,3,4,5,6,8,10,12,14,16,18,20", help="epochs after which to save")
p.add_argument("--subset-per-class", type=int, default=0, help="TEST ONLY: use only N train images per class")
p.add_argument("--val-subset-per-class", type=int, default=0, help="TEST ONLY: use only N val images per class")
p.add_argument("--run-name", default=None, help="default alexnet_seed<seed>")
p.add_argument("--no-wandb", action="store_true")
p.add_argument("--entity", default=os.environ.get("WANDB_ENTITY"))
p.add_argument("--project", default=os.environ.get("WANDB_PROJECT"))
args = p.parse_args()

run_name = args.run_name or f"alexnet_seed{args.seed}"
ckpt_dir = os.path.join(args.ckpt_dir, run_name)
os.makedirs(ckpt_dir, exist_ok=True)
os.makedirs("logs", exist_ok=True)
log_path = f"logs/train_{run_name}.jsonl"
ckpt_steps = {int(x) for x in args.ckpt_steps.split(",") if x != ""}
ckpt_epochs = {int(x) for x in args.ckpt_epochs.split(",") if x != ""}


def log(d):
    """Append one JSON line to the local log (the fallback record) and print it."""
    d = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), **d}
    with open(log_path, "a") as f:
        f.write(json.dumps(d) + "\n")
    print(json.dumps(d), flush=True)


# ------------------------------------------------------------------ seeds
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed); torch.cuda.manual_seed_all(args.seed)
torch.backends.cudnn.benchmark = True  # faster; means runs are not bit-for-bit reproducible
device = "cuda"
assert torch.cuda.is_available(), "No GPU visible - did you set CUDA_VISIBLE_DEVICES to a free GPU?"


# ------------------------------------------------------------------ data
class ImageNetFolder(Dataset):
    """Reads <root>/<split>/<wnid>/*.JPEG. Class index = sorted wnid order (the standard one)."""

    def __init__(self, root, split, transform, per_class=0, classes=None, val_csv=None):
        base = os.path.join(root, split)
        flat_val = split == "val" and not os.path.isdir(os.path.join(base, sorted(os.listdir(base))[0]))
        self.classes = classes if classes is not None else sorted(os.listdir(base))
        self.samples = []
        if flat_val:
            # Original ImageNet layout: all val images in ONE folder, labels in LOC_val_solution.csv
            # (column ImageId, PredictionString = "<wnid> <box> ..."; the first token is the class).
            import csv
            idx = {c: i for i, c in enumerate(self.classes)}
            counts = {}
            for r in sorted(csv.DictReader(open(val_csv)), key=lambda r: r["ImageId"]):
                ci = idx[r["PredictionString"].split()[0]]
                if per_class and counts.get(ci, 0) >= per_class:
                    continue
                counts[ci] = counts.get(ci, 0) + 1
                self.samples.append((os.path.join(base, r["ImageId"] + ".JPEG"), ci))
        else:  # <split>/<wnid>/*.JPEG
            for ci, c in enumerate(self.classes):
                files = sorted(os.listdir(os.path.join(base, c)))
                if per_class:
                    files = files[:per_class]
                self.samples += [(os.path.join(base, c, f), ci) for f in files]
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, i):
        # If a file is unreadable, log it and use the next image of the same split instead of crashing.
        for attempt in range(5):
            path, y = self.samples[(i + attempt) % len(self.samples)]
            try:
                with Image.open(path) as im:
                    x = self.transform(im.convert("RGB"))  # also handles the few CMYK/greyscale/PNG-disguised files
                return x, y
            except Exception as e:
                with open("logs/unreadable_images.txt", "a") as f:
                    f.write(f"{path}\t{repr(e)}\n")
        raise RuntimeError("5 unreadable images in a row")


norm = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
train_tf = T.Compose([T.RandomResizedCrop(224), T.RandomHorizontalFlip(), T.ToTensor(), norm])
val_tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(), norm])
train_ds = ImageNetFolder(args.data_root, "train", train_tf, args.subset_per_class)
val_ds = ImageNetFolder(args.data_root, "val", val_tf, args.val_subset_per_class,
                        classes=train_ds.classes, val_csv=args.val_csv)
assert len(train_ds.classes) == 1000
print(f"train images: {len(train_ds)}  val images: {len(val_ds)}", flush=True)


def make_train_loader(epoch):
    g = torch.Generator(); g.manual_seed(args.seed * 1000 + epoch)  # data order depends on (seed, epoch) -> resumable
    return DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, drop_last=True, generator=g,
                      num_workers=args.workers, pin_memory=True, persistent_workers=False, prefetch_factor=4)


val_loader = DataLoader(val_ds, batch_size=500, shuffle=False, num_workers=args.workers, pin_memory=True)
steps_per_epoch = len(train_ds) // args.batch_size
total_steps = steps_per_epoch * args.epochs

# ------------------------------------------------------------------ model / optimiser
model = torchvision.models.alexnet(weights=None, num_classes=1000).to(device)  # standard AlexNet, random init
opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
scaler = torch.amp.GradScaler("cuda")
loss_fn = nn.CrossEntropyLoss()


def lr_at(step):
    """Cosine decay from args.lr to 0 over all steps."""
    return float(0.5 * args.lr * (1 + np.cos(np.pi * step / total_steps)))  # plain float: numpy scalars break safe torch.load


def save_ckpt(step, epoch):
    """Weights-only checkpoint (plus 3 plain numbers describing when it was taken)."""
    path = os.path.join(ckpt_dir, f"ckpt_step{step:07d}_epoch{epoch:02d}.pt")
    torch.save({"model": model.state_dict(), "step": step, "epoch": epoch, "seed": args.seed}, path)
    log({"event": "checkpoint", "path": path, "step": step, "epoch": epoch})
    return path


@torch.no_grad()
def evaluate():
    model.eval()  # dropout off
    tot_loss = correct = n = 0
    for x, y in val_loader:
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
        tot_loss += loss_fn(out.float(), y).item() * len(y)
        correct += (out.argmax(1) == y).sum().item(); n += len(y)
    return tot_loss / n, correct / n


# ------------------------------------------------------------------ resume
start_epoch, step = 0, 0
last_path = os.path.join(ckpt_dir, "last.pt")
if os.path.exists(last_path):
    s = torch.load(last_path, map_location=device)
    model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
    start_epoch, step = s["epoch"], s["step"]
    log({"event": "resumed", "from_epoch": start_epoch, "step": step})

# ------------------------------------------------------------------ W&B (optional, never fatal)
wb = None
if not args.no_wandb:
    try:
        import wandb
        wb = wandb.init(entity=args.entity, project=args.project, name=run_name, id=run_name + "_fullres", resume="allow",
                        config=dict(seed=args.seed, epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay, workers=args.workers,
                                    optimizer="SGD", lr_schedule="cosine (per step)", amp="fp16",
                                    train_images=len(train_ds), ckpt_steps=sorted(ckpt_steps),
                                    ckpt_epochs=sorted(ckpt_epochs), subset_per_class=args.subset_per_class))
    except Exception as e:
        print("W&B unavailable, continuing with local logs only:", repr(e), flush=True)
        log({"event": "wandb_unavailable", "error": repr(e)})


def wlog(d, **kw):
    if wb is not None:
        try:
            wb.log(d, **kw)
        except Exception as e:  # keep training even if logging hiccups
            print("W&B log failed:", repr(e), flush=True)


log({"event": "start", "run": run_name, "args": vars(args), "steps_per_epoch": steps_per_epoch})
if step == 0 and 0 in ckpt_steps:
    wlog({"checkpoint_path": save_ckpt(0, 0)}, step=0)

# ------------------------------------------------------------------ training loop
for epoch in range(start_epoch, args.epochs):
    model.train()
    t0 = time.time(); run_loss = run_correct = run_n = 0
    for x, y in make_train_loader(epoch):
        for gparam in opt.param_groups:
            gparam["lr"] = lr_at(step)
        x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16):
            out = model(x)
            loss = loss_fn(out.float(), y)
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
        step += 1
        run_loss += loss.item() * len(y); run_correct += (out.argmax(1) == y).sum().item(); run_n += len(y)
        if step % 100 == 0:
            wlog({"train/loss_step": loss.item(), "lr": lr_at(step)}, step=step)
        if step in ckpt_steps and step > 0:
            wlog({"checkpoint_path": save_ckpt(step, epoch)}, step=step)
    train_loss, train_acc = run_loss / run_n, run_correct / run_n
    val_loss, val_acc = evaluate()
    rec = {"event": "epoch", "epoch": epoch + 1, "step": step, "train_loss": train_loss, "train_acc": train_acc,
           "val_loss": val_loss, "val_acc": val_acc, "lr": lr_at(step), "epoch_seconds": time.time() - t0,
           "images_per_sec": run_n / (time.time() - t0)}
    log(rec)
    wlog({"epoch": epoch + 1, "train/loss": train_loss, "train/acc": train_acc, "val/loss": val_loss, "val/acc": val_acc,
          "epoch_seconds": rec["epoch_seconds"], "images_per_sec": rec["images_per_sec"]}, step=step)
    if (epoch + 1) in ckpt_epochs:
        wlog({"checkpoint_path": save_ckpt(step, epoch + 1)}, step=step)
    torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch + 1, "step": step}, last_path)  # for resuming (overwritten each epoch)

log({"event": "finished", "final_val_acc": val_acc})
if wb is not None:
    wb.finish()
