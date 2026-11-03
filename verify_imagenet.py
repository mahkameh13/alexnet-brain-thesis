"""Read-only checks that /home/data/ILSVRC_real is the genuine ILSVRC2012 data. Nothing is modified or copied.
1. val labels: official devkit ground truth (ILSVRC2012_validation_ground_truth.txt + meta.mat) vs the val/<wnid>/ folder each file sits in
2. the two copies on the server (ILSVRC_real vs ImageNet/ILSVRC) contain identical train file names, and identical sizes for a sample
3. the Kaggle-style val labels (LOC_val_solution.csv) agree with the official ground truth
"""
import csv, io, os, re, tarfile
import numpy as np
from scipy.io import loadmat
A, B = "/home/data/ILSVRC_real", "/home/data/ImageNet/ILSVRC/Data/CLS-LOC"

# --- official devkit
with tarfile.open(f"{A}/ILSVRC2012_devkit_t12.tar.gz") as t:
    gt = [int(x) for x in t.extractfile("ILSVRC2012_devkit_t12/data/ILSVRC2012_validation_ground_truth.txt").read().split()]
    meta = loadmat(io.BytesIO(t.extractfile("ILSVRC2012_devkit_t12/data/meta.mat").read()), squeeze_me=True)["synsets"]
id2wnid = {int(r["ILSVRC2012_ID"]): str(r["WNID"]) for r in meta if int(r["ILSVRC2012_ID"]) <= 1000}
print("devkit: %d val labels, %d classes" % (len(gt), len(id2wnid)))

# --- 1. val folder vs official ground truth
bad = tot = 0
for wnid in sorted(os.listdir(f"{A}/val")):
    for f in os.listdir(f"{A}/val/{wnid}"):
        n = int(re.search(r"val_(\d+)", f).group(1)); tot += 1
        bad += id2wnid[gt[n - 1]] != wnid
print(f"1. val: {tot} files checked against official ground truth -> {bad} mismatches")

# --- 3. Kaggle csv vs official
bad3 = tot3 = 0
for r in csv.DictReader(open("/home/data/ImageNet/LOC_val_solution.csv")):
    n = int(r["ImageId"].split("_")[-1]); tot3 += 1
    bad3 += r["PredictionString"].split()[0] != id2wnid[gt[n - 1]]
print(f"3. LOC_val_solution.csv: {tot3} rows -> {bad3} mismatches vs official ground truth")

# --- 2. train: both copies identical?
diff_names = diff_sizes = n_checked = n_files = 0
for wnid in sorted(os.listdir(f"{A}/train")):
    fa, fb = sorted(os.listdir(f"{A}/train/{wnid}")), sorted(os.listdir(f"{B}/train/{wnid}"))
    n_files += len(fa)
    if fa != fb: diff_names += 1; continue
    for f in fa[:15]:
        n_checked += 1
        diff_sizes += os.path.getsize(f"{A}/train/{wnid}/{f}") != os.path.getsize(f"{B}/train/{wnid}/{f}")
print(f"2. train: {n_files} files; classes with different file lists: {diff_names}/1000; "
      f"sampled {n_checked} files, size mismatches: {diff_sizes}")
