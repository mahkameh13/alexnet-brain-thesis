"""Check ImageNet images for corruption by fully decoding them (read-only on the dataset).

Usage: python check_images.py --split train --per-class 10 --workers 4     (quick sample)
       python check_images.py --split train --workers 8                    (everything)
Writes the list of bad files to logs/corrupt_<split>.txt
"""
import argparse
import os
import warnings
from multiprocessing import Pool

from PIL import Image

p = argparse.ArgumentParser()
p.add_argument("--root", default="/home/data/ILSVRC_real")
p.add_argument("--split", default="train")
p.add_argument("--per-class", type=int, default=0)
p.add_argument("--workers", type=int, default=4)
args = p.parse_args()
warnings.simplefilter("error")  # turn PIL warnings (e.g. truncated EXIF/data) into errors we can see


def check(path):
    try:
        with Image.open(path) as im:
            im.convert("RGB").load()  # full decode, same as training
        return None
    except Exception as e:
        return f"{path}\t{type(e).__name__}: {e}"


if __name__ == "__main__":
    base = os.path.join(args.root, args.split)
    paths = []
    for c in sorted(os.listdir(base)):
        files = sorted(os.listdir(os.path.join(base, c)))
        paths += [os.path.join(base, c, f) for f in (files[:args.per_class] if args.per_class else files)]
    print(f"checking {len(paths)} images in {args.split} with {args.workers} workers", flush=True)
    bad = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(check, paths, chunksize=64), 1):
            if r:
                bad.append(r)
            if i % 50000 == 0:
                print(f"  {i}/{len(paths)} checked, {len(bad)} bad so far", flush=True)
    out = f"logs/corrupt_{args.split}{'_sample' if args.per_class else ''}.txt"
    with open(out, "w") as fh:
        fh.write("\n".join(bad) + ("\n" if bad else ""))
    print(f"DONE: {len(paths)} images, {len(bad)} problems -> {out}", flush=True)
