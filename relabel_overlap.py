"""TODO 3: compare WordNet-based relabeling with PCA-based relabeling (labels from the paper's repo).

For each pair (WordNet with N classes, PCA with M classes) we build an N x M table:
    count[i, j] = number of ImageNet training images that WordNet puts in category i AND PCA puts in category j.
From it we plot a heatmap of the Jaccard overlap
    jaccard[i, j] = count[i, j] / (size of WordNet cat i + size of PCA cat j - count[i, j])
(1 = the two categories contain exactly the same images, 0 = no image in common).
Rows/columns are the category ids 0..N-1 / 0..M-1, in their original order.
We also report one summary number per pair: adjusted Rand index (ARI; 0 = chance agreement, 1 = identical partitions).

WordNet labels exist for N = 2, 3, 4, 10, 20, 57; PCA labels for powers of two. Each WordNet N is paired with the
PCA N that is equal or closest (on a log scale).

Usage: python relabel_overlap.py --labels-dir ~/visreps/pca_labels --pca-sources alexnet clip
Outputs: results/relabel_overlap/
"""
import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

p = argparse.ArgumentParser()
p.add_argument("--labels-dir", default=os.path.expanduser("~/visreps/pca_labels"))
p.add_argument("--pca-sources", nargs="+", default=["alexnet", "clip"])
p.add_argument("--out", default="results/relabel_overlap")
args = p.parse_args()
os.makedirs(args.out, exist_ok=True)

WORDNET_N = [2, 3, 4, 10, 20, 57]
PCA_N = [2, 4, 8, 16, 32, 64]
PAIRS = {2: 2, 3: 4, 4: 4, 10: 8, 20: 16, 57: 64}  # WordNet N -> closest PCA N (log scale)


def load(folder, n):
    df = pd.read_csv(f"{args.labels_dir}/pca_labels_{folder}/n_classes_{n}.csv")
    return df.set_index("image")["pca_label"]


wordnet = {n: load("wordnet", n) for n in WORDNET_N}
summary = []
for src in args.pca_sources:
    pca = {m: load(src, m) for m in PCA_N}
    # ARI for every (WordNet N, PCA M) combination -> small overview grid
    ari = np.zeros((len(WORDNET_N), len(PCA_N)))
    for a, n in enumerate(WORDNET_N):
        for b, m in enumerate(PCA_N):
            w, q = wordnet[n].align(pca[m], join="inner")
            ari[a, b] = adjusted_rand_score(w.values, q.values)
            summary.append(dict(pca_source=src, wordnet_n=n, pca_n=m, n_images=len(w), ari=ari[a, b]))

    # one heatmap per WordNet N, against its paired PCA N
    fig, axes = plt.subplots(2, 3, figsize=(17, 11))
    for ax, n in zip(axes.flat, WORDNET_N):
        m = PAIRS[n]
        w, q = wordnet[n].align(pca[m], join="inner")
        count = pd.crosstab(w, q).reindex(index=range(n), columns=range(m), fill_value=0).values
        jac = count / (count.sum(1, keepdims=True) + count.sum(0, keepdims=True) - count)
        pd.DataFrame(count).to_csv(f"{args.out}/counts_wordnet{n}_{src}{m}.csv")
        im = ax.imshow(jac, cmap="viridis", vmin=0, vmax=max(jac.max(), 1e-9), aspect="auto")
        if n * m <= 64:  # write the numbers in small heatmaps
            for i in range(n):
                for j in range(m):
                    ax.text(j, i, f"{jac[i, j]:.2f}", ha="center", va="center", fontsize=8,
                            color="white" if jac[i, j] < 0.6 * jac.max() else "black")
        ax.set_xticks(range(m) if m <= 16 else range(0, m, 8))
        ax.set_yticks(range(n) if n <= 20 else range(0, n, 8))
        ax.set_xlabel(f"PCA ({src}) category, N = {m}")
        ax.set_ylabel(f"WordNet category, N = {n}")
        ari_val = adjusted_rand_score(w.values, q.values)
        ax.set_title(f"WordNet {n} vs PCA-{src} {m}   (ARI = {ari_val:.3f})")
        fig.colorbar(im, ax=ax, fraction=0.046, label="Jaccard overlap")
    fig.suptitle(f"Overlap between WordNet and PCA-{src} relabeling of ImageNet training images "
                 f"(cell = |both| / |either|)", fontsize=13)
    fig.tight_layout()
    fig.savefig(f"{args.out}/overlap_heatmaps_wordnet_vs_{src}.png", dpi=130)
    plt.close(fig)

    # overview: ARI for all combinations
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    im = ax.imshow(ari, cmap="magma", vmin=0)
    for a in range(len(WORDNET_N)):
        for b in range(len(PCA_N)):
            ax.text(b, a, f"{ari[a, b]:.3f}", ha="center", va="center", fontsize=8,
                    color="white" if ari[a, b] < 0.6 * ari.max() else "black")
    ax.set_xticks(range(len(PCA_N)), PCA_N); ax.set_yticks(range(len(WORDNET_N)), WORDNET_N)
    ax.set_xlabel(f"PCA ({src}) number of classes"); ax.set_ylabel("WordNet number of classes")
    ax.set_title(f"Agreement WordNet vs PCA-{src} (adjusted Rand index)")
    fig.colorbar(im, ax=ax, label="ARI (0 = chance, 1 = identical)")
    fig.tight_layout(); fig.savefig(f"{args.out}/ari_grid_wordnet_vs_{src}.png", dpi=130); plt.close(fig)

pd.DataFrame(summary).to_csv(f"{args.out}/ari_summary.csv", index=False)
print(pd.DataFrame(summary).pivot_table(index=["pca_source", "wordnet_n"], columns="pca_n", values="ari").round(3))
print("images per WordNet category (N=2,3,4):")
for n in [2, 3, 4]:
    print(n, wordnet[n].value_counts().sort_index().tolist())
