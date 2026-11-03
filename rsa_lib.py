"""Reusable pieces of the RSA (representational similarity analysis) pipeline.

Pipeline:  36 images -> model activations per layer -> model RDM
           brain matrices (per age group & region) -> brain RDM
           Spearman correlation between the two RDMs (upper triangles).

An RDM ("representational dissimilarity matrix") is a 36x36 table: entry (i, j)
says how *different* the representation of image i is from image j.
"""
import glob
import pickle
import warnings

import numpy as np
import torch
from PIL import Image
from scipy.stats import spearmanr

STIM_DIR = "data/stimuli_openneuro"
RDM_DIR = "data/brain_rdms"
AGE_GROUPS = ["twomonth", "ninemonth", "adult"]  # order used in plots

# Names of the 7 layers we read (post-ReLU), mapped to torchvision AlexNet modules.
# The final classification layer (logits) is excluded, as in the Mehta & Bonner paper.
ALEXNET_LAYERS = {
    "conv1": ("features", 1),
    "conv2": ("features", 4),
    "conv3": ("features", 7),
    "conv4": ("features", 9),
    "conv5": ("features", 11),
    "fc1": ("classifier", 2),   # first fully connected layer (fc6 in torchvision naming)
    "fc2": ("classifier", 5),   # second fully connected layer (fc7)
}


# ----------------------------------------------------------------------------- images
def load_brain_labels():
    """Image order used inside the brain matrices (stored in the pickle itself)."""
    path = glob.glob(f"{RDM_DIR}/*adult_exemplar*.pickle")[0]
    with open(path, "rb") as f:  # public dataset from the paper's authors
        return pickle.load(f)["rdm_xy_labels"]


def image_path(label):
    """'tree1' -> data/stimuli_openneuro/tree_1.png (the tree files have an underscore)."""
    if label.startswith("tree"):
        label = "tree_" + label[4:]
    return f"{STIM_DIR}/{label}.png"


def load_stimuli(labels, size=224, resize=256):
    """Load the images in brain-matrix order -> tensor (36, 3, 224, 224).

    The PNGs have a transparent background, so we first paste each image on white
    (a plain RGB conversion would make the background black). Then standard ImageNet
    preprocessing: resize shorter side to 256, center-crop 224, normalize.
    NOTE: the images are 16:9, so the center crop cuts off some of the left/right sides.
    """
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    out = []
    for lab in labels:
        im = Image.open(image_path(lab)).convert("RGBA")
        white = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(white, im).convert("RGB")
        w, h = im.size
        s = resize / min(w, h)
        im = im.resize((round(w * s), round(h * s)), Image.BICUBIC)
        w, h = im.size
        l, t = (w - size) // 2, (h - size) // 2
        im = im.crop((l, t, l + size, t + size))
        x = (np.asarray(im, dtype=np.float32) / 255.0 - mean) / std
        out.append(torch.from_numpy(x.transpose(2, 0, 1)))
    return torch.stack(out)


# --------------------------------------------------------------------------- activations
@torch.no_grad()
def extract_activations(model, images, device="cpu"):
    """Return {layer_name: array (n_images, n_features)} for the 7 AlexNet layers."""
    model = model.to(device).eval()  # eval mode: dropout OFF, so activations are deterministic
    acts, hooks = {}, []
    for name, (part, idx) in ALEXNET_LAYERS.items():
        module = getattr(model, part)[idx]
        hooks.append(module.register_forward_hook(
            lambda m, i, o, name=name: acts.__setitem__(name, o.detach().flatten(1).cpu().numpy().copy())))
    model(images.to(device))
    for h in hooks:
        h.remove()
    return acts


# ------------------------------------------------------------------------------- RDMs
def model_rdm(features):
    """Correlation distance (1 - Pearson r) between the feature vectors of each image pair."""
    f = features - features.mean(axis=1, keepdims=True)
    f = f / (np.linalg.norm(f, axis=1, keepdims=True) + 1e-12)
    return 1.0 - np.clip(f @ f.T, -1, 1)


def load_brain_rdms():
    """{group: {region: 36x36 dissimilarity matrix}}.

    The stored matrices are SIMILARITIES (same-category pairs score higher; verified
    empirically), averaged here over all subject pairs ('acrosssub'), ignoring NaNs.
    We flip the sign to get dissimilarities (Spearman only uses ranks, so the scale
    does not matter).
    """
    warnings.filterwarnings("ignore")
    out = {}
    for g in AGE_GROUPS:
        path = glob.glob(f"{RDM_DIR}/*{g}_exemplar*.pickle")[0]
        with open(path, "rb") as f:
            d = pickle.load(f)
        out[g] = {r: -np.nanmean(v["acrosssub"], axis=0)
                  for r, v in d.items() if r != "rdm_xy_labels"}
    return out


def upper(m):
    """Upper triangle without the diagonal -> vector of 630 pair values."""
    return m[np.triu_indices(m.shape[0], k=1)]


def rsa(model_m, brain_m):
    """Spearman correlation between two RDMs (upper triangles)."""
    return spearmanr(upper(model_m), upper(brain_m)).statistic


def rsa_bootstrap_ci(model_m, brain_m, n_boot=1000, seed=0):
    """95% CI by resampling the 36 images with replacement (as in the paper).
    Pairs made of the same image twice are dropped so they cannot inflate agreement."""
    rng = np.random.default_rng(seed)
    n = model_m.shape[0]
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        a, b = np.triu_indices(n, k=1)
        keep = idx[a] != idx[b]
        ia, ib = idx[a][keep], idx[b][keep]
        vals.append(spearmanr(model_m[ia, ib], brain_m[ia, ib]).statistic)
    return np.nanpercentile(vals, [2.5, 97.5])
