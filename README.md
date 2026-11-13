# Developmental alignment with coarse objectives

Bachelor thesis code (Mahkameh). Question: does training make an image network's representations more like infant (2-month, 9-month) or adult brains?

Method (RSA): show the 36 FOUNDCOG stimulus images to a network in eval mode, read 7 layers (conv1-conv5, fc1, fc2, after ReLU), build a 36x36 model RDM (1 - Pearson r), and Spearman-correlate it with the brain RDMs of each age group and region (VVC, LOC, V1). Brain RDMs: OpenNeuro ds006883 derivatives (O'Doherty et al., Nature Neuroscience 2025). Repeat for every training checkpoint to get curves over training.

Reference paper: Mehta & Bonner, arXiv 2605.05556. Its code: github.com/yashsmehta/visreps (we use commit 43ba5e2).

## Weekly updates

**Week of 26 Oct.** Direction agreed with advisors: infant fMRI data, AlexNet on ImageNet-1k with 3 seeds, coarse-to-fine curriculum later.

**Week 1.** RSA pipeline built and checked on pretrained torchvision AlexNet. Our own AlexNet trained 3 times for 20 epochs on full-resolution ImageNet (top-1 50.5% on the official val set), RSA on 17 checkpoints per seed. Training makes the network more brain-like, and more like 9-month-olds and adults than 2-month-olds (VVC, fc2: rho about 0.45 / 0.43 / 0.28).

**Week 2 (advisor TODOs from 03/11/2026).**
1. Retrained with the paper's own code and recipe (CustomCNN, 3 seeds, 20 epochs, a checkpoint every epoch). Top-1 54.2-55.1% on the repo's held-out 20% split. RSA recomputed with the same pipeline.
2. Plots over training: mean +- 1 std across seeds, week-1 run as a dashed shadow, pretrained AlexNet dotted (`results/rsa_paper_vs_old_*.png`). The paper's CNN matches VVC best early in training (about epoch 3), then its late layers drift away. 2-month-olds remain the least matched group.
3. WordNet vs PCA label overlap (`results/relabel_overlap/`): Jaccard heatmaps and adjusted Rand index. WordNet N = 2-4 are extremely unbalanced; best agreement WordNet 10 vs CLIP-PCA 2 (ARI 0.61).

## Files

| File | What it does |
|---|---|
| `rsa_lib.py` | Shared toolbox: stimuli, brain RDMs (stored as similarities, sign flipped), layer activations, model RDM, Spearman, bootstrap CI |
| `stage3_pretrained_rsa.py` | Sanity check on pretrained torchvision AlexNet |
| `train.py`, `run_seed.sh` | Week 1 training (torchvision AlexNet, SGD, 20 epochs) |
| `auto_pipeline.py` | Week 1 supervisor: runs 3 seeds, restarts failed runs, then RSA and plots |
| `rsa_checkpoints.py`, `plot_rsa_checkpoints.py` | Week 1 RSA over checkpoints and figures |
| `verify_imagenet.py`, `check_images.py`, `inspect_rdms*.py` | Data checks |
| `rsa_checkpoints_paper.py` | Week 2 RSA on the paper-recipe checkpoints |
| `plot_rsa_paper_vs_old.py` | Week 2 figures (TODO 2) |
| `relabel_overlap.py` | WordNet vs PCA label comparison (TODO 3) |
| `after_paper_training.sh`, `log_paper_rsa_wandb.py` | Week 2 automation: RSA + plots while training, then upload to W&B |
| `visreps_changes/` | Our launch script for the paper's code and the patch of our logging-only edits to visreps |
| `results/` | CSVs and figures for both weeks |

Not in git (too large or not ours): `venv/`, `data/` (brain data), `checkpoints/`, `logs/`, `wandb/`. Week-2 checkpoints live in the visreps clone on the server. All runs are also logged in the W&B project `alexnet-brain`.

## Reproduce week 2
1. `git clone https://github.com/yashsmehta/visreps && cd visreps && git checkout 43ba5e2 && git apply ../alexnet_brain/visreps_changes/visreps_logging.patch`
2. Copy `visreps_changes/run_paper_seed.sh` into the visreps folder and run `./run_paper_seed.sh <seed 1-3> <gpu>`.
3. In this folder: `./after_paper_training.sh` (or run `rsa_checkpoints_paper.py --seed N` and `plot_rsa_paper_vs_old.py` by hand).

Environment: Python 3.12, PyTorch CUDA 12.6 build (V100 GPUs), see `requirements.txt`.
