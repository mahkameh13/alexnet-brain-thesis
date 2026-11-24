# alexnet-brain-thesis

Measures how similar a CNN's internal representations are to infant and adult visual cortex, and how that similarity changes over the course of ImageNet training.

For each training checkpoint, the pipeline runs 36 stimulus images through the network, builds a representational dissimilarity matrix (RDM) for 7 layers, and compares each one with fMRI RDMs from 2-month-olds, 9-month-olds and adults (FOUNDCOG, OpenNeuro `ds006883`) in three regions (VVC, LOC, V1). The output is a curve of brain alignment (Spearman ρ) against training time for each layer, region and age group.

Three training setups are supported:

| Setup | Model | Trained with | Seeds | Checkpoint format |
|---|---|---|---|---|
| week 1 (`ours`) | torchvision `alexnet` (random init) | `train.py` in this repo | 0–2 | `checkpoints/alexnet_seed<N>/ckpt_step<S>_epoch<E>.pt` (state_dict) |
| week 2 (`paper`) | `CustomCNN` (AlexNet with BatchNorm) | [yashsmehta/visreps](https://github.com/yashsmehta/visreps) @ `43ba5e2` + `visreps_changes/` | 1–3 | `~/visreps/model_checkpoints/alexnet_paper_recipe/cfg1000<a\|b\|c>/checkpoint_epoch_<E>.pth` (pickled model) |
| week 3 (`alexnet_paper`) | torchvision `alexnet` (random init) | visreps, same recipe as week 2 | 1–3 | `~/visreps/model_checkpoints/alexnet_torchvision_paper_recipe/cfg1000<a\|b\|c>/checkpoint_epoch_<E>.pth` (pickled model) |

The week 2 and week 3 setups use the training recipe of Mehta & Bonner (arXiv 2605.05556). Weeks 1 and 3 use the same network but different code and recipes. Weeks 2 and 3 use the same code and recipe but different networks. Comparing all three shows whether a result comes from the network or from the recipe.

## Pipeline

```
data/stimuli_openneuro/*.png ──► load_stimuli()  (alpha → white, resize 256, center-crop 224, ImageNet norm)
                                      │
checkpoint ──► extract_activations() ─┘  forward hooks on 7 post-ReLU layers: conv1-5, fc1, fc2
                                      │
                               model_rdm()       36×36, 1 − Pearson r
                                      │
data/brain_rdms/*.pickle ──► load_brain_rdms()   mean over subject pairs, sign-flipped (stored as similarity)
                                      │
                                   rsa()         Spearman ρ on the 630 upper-triangle entries
                                      │
                         results/rsa_*_seed<N>.csv ──► plot_*.py ──► results/*.png
```

All of the shared logic is in `rsa_lib.py`. Every other script is a thin driver around it.

## Repository layout

```
rsa_lib.py                  core: stimuli, activations, RDMs, Spearman, bootstrap CI, layer maps
stage3_pretrained_rsa.py    sanity check of the pipeline on pretrained torchvision AlexNet
train.py                    AlexNet / ImageNet-1k trainer (SGD, AMP, resumable, W&B)
run_seed.sh                 launch train.py for one seed on one GPU
auto_pipeline.py            supervisor: 3 seeds in parallel, restarts crashes, then RSA + plots
rsa_checkpoints.py          RSA over all checkpoints of a train.py run
rsa_checkpoints_paper.py    RSA over all checkpoints of a visreps run (--model customcnn|alexnet, --bn checkpoint|imagenet)
plot_rsa_checkpoints.py     figures for train.py runs
plot_rsa_paper_vs_old.py    figures for CustomCNN runs (train.py run overlaid as reference)
after_paper_training.sh     polls for new CustomCNN checkpoints → RSA → plots → W&B upload
log_paper_rsa_wandb.py      upload CustomCNN RSA CSVs and figures to W&B
relabel_overlap.py          WordNet vs PCA coarse-label agreement (Jaccard heatmaps, ARI)
calibrate_bn_imagenet.py    re-estimate CustomCNN BatchNorm statistics on 512,000 ImageNet images (visreps recipe)
eval_imagenet_checkpoints.py  ImageNet train / val / held-out accuracy and loss for every checkpoint of a setup
live_eval_alexnet_paper.py  same evaluation, run on each new week 3 checkpoint while training is still going
plot_imagenet_curves.py     accuracy and loss over training, train vs val (vs held-out), per setup
plot_rsa_bn_compare.py      RSA over training, BatchNorm from the checkpoint vs re-estimated on ImageNet
plot_rsa_bn_difference.py   Δρ (re-estimated − checkpoint) per epoch, region, age group and layer
plot_rsa_three_setups.py    RSA over training, the three setups overlaid
plot_wandb_training_logs.py charts of the metrics W&B logged during training (weeks 1 and 2)
log_todo_1711_wandb.py      upload the week 3 ImageNet-eval and BatchNorm-RSA results to W&B
verify_imagenet.py          read-only checks that the ImageNet copy is genuine ILSVRC2012
check_images.py             decode every image to find corrupt files
inspect_rdms*.py            print the structure of the FOUNDCOG RDM pickles
visreps_changes/            launch script + patch for the visreps repo (logging/checkpointing only)
results/                    committed CSVs and PNGs
```

## Setup

Requires Python 3.12. The GPUs used are V100s, and the default CUDA 13 PyTorch wheels don't support them, so install from the cu126 index:

```bash
python3.12 -m venv venv
./venv/bin/pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu126
```

### Data (not in git)

```
data/
  ds006883/          git clone https://github.com/OpenNeuroDatasets/ds006883  (git-annex; fetch files with `datalad get`)
  brain_rdms/        copy of ds006883/derivatives/foundcog_rdms/*exemplar-level_correlation-distance_julich-rois.pickle
  stimuli_openneuro/ copy of ds006883/stimuli/*.png
```

Both paths are hardcoded in `rsa_lib.py` (`RDM_DIR`, `STIM_DIR`) and are relative to the repo root, so run every script from there.

You only need ImageNet-1k (ILSVRC2012, full resolution) for training. The defaults in `train.py` point to `/home/data/ImageNet/ILSVRC/Data/CLS-LOC` and `/home/data/ImageNet/LOC_val_solution.csv`. Override them with `--data-root` and `--val-csv`.

### W&B

W&B logging is optional. The scripts default to a self-hosted server (`WANDB_BASE_URL=https://api.forge.coreweave.com`, project `alexnet-brain`). To use your own setup, set `WANDB_BASE_URL`, `WANDB_PROJECT` and `WANDB_ENTITY`. Pass `--no-wandb` to `train.py` to turn logging off.

## Usage

### 1. Sanity check (CPU)

```bash
./venv/bin/python stage3_pretrained_rsa.py        # → results/stage3_pretrained_rsa.{csv,png}
```

### 2. torchvision AlexNet runs

```bash
./run_seed.sh <seed> <gpu>                        # 20 epochs; re-running resumes from checkpoints/.../last.pt
# or all 3 seeds with crash recovery (meant to run inside tmux):
./venv/bin/python auto_pipeline.py

./venv/bin/python rsa_checkpoints.py --seed 0     # → results/rsa_alexnet_seed0.csv
./venv/bin/python plot_rsa_checkpoints.py         # → results/rsa_across_training_*.png
```

Useful `train.py` flags: `--ckpt-steps 0,100,500,2000` and `--ckpt-epochs 1,2,...,20` choose which checkpoints get saved (step 0 is the untrained init). `--subset-per-class N` and `--val-subset-per-class N` run a smoke test on a small subset.

### 3. visreps CustomCNN runs

```bash
git clone https://github.com/yashsmehta/visreps ~/visreps
cd ~/visreps && git checkout 43ba5e2
git apply ~/alexnet_brain/visreps_changes/visreps_logging.patch
cp ~/alexnet_brain/visreps_changes/run_paper_seed.sh . && ./run_paper_seed.sh <seed 1-3> <gpu>

cd ~/alexnet_brain
./after_paper_training.sh                         # polls every 5 min, stops when all seeds reach epoch 20
# or by hand:
~/visreps/.venv/bin/python rsa_checkpoints_paper.py --seed 1
~/visreps/.venv/bin/python plot_rsa_paper_vs_old.py [--no-old]
```

`rsa_checkpoints_paper.py` must run with the visreps venv, because the checkpoints are pickled model objects and unpickling them imports the `visreps` package.

### 4. Label overlap

```bash
./venv/bin/python relabel_overlap.py --labels-dir ~/visreps/pca_labels   # → results/relabel_overlap/
```

### 5. torchvision AlexNet with the paper's code and recipe (week 3)

```bash
cp ~/alexnet_brain/visreps_changes/run_alexnet_paper_seed.sh ~/visreps/
cd ~/visreps && ./run_alexnet_paper_seed.sh <seed 1-3> <gpu>    # config: visreps_changes/alexnet_paper_recipe.json

cd ~/alexnet_brain
CUDA_VISIBLE_DEVICES=0 ~/visreps/.venv/bin/python live_eval_alexnet_paper.py   # optional: eval each epoch as it appears
~/visreps/.venv/bin/python rsa_checkpoints_paper.py --model alexnet --seed 1   # → results/rsa_alexnet_paper_recipe_seed1.csv
```

The config forces the paper's "mild" augmentation. Without it, the paper's code switches to `RandomResizedCrop` for any model other than CustomCNN.

### 6. BatchNorm re-estimated on ImageNet (week 3)

FOUNDCOG has only the 36 test images, so the paper's default, re-estimating BatchNorm on the neural dataset's training images, isn't possible here. Instead we use the paper code's ImageNet option: reset the running statistics, then take one cumulative-average pass over a fixed random subset of 512,000 images from the paper's training split, in batches of 256.

```bash
CUDA_VISIBLE_DEVICES=1 ~/visreps/.venv/bin/python calibrate_bn_imagenet.py --seed 1   # → cache/bn_imagenet/paper_seed1_epoch<E>.pt
~/visreps/.venv/bin/python rsa_checkpoints_paper.py --seed 1 --bn imagenet           # → results/rsa_paper_bnimagenet_seed1.csv
./venv/bin/python plot_rsa_bn_compare.py && ./venv/bin/python plot_rsa_bn_difference.py
```

### 7. ImageNet accuracy and loss over training (week 3)

```bash
CUDA_VISIBLE_DEVICES=1 ~/visreps/.venv/bin/python eval_imagenet_checkpoints.py --setup ours --seed 0
CUDA_VISIBLE_DEVICES=1 ~/visreps/.venv/bin/python eval_imagenet_checkpoints.py --setup paper --seed 1 [--bn imagenet] [--splits heldout]
CUDA_VISIBLE_DEVICES=1 ~/visreps/.venv/bin/python eval_imagenet_checkpoints.py --setup alexnet_paper --seed 1
./venv/bin/python plot_imagenet_curves.py          # → results/imagenet_curves_<setup>_{acc,loss}.png
./venv/bin/python plot_rsa_three_setups.py         # → results/rsa_three_setups_<region>.png, ..._fc2_summary.png
```

All setups are measured the same way, so the curves are comparable:
- **Network mode:** eval mode, fp32, resize 256 + center crop 224.
- **val:** the 50,000 official ILSVRC2012 validation images.
- **train:** a fixed sample of 50,000 images from the paper's 80% training split.
- **heldout:** a fixed sample of 50,000 images from the paper's 20% held-out split, the split the paper reports accuracy on. Only meaningful for the visreps setups, because `train.py` trained on those images.

Each image is decoded once and fed to every checkpoint of the seed.

## Output format

`results/rsa_alexnet_seed<N>.csv` and `results/rsa_paper_seed<N>.csv` are long-format tables with one row per (checkpoint, region, age group, layer):

```
seed,epoch,region,group,layer,rho
1,0,LOC,adult,conv1,0.0361
```

The torchvision runs also have a `step` column. The RSA scripts are incremental: checkpoints already in the CSV are skipped, so you can safely re-run them while training is still going. `rsa_paper_bnimagenet_seed<N>.csv` (BatchNorm re-estimated) and `rsa_alexnet_paper_recipe_seed<N>.csv` (week 3 setup) use the same format.

`results/imagenet_eval_<setup>[_bnimagenet][_heldout]_seed<N>.csv` has one row per checkpoint:

```
setup,bn,seed,epoch,step,train_loss,train_loss_median,train_acc,train_top5,n_train,val_loss,val_loss_median,val_acc,val_top5,n_val
```

The `_heldout` files have `heldout_*` columns instead of train/val. The loss is plain cross-entropy with no label smoothing. The median is reported because, with its checkpoint BatchNorm statistics, the CustomCNN gives huge logits on a few images, and those dominate the mean. `results/live/` holds the per-epoch files that `live_eval_alexnet_paper.py` wrote during week 3 training.

## Implementation notes

- **Layer indices differ between the two models.** CustomCNN has a BatchNorm between each conv/linear layer and its ReLU, so the hook positions are not the same. See `ALEXNET_LAYERS` and `CUSTOMCNN_LAYERS` in `rsa_lib.py`. The logits layer is excluded.
- **The brain matrices are similarities**, despite the `correlation-distance` filename: same-category pairs score higher, which I checked empirically. `load_brain_rdms()` negates them. Spearman only uses ranks, so the sign flip is the only correction needed.
- **The stimuli have transparent backgrounds.** They are composited onto white before conversion to RGB. A plain `.convert("RGB")` would give a black background. The images are 16:9, so the 224 center crop cuts off part of the left and right edges.
- **Bootstrap CIs** (`rsa_bootstrap_ci`) resample the 36 images and drop any pair made of the same image twice, so duplicated pairs can't inflate ρ.
- Activations are extracted in `eval()` mode, so dropout is off and RDMs are deterministic.
- RSA runs on CPU (about 2 s per checkpoint) with `torch.set_num_threads(4)`.

## Results so far

ImageNet top-1 at epoch 20, mean ± std over 3 seeds, measured with `eval_imagenet_checkpoints.py`:

| Setup | Train sample | Official val | Paper's held-out split |
|---|---|---|---|
| week 1: torchvision AlexNet, `train.py` | 59.9 ± 0.2% | 50.6 ± 0.1% | — |
| week 2: CustomCNN, paper code, BatchNorm from checkpoint | 75.9 ± 0.6% | 51.7 ± 0.3% | 54.5 ± 0.5% |
| week 2: CustomCNN, BatchNorm re-estimated on ImageNet | 77.0 ± 0.2% | 51.6 ± 0.2% | 54.4 ± 0.1% |
| week 3: torchvision AlexNet, paper code and recipe | 18.7 ± 0.4% | 16.5 ± 0.5% | 17.6 ± 0.5% |

VVC alignment of fc2 (Spearman ρ, mean of 3 seeds), adult / 9-month / 2-month:

| Setup | Peak (epoch) | Epoch 20 |
|---|---|---|
| week 1 | 0.49 (10) / 0.44 (10) / 0.31 (6) | 0.45 / 0.43 / 0.28 |
| week 2, BatchNorm from checkpoint | 0.41 (3) / 0.37 (5) / 0.31 (2) | 0.25 / 0.24 / 0.14 |
| week 2, BatchNorm re-estimated | 0.41 (6) / 0.35 (3) / 0.31 (2) | 0.32 / 0.27 / 0.20 |
| week 3 | 0.39 (1) / 0.35 (1) / 0.31 (1) | 0.29 / 0.30 / 0.18 |

- **Week 2 drop:** the early peak followed by a decline is still there after re-estimating BatchNorm. Re-estimation narrows the late decline but doesn't remove it.
- **Week 3 drop:** torchvision AlexNet trained with the paper's recipe peaks even earlier (epoch 1) and then declines too. The same network trained with `train.py` declines only slightly (adult 0.49 → 0.45). This points to the recipe rather than the CustomCNN architecture.
- **Week 3 accuracy:** with the paper's recipe, torchvision AlexNet reaches only ~17% top-1, far below CustomCNN under the same recipe. That weaker training is itself a confound for the RSA comparison.
- **Age groups:** in every setup, the 2-month-olds are the group the networks match least.

Figures are in `results/`. `results/relabel_overlap/ari_summary.csv` has the label-agreement numbers (best: WordNet-10 vs CLIP-PCA-2, ARI 0.61).

## Known limitations

- Several paths are absolute and assume `/home/mahkameh/...`: `run_seed.sh`, `after_paper_training.sh`, `visreps_changes/run_paper_seed.sh`, `visreps_changes/run_alexnet_paper_seed.sh`, and `~/visreps` in `rsa_checkpoints_paper.py`, `calibrate_bn_imagenet.py` and `eval_imagenet_checkpoints.py`. Edit these before running on another machine.
- The re-estimated BatchNorm statistics (`cache/bn_imagenet/`) and the W&B training-log export that `plot_wandb_training_logs.py` reads (`results/wandb/`) are gitignored, so you have to regenerate them.
- `auto_pipeline.py` has the seed→GPU mapping hardcoded (`SEEDS = {0: 0, 1: 1, 2: 2}`) and includes safety rules for a shared cluster (free-GPU and free-disk checks).
- There is no test suite. `stage3_pretrained_rsa.py` serves as the end-to-end smoke test.

## References

- Mehta & Bonner, arXiv 2605.05556. Code: https://github.com/yashsmehta/visreps
- O'Doherty et al., *Nature Neuroscience* 2025. FOUNDCOG data: OpenNeuro `ds006883`
