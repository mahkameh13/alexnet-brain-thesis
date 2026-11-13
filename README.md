# alexnet-brain-thesis

Measures how similar a CNN's internal representations are to infant and adult visual cortex, and how that similarity changes over the course of ImageNet training.

For each training checkpoint, the pipeline runs 36 stimulus images through the network, builds a representational dissimilarity matrix (RDM) for 7 layers, and compares each one with fMRI RDMs from 2-month-olds, 9-month-olds and adults (FOUNDCOG, OpenNeuro `ds006883`) in three regions (VVC, LOC, V1). The output is a curve of brain alignment (Spearman ρ) against training time for each layer, region and age group.

Two model families are supported:

| Model | Trained with | Checkpoint format |
|---|---|---|
| torchvision `alexnet` (random init) | `train.py` in this repo | `checkpoints/alexnet_seed<N>/ckpt_step<S>_epoch<E>.pt` (state_dict) |
| `CustomCNN` (AlexNet with BatchNorm) | [yashsmehta/visreps](https://github.com/yashsmehta/visreps) @ `43ba5e2` + `visreps_changes/` | `~/visreps/model_checkpoints/alexnet_paper_recipe/cfg1000<a\|b\|c>/checkpoint_epoch_<E>.pth` (pickled model) |

The second family follows the training recipe of Mehta & Bonner (arXiv 2605.05556).

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
rsa_checkpoints_paper.py    RSA over all checkpoints of a visreps CustomCNN run
plot_rsa_checkpoints.py     figures for train.py runs
plot_rsa_paper_vs_old.py    figures for CustomCNN runs (train.py run overlaid as reference)
after_paper_training.sh     polls for new CustomCNN checkpoints → RSA → plots → W&B upload
log_paper_rsa_wandb.py      upload CustomCNN RSA CSVs and figures to W&B
relabel_overlap.py          WordNet vs PCA coarse-label agreement (Jaccard heatmaps, ARI)
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

## Output format

`results/rsa_alexnet_seed<N>.csv` and `results/rsa_paper_seed<N>.csv` are long-format tables with one row per (checkpoint, region, age group, layer):

```
seed,epoch,region,group,layer,rho
1,0,LOC,adult,conv1,0.0361
```

The torchvision runs also have a `step` column. The RSA scripts are incremental: checkpoints already in the CSV are skipped, so you can safely re-run them while training is still going.

## Implementation notes

- **Layer indices differ between the two models.** CustomCNN has a BatchNorm between each conv/linear layer and its ReLU, so the hook positions are not the same. See `ALEXNET_LAYERS` and `CUSTOMCNN_LAYERS` in `rsa_lib.py`. The logits layer is excluded.
- **The brain matrices are similarities**, despite the `correlation-distance` filename: same-category pairs score higher, which I checked empirically. `load_brain_rdms()` negates them. Spearman only uses ranks, so the sign flip is the only correction needed.
- **The stimuli have transparent backgrounds.** They are composited onto white before conversion to RGB. A plain `.convert("RGB")` would give a black background. The images are 16:9, so the 224 center crop cuts off part of the left and right edges.
- **Bootstrap CIs** (`rsa_bootstrap_ci`) resample the 36 images and drop any pair made of the same image twice, so duplicated pairs can't inflate ρ.
- Activations are extracted in `eval()` mode, so dropout is off and RDMs are deterministic.
- RSA runs on CPU (about 2 s per checkpoint) with `torch.set_num_threads(4)`.

## Results so far

| Run | Val top-1 | Best VVC alignment |
|---|---|---|
| torchvision AlexNet, 3 seeds × 20 epochs | 50.5% (official val) | fc2 at epoch 20: ρ ≈ 0.45 adult / 0.43 9-month / 0.28 2-month |
| visreps CustomCNN, 3 seeds × 20 epochs | 54.2–55.1% (repo's 20% held-out split) | peaks around epoch 3, late layers drift away afterwards |

In both runs, the 2-month-olds are the age group the networks match least. Figures are in `results/`, and `results/relabel_overlap/ari_summary.csv` has the label-agreement numbers (best: WordNet-10 vs CLIP-PCA-2, ARI 0.61).

## Known limitations

- Several paths are absolute and assume `/home/mahkameh/...`: `run_seed.sh`, `after_paper_training.sh`, `visreps_changes/run_paper_seed.sh`, and `~/visreps` in `rsa_checkpoints_paper.py`. Edit these before running on another machine.
- `auto_pipeline.py` has the seed→GPU mapping hardcoded (`SEEDS = {0: 0, 1: 1, 2: 2}`) and includes safety rules for a shared cluster (free-GPU and free-disk checks).
- There is no test suite. `stage3_pretrained_rsa.py` serves as the end-to-end smoke test.

## References

- Mehta & Bonner, arXiv 2605.05556. Code: https://github.com/yashsmehta/visreps
- O'Doherty et al., *Nature Neuroscience* 2025. FOUNDCOG data: OpenNeuro `ds006883`
