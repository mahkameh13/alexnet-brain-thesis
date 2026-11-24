#!/bin/bash
# Train torchvision AlexNet (the week 1 network) with the reference paper's code and training recipe:
# AdamW lr 5e-4, wd 1e-3 (weights only), batch 32, 2 warmup epochs + cosine to 5%, label smoothing 0.1, grad clip 1.0,
# AMP, the paper's mild augmentation (forced: the paper code switches to RandomResizedCrop for non-CustomCNN models),
# the paper's 80/20 split, 20 epochs, a checkpoint every epoch. Only the network differs from the week 2 run.
# Usage: ./run_alexnet_paper_seed.sh <seed 1-3> <gpu>
cd /home/mahkameh/visreps
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_PROJECT=alexnet-brain
export WANDB_RUN_NAME=alexnet_paper_recipe_seed$1
CUDA_VISIBLE_DEVICES=$2 .venv/bin/python -m visreps.run --mode train \
  --config configs/train/base.json /home/mahkameh/alexnet_brain/visreps_changes/alexnet_paper_recipe.json \
  --override seed=$1 checkpoint_interval=1 checkpoint_dir=alexnet_torchvision_paper_recipe num_workers=18 use_wandb=true \
  >> logs_mahkameh/train_alexnet_paper_seed$1.log 2>&1
