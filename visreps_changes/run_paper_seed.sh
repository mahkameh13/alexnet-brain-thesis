#!/bin/bash
# Train the paper's AlexNet-style CNN (CustomCNN) on ImageNet-1k with the paper's recipe.
# Changes from the repo defaults: save a checkpoint every epoch (checkpoint_interval=1) instead of every 10,
# and log to our own W&B project (alexnet-brain, CoreWeave W&B server, same as week 1).
# Usage: ./run_paper_seed.sh <seed 1-3> <gpu>
cd /home/mahkameh/visreps
export WANDB_BASE_URL=https://api.forge.coreweave.com
export WANDB_PROJECT=alexnet-brain
CUDA_VISIBLE_DEVICES=$2 .venv/bin/python -m visreps.run --mode train \
  --config configs/train/base.json configs/train/architectures/custom_cnn.json \
  --override seed=$1 checkpoint_interval=1 checkpoint_dir=alexnet_paper_recipe num_workers=20 use_wandb=true \
  2>&1 | tee -a logs_mahkameh/train_seed$1.log
