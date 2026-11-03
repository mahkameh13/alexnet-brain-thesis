#!/bin/bash
# Usage: ./run_seed.sh <seed> <gpu>   -> trains AlexNet (20 epochs) on ONE GPU; safe to re-run: it resumes.
cd /home/mahkameh/alexnet_brain
export WANDB_BASE_URL=https://api.forge.coreweave.com
export CUDA_VISIBLE_DEVICES=$2
./venv/bin/python train.py --seed $1 --epochs 20 --workers 24 --project alexnet-brain 2>&1 | grep -v -i deprec | tee -a logs/stdout_alexnet_seed$1.log
