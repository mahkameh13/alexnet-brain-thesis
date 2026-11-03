# AlexNet vs infant/adult brain RSA

Status: Stage 5 RUNNING since 2026-11-03 01:17 on the ORIGINAL full-resolution ImageNet (/home/data/ImageNet/ILSVRC/Data/CLS-LOC), 3 seeds in parallel on GPUs 0,1,2, 24 loader workers each. Supervisor in tmux session alexnet_pipeline.
Stages: 1 inspect -> 2 brain data + stimuli -> 3 RSA with pretrained AlexNet -> 4 training script -> 5 three seeded runs + RSA over checkpoints.
Logging: local logs in logs/; Weights & Biases planned (needs user login, see below).

## How to run
- Stage 3: `WANDB_BASE_URL=https://api.forge.coreweave.com ./venv/bin/python stage3_pretrained_rsa.py --wandb --project alexnet-brain` (CPU only; helper code in rsa_lib.py)
- Note: brain matrices in ds006883 are stored as similarities (verified: same-category pairs higher); rsa_lib flips the sign.

## Check status (while the experiment runs)
- `tmux ls` : sessions alexnet_seed0/1/2 (training) and alexnet_pipeline (supervisor). Look inside: `tmux attach -t alexnet_seed0` (detach with Ctrl-b then d).
- `tail logs/auto_pipeline.log` : supervisor decisions (launch, stop, restart, finish).
- `tail logs/train_alexnet_seed0.jsonl` : per-epoch loss/accuracy and checkpoint paths. Same data in W&B project alexnet-brain.
- When everything ends the supervisor runs RSA on all checkpoints and writes: results/SUMMARY.md, results/rsa_alexnet_seed*.csv, results/rsa_across_training_*.png
- Stop everything by hand: `tmux kill-server`  (training can be resumed later with ./run_seed.sh <seed> <gpu>).
- Checkpoints: checkpoints/alexnet_seed<N>/ (244 MB each, ~14 GB total). Delete them when no longer needed.
- Environment note: PyTorch must be the CUDA 12.6 build (V100 GPUs); see requirements.txt.

## Dataset note
Training first started on /home/data/ILSVRC_real (a 256-px-shorter-side resized copy, labels verified identical to the official devkit). At Mahkameh's request the runs were restarted on the ORIGINAL full-resolution ImageNet: /home/data/ImageNet/ILSVRC/Data/CLS-LOC (val labels from /home/data/ImageNet/LOC_val_solution.csv). Old logs: logs/old_run_256px_resized/.
