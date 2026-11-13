#!/bin/bash
# Runs alongside the 3 paper-recipe trainings (TODO 1+2), no manual steps:
# every few minutes it runs RSA on any new checkpoint (CPU, ~2 s each) and redraws the plots,
# so results/rsa_paper_vs_old_*.png is always up to date. It stops once all seeds have epoch 20,
# or once no training process is left (then it logs which seeds are incomplete).
cd /home/mahkameh/alexnet_brain
CK=/home/mahkameh/visreps/model_checkpoints/alexnet_paper_recipe
PY=/home/mahkameh/visreps/.venv/bin/python

update() {
  for s in 1 2 3; do CUDA_VISIBLE_DEVICES= $PY rsa_checkpoints_paper.py --seed $s || echo "RSA seed $s failed (will retry)"; done
  $PY plot_rsa_paper_vs_old.py && $PY plot_rsa_paper_vs_old.py --no-old
}

while true; do
  all_done=1
  for d in a b c; do [ -f $CK/cfg1000$d/checkpoint_epoch_20.pth ] || all_done=0; done
  training_alive=$(pgrep -fc "visreps.run --mode train.*checkpoint_dir=alexnet_paper_recipe")
  if [ $all_done = 1 ] || [ "$training_alive" = 0 ]; then
    sleep 60  # let the last checkpoint write finish
    update
    break
  fi
  update
  echo "[$(date '+%H:%M')] updated; waiting for next checkpoints"
  sleep 300
done

for d in a b c; do [ -f $CK/cfg1000$d/checkpoint_epoch_20.pth ] || echo "WARNING: cfg1000$d stopped before epoch 20 — check ~/visreps/logs_mahkameh/"; done

# Copy final RSA results + plots to W&B (project alexnet-brain). Retries if the W&B server is down.
export WANDB_BASE_URL=https://api.forge.coreweave.com WANDB_PROJECT=alexnet-brain
for try in 1 2 3 4 5 6; do
  $PY log_paper_rsa_wandb.py && { echo "W&B upload done"; break; }
  echo "W&B upload failed (try $try/6), retrying in 10 min"; sleep 600
  [ $try = 6 ] && echo "WARNING: W&B upload gave up; local results in results/ are complete. Rerun: $PY log_paper_rsa_wandb.py"
done
echo "ALL DONE $(date)"
