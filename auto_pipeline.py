"""Supervisor for the whole experiment: 3 seeded AlexNet runs in parallel -> RSA on all checkpoints -> plots.

Started once inside tmux (session 'alexnet_pipeline'):  ./venv/bin/python auto_pipeline.py
Safe to restart: it sees which runs already finished / are alive and carries on.

Safety rules (shared cluster):
  * one GPU per run: seed N -> GPU N (0,1,2); GPU 3 is never used. A run is only (re)launched if its GPU has < 1 GB in use.
  * a run still at chance level after 2 epochs is STOPPED (frees the GPU) and flagged in results/STUCK_seedN.txt
  * a crashed run is relaunched from its last saved epoch (resume) at most 3 times
  * everything is stopped if free space on /home falls below 40 GB
"""
import json
import os
import shutil
import subprocess
import time

os.chdir("/home/mahkameh/alexnet_brain")
SEEDS = {0: 0, 1: 1, 2: 2}          # seed -> GPU index
EPOCHS = 20
PY = "./venv/bin/python"
MAX_RESTARTS, MIN_FREE_GB, POLL_S, RSA_EVERY_S = 3, 40, 120, 1800
restarts = {s: 0 for s in SEEDS}
stuck, failed = set(), set()
os.makedirs("results", exist_ok=True)


def say(msg):
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open("logs/auto_pipeline.log", "a") as f:
        f.write(line + "\n")


def sh(cmd):
    return subprocess.run(cmd, shell=True, capture_output=True, text=True)


def events(seed):
    path = f"logs/train_alexnet_seed{seed}.jsonl"
    out = []
    if os.path.exists(path):
        for l in open(path):
            try:
                out.append(json.loads(l))
            except Exception:
                pass
    return out


def finished(seed):
    return any(e.get("event") == "finished" for e in events(seed))


def alive(seed):
    return sh(f"tmux has-session -t alexnet_seed{seed}").returncode == 0


def gpu_mem_mb(gpu):
    r = sh(f"nvidia-smi --id={gpu} --query-gpu=memory.used --format=csv,noheader,nounits")
    return int(r.stdout.strip() or 99999)


def launch(seed):
    gpu = SEEDS[seed]
    if gpu_mem_mb(gpu) > 1000:
        say(f"seed {seed}: GPU {gpu} is busy ({gpu_mem_mb(gpu)} MiB used by someone else) - not launching, will retry")
        return False
    sh(f"tmux new-session -d -s alexnet_seed{seed} './run_seed.sh {seed} {gpu}'")
    say(f"seed {seed}: launched on GPU {gpu} (restart #{restarts[seed]})")
    return True


def stop(seed, why):
    sh(f"tmux kill-session -t alexnet_seed{seed}")
    say(f"seed {seed}: STOPPED - {why}")


def run_rsa(wandb=False):
    for s in SEEDS:
        if os.path.isdir(f"checkpoints/alexnet_seed{s}"):
            sh(f"CUDA_VISIBLE_DEVICES= WANDB_BASE_URL=https://api.forge.coreweave.com {PY} rsa_checkpoints.py --seed {s}"
               f"{' --wandb --project alexnet-brain' if wandb else ''} >> logs/rsa_alexnet_seed{s}.log 2>&1")


say("pipeline supervisor started")
last_rsa = time.time()
while True:
    free_gb = shutil.disk_usage("/home").free / 1e9
    if free_gb < MIN_FREE_GB:
        for s in SEEDS:
            stop(s, f"/home has only {free_gb:.0f} GB free (limit {MIN_FREE_GB})")
        failed |= set(SEEDS)
        break
    for s in SEEDS:
        if s in stuck or s in failed or finished(s):
            continue
        ev = events(s)
        ep = [e for e in ev if e.get("event") == "epoch"]
        if len(ep) >= 2 and ep[-1]["epoch"] >= 2 and min(e["train_loss"] for e in ep[1:3]) > 6.90:
            stop(s, f"still at chance after {ep[-1]['epoch']} epochs (train_loss {ep[-1]['train_loss']:.3f})")
            open(f"results/STUCK_seed{s}.txt", "w").write("Run stopped by supervisor: loss did not leave chance level (6.908) after 2 epochs.\n")
            stuck.add(s)
            continue
        if not alive(s):
            if restarts[s] >= MAX_RESTARTS + 0:
                say(f"seed {s}: no longer running and restart limit reached - giving up")
                failed.add(s)
            else:
                if launch(s):
                    restarts[s] += 1 if os.path.exists(f"logs/stdout_alexnet_seed{s}.log") and events(s) else 0
    if all(finished(s) or s in stuck or s in failed for s in SEEDS):
        break
    if time.time() - last_rsa > RSA_EVERY_S:
        run_rsa(); last_rsa = time.time(); say("intermediate RSA on checkpoints so far done")
    time.sleep(POLL_S)

say(f"training phase over. finished={[s for s in SEEDS if finished(s)]} stuck={sorted(stuck)} failed={sorted(failed)}")
run_rsa(wandb=True)
r = sh(f"{PY} plot_rsa_checkpoints.py")
say("RSA + plots done:\n" + (r.stdout + r.stderr)[-600:])
with open("results/SUMMARY.md", "w") as f:
    f.write("# Automatic run summary\n\n")
    for s in SEEDS:
        ep = [e for e in events(s) if e.get("event") == "epoch"]
        last = ep[-1] if ep else {}
        f.write(f"- seed {s}: finished={finished(s)} stuck={s in stuck} failed={s in failed}; epochs done={len(ep)}; "
                f"final val_acc={last.get('val_acc', 'n/a')}\n")
    f.write("\nPlots: results/rsa_across_training_*.png ; numbers: results/rsa_alexnet_seed*.csv ; W&B project alexnet-brain\n")
say("ALL DONE - see results/SUMMARY.md")
