"""Upload the paper-recipe RSA results and plots to W&B (project alexnet-brain on the CoreWeave W&B server).

Run once after the final RSA pass (after_paper_training.sh does this). Creates:
  paper_recipe_seed<N>_rsa  one run per seed, metrics rsa/<region>/<group>/<layer> against epoch (group: paper_recipe_rsa)
  paper_recipe_rsa_summary  all three seeds as a table + every rsa_paper_vs_old_*.png plot
Local results in results/ stay the source of truth; this only copies them.
"""
import glob
import os
import time

import pandas as pd
import wandb

os.environ.setdefault("WANDB_BASE_URL", "https://api.forge.coreweave.com")
os.environ.setdefault("WANDB_PROJECT", "alexnet-brain")
stamp = time.strftime("%Y%m%d-%H%M")
tables = []

for seed in (1, 2, 3):
    df = pd.read_csv(f"results/rsa_paper_seed{seed}.csv")
    tables.append(df)
    run = wandb.init(name=f"paper_recipe_seed{seed}_rsa", id=f"paper_recipe_seed{seed}_rsa_{stamp}",
                     group="paper_recipe_rsa", job_type="rsa", config=dict(seed=seed, recipe="paper", n_epochs=int(df.epoch.max())))
    run.define_metric("epoch")
    run.define_metric("rsa/*", step_metric="epoch")
    for epoch, d in df.groupby("epoch"):
        run.log({f"rsa/{r.region}/{r.group}/{r.layer}": r.rho for r in d.itertuples()} | {"epoch": int(epoch)})
    run.finish()
    print(f"uploaded seed {seed}: {len(df)} rows", flush=True)

run = wandb.init(name="paper_recipe_rsa_summary", id=f"paper_recipe_rsa_summary_{stamp}",
                 group="paper_recipe_rsa", job_type="summary")
run.log({"rsa_table_all_seeds": wandb.Table(dataframe=pd.concat(tables, ignore_index=True))})
plots = sorted(glob.glob("results/rsa_paper_vs_old_*.png"))
run.log({f"plots/{os.path.basename(p)[:-4]}": wandb.Image(p) for p in plots})
run.finish()
print(f"uploaded summary: table + {len(plots)} plots")
