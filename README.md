# Group 38: Which Uncertainty Helps LLM Agents Learn?

CSE 572 (Fall 2026) team project. We train Qwen2.5-1.5B-Instruct with GRPO on ALFWorld and compare two uncertainty signals as extra reward for failed attempts.

| Version | What it adds to plain GRPO | Owner | Folder |
|---|---|---|---|
| V1 | nothing (baseline) | Uttam | `2_V1_baseline_Uttam/` |
| V2 | Method A: single-attempt uncertainty (token entropy, from SELAUR) | Ankith | `3_V2_methodA_Ankith/` |
| V3 | Method B: agreement uncertainty (disagreement among 5 sampled actions) | Harshrajsinh | `4_V3_methodB_Harshrajsinh/` |
| V4 | A + B (average of the two) | Yugant | `5_V4_combined_Yugant/` |

This repo holds only our team's work. The training framework is [verl-agent](https://github.com/langfengQ/verl-agent) (Apache-2.0), which each of us clones separately (see "Getting set up" below). Our code is built on top of it.

## Results so far

| Model | Seen (140 games) | Unseen (134 games) |
|---|---|---|
| Untrained Qwen2.5-1.5B-Instruct | 9.3% | 6.0% |
| **V1 plain GRPO, seed 0, step 150** | **70.7%** | **76.9%** |
| GiGPO paper, plain GRPO (3 seeds) | 72.8 ± 3.6% (split not stated) | |

Full numbers: `7_results/summary.csv` (by task type) and `7_results/v1_seen_curve.csv` (success every 5 steps).

With ~135 test games, one run has a margin of roughly ±7–8 points, so differences smaller than that between versions are not conclusive without extra seeds.

## Where to find things

| Folder | What's in it |
|---|---|
| `1_setup_Sol/` | Sol training script (`run_alfworld_sol.sh`), a script that installs it into `~/verl-agent`, and `apply_code.sh` to copy a version's code into `~/verl-agent` |
| `2_V1_baseline_Uttam/` | V1 training job + unseen test job |
| `3_V2_methodA_Ankith/` | V2: job files here, changed verl-agent files in `code/` |
| `4_V3_methodB_Harshrajsinh/` | V3: same layout |
| `5_V4_combined_Yugant/` | V4: same layout |
| `6_job_templates/` | Copy these to make your own training / unseen-test jobs |
| `7_results/` | All versions' numbers in one place (CSV) |

`run_alfworld_sol.sh` is verl-agent's `run_alfworld.sh` fixed for Sol: `unset ROCR_VISIBLE_DEVICES`, 1 GPU, checkpoints every 10 steps, console logging, `TRAIN_SIZE` / `VAL_SIZE` / `GROUP_SIZE` settable, extra settings passed through.

## Rules for a fair comparison (V2–V4)

Keep **everything** identical to V1: Qwen2.5-1.5B-Instruct, GRPO, 150 steps, 16 tasks per step × 8 attempts, lr 1e-6, KL 0.01, max 50 steps per game, 1 A100, micro-batch 16, test every 5 steps on the 140 seen games, seed 0 first, final test on the 134 unseen games.

Change **only** the reward for failed attempts:

- Success stays **10**, each invalid action **−0.1** (verl-agent's values).
- A failed attempt gets `c × U(τ)` with **`c = 1.0`** for all versions and `U` in [0, 1], added **once per attempt**, not once per step.
- V2: `U` = mean over steps of normalized token entropy (entropy / log|V|), response tokens only.
- V3: `U` = fraction of 5 sampled actions that disagree with the most common one.
- V4: average of the two.
- All three go through the shared function `apply_failure_uncertainty_bonus` in `verl/trainer/ppo/uncertainty.py` (Ankith's, in `3_V2_methodA_Ankith/code/`).

Known interaction (Ankith): GRPO divides advantages by each group's reward spread. In all-fail groups the small differences between bonuses get scaled up to full strength. V1 used the same normalization, so we keep it for fairness and discuss it in the report.

## AUROC data (Yugant's eval scripts)

For each of V2–V4, save **one row per game**: `episode_id, step, uncertainty (0–1), success (0/1)`, as CSV or JSONL. Easiest way: after training, run a test-only pass on the final checkpoint (like `6_job_templates/TEMPLATE_eval_unseen.sbatch`) with that logging switched on. No retraining needed.

## Getting set up on Sol

Run git commands from the **login terminal** (compute nodes can't reach GitHub). The repo is private, so git needs a GitHub **personal access token** as the password (GitHub → Settings → Developer settings → Personal access tokens; give it access to this repo, Contents: Read and write).

```bash
cd ~
git clone https://github.com/langfengQ/verl-agent.git          # skip if you already have ~/verl-agent
git clone https://github.com/uttamsgowdamain/group38-llm-uncertainty.git
cd ~/group38-llm-uncertainty
bash 1_setup_Sol/install_into_verl_agent.sh                            # puts run_alfworld_sol.sh into ~/verl-agent
```

Then do the one-time install below if you haven't already.

## Sharing method code

Your method changes files **inside** `~/verl-agent`. To share them, copy each changed file into your folder's `code/` at the **same path**, then commit and push:

```bash
cd ~/group38-llm-uncertainty
git pull
mkdir -p 3_V2_methodA_Ankith/code/verl/trainer/ppo
cp ~/verl-agent/verl/trainer/ppo/uncertainty.py 3_V2_methodA_Ankith/code/verl/trainer/ppo/
# ...repeat for every verl-agent file you changed (cd ~/verl-agent && git status shows them)
git add 3_V2_methodA_Ankith && git commit -m "Method A: entropy bonus on failed attempts" && git push
```

To use someone's code (e.g. Yugant building V4 from A and B):

```bash
cd ~/group38-llm-uncertainty && git pull
bash 1_setup_Sol/apply_code.sh 3_V2_methodA_Ankith      # copies A's files into ~/verl-agent
```

If A and B both change the same file, merge those by hand for V4 and put the combined files in `5_V4_combined_Yugant/code/`. Put your job files in your own folder (e.g. `3_V2_methodA_Ankith/v2_methodA.sbatch`) and your results in `7_results/`. Never commit checkpoints, logs or W&B folders.

## Installing on Sol (one time, ~2 h)

1. **Login terminal:** get verl-agent and this repo (section above).
2. **Sol Desktop:** partition `public`, QOS `class`, GPU `gpu:a100:1`, 8 cores, 64 GB, 4 h. Then:
   ```bash
   module load mamba/latest
   mamba create -n verl-agent -c conda-forge python=3.12 -y
   source activate verl-agent
   cd ~/verl-agent
   pip3 install vllm==0.11.0
   module load gcc/12.5.0 cuda-12.8.1-gcc-12.1.0
   export CC=$(which gcc) CXX=$(which g++) CUDAHOSTCXX=$(which g++) MAX_JOBS=4
   pip3 install flash-attn==2.7.4.post1 --no-build-isolation --no-cache-dir   # ~1.5 h compile
   pip install -e .
   pip3 install gymnasium==0.29.1 stable-baselines3==2.6.0 alfworld
   alfworld-download -f            # run from the login terminal if it can't connect
   mkdir -p /scratch/$USER/hf_cache /scratch/$USER/logs
   echo 'export HF_HOME=/scratch/$USER/hf_cache' >> ~/.bashrc
   ```
3. **Check:** `python -c "import torch, vllm, flash_attn, verl; print(torch.__version__, torch.cuda.is_available(), vllm.__version__, flash_attn.__version__)"` → `2.8.0+cu128 True 0.11.0 2.7.4.post1`

## Running

Every new terminal:

```bash
module load mamba/latest && source activate verl-agent
module load gcc/12.5.0 cuda-12.8.1-gcc-12.1.0
cd ~/verl-agent && export HF_HOME=/scratch/$USER/hf_cache
F=examples/grpo_trainer/run_alfworld_sol.sh
```

**2-step training test** (Sol Desktop, 16 cores, 192 GB), before any full run:

```bash
bash $F vllm trainer.total_epochs=2 trainer.val_before_train=False trainer.save_freq=2 trainer.default_local_dir=/scratch/$USER/ckpt/test actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
```

**Full run** (login terminal, no GPU desktop open), 3 chained ~16 h jobs:

```bash
cd ~/group38-llm-uncertainty
J1=$(sbatch --parsable 3_V2_methodA_Ankith/v2_methodA.sbatch)
J2=$(sbatch --parsable --dependency=afterany:$J1 3_V2_methodA_Ankith/v2_methodA.sbatch)
sbatch --dependency=afterany:$J2 3_V2_methodA_Ankith/v2_methodA.sbatch
```

**Progress:** `grep "val/success_rate" /scratch/$USER/logs/<name>_*.out | grep -o "step:[0-9]* \|val/success_rate:[0-9.]*" | paste - -`

## Sol rules for our class account

- Partition `public` (not `general`), QOS `class`, account `class_cse57268549fall2026` (check yours: `sacctmgr show assoc user=$USER format=account%30,qos%20`).
- Max 24 h per job, and max **960 GPU-minutes running per person** (16 h on one A100). An open GPU desktop counts. Long runs = chained ~16 h jobs that resume from checkpoints. The limit is per person, so we can all train at once.
- Request **16 cores + 192 GB RAM** for any test or training run (a run uses ~157 GB; each ALFWorld game is its own process).
- Checkpoints are ~17 GB each: always save to `/scratch`, never home (100 GB).
- Cost: ~12 min per step for V1 (~32 h total with testing); Method A ~15 min per step.

## Errors we hit and the fix

| Error | Cause | Fix |
|---|---|---|
| `git clone … unable to access` | Compute nodes can't reach GitHub | Use the login terminal |
| flash-attn build fails, gcc 8.5 | Default gcc too old (16 too new for CUDA 12.8) | `module load gcc/12.5.0 cuda-12.8.1-gcc-12.1.0` |
| `nvidia-smi`: no GPU | Desktop launched with the GPU field empty | Enter `gpu:a100:1` |
| `ActorDiedError … OOM killer` | Not enough RAM | 16 cores, 192 GB |
| `Please don't set ROCR_VISIBLE_DEVICES` | Sol sets an AMD GPU variable | Fixed in `run_alfworld_sol.sh` |
| `'general' partition … privately-owned nodes` | Partition renamed | `-p public` |
| `Invalid qos specification` | Class accounts use QOS `class` | `-q class -A class_cse57268549fall2026` |
| `MaxGRESRunMinsPerUser` | Over 16 GPU-hours running, or a desktop open | `-t 15:55:00`, close desktops, chain jobs |
| `This does not look like a batch script` | Something above `#!/bin/bash` | Make it the first line |
| `No space left on device` | Checkpoints went to home | `trainer.default_local_dir=/scratch/...` |
| `trainer.xxx=...: command not found`, exit 127 | Pasting into Sol Shell Access adds blank lines that split `\` continuations | Keep the command on one line (the job files here do), or get files with git instead of pasting |
| `CUDA out of memory` in `update_actor` | Micro-batch 32 too big next to vLLM's cache | Micro-batch 16 (in all job files) |
| `Expandable segments are not compatible with memory pool` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` | Don't set it |
| Bad checkpoint after a crash | Crash during a save | Delete any `global_step_N` newer than `latest_checkpointed_iteration.txt` |
| Edited job file has no effect | Slurm copies the file at submit time | `scancel` waiting jobs and resubmit |

