#!/bin/bash
#SBATCH --job-name=alfworld-v1-seed1
#SBATCH --partition=public
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=alfworld-%j.log
cd ~/verl-agent || exit 1
source ~/.venvs/verl-alfworld/bin/activate
export WANDB_MODE=disabled
export HYDRA_FULL_ERROR=1
bash examples/grpo_trainer/run_alfworld_v1_seed1.sh
