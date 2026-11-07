#!/bin/bash
#SBATCH -J v1_grpo
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57268549fall2026
#SBATCH -G a100:1
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 15:55:00
#SBATCH -o /scratch/%u/logs/v1_grpo_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --export=NONE
# Version 1: plain GRPO, seed 0. Exactly the job that produced our baseline
# (70.7% seen / 76.9% unseen). Resubmit the same file to resume from the
# latest checkpoint; chain with: sbatch --dependency=afterany:<jobid> <file>
module load mamba/latest
source activate verl-agent
module load gcc/12.5.0 cuda-12.8.1-gcc-12.1.0
cd ~/verl-agent
export HF_HOME=/scratch/$USER/hf_cache
export WANDB_MODE=offline
export WANDB_DIR=/scratch/$USER/wandb
mkdir -p "$WANDB_DIR"
ray stop --force 2>/dev/null
VAL_SIZE=140 bash examples/grpo_trainer/run_alfworld_sol.sh vllm trainer.logger="['console','wandb']" trainer.experiment_name=v1_grpo_seed0 trainer.default_local_dir=/scratch/$USER/ckpt/v1_grpo_seed0 trainer.val_before_train=False trainer.save_freq=5 env.seed=0 actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
