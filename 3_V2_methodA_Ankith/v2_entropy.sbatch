#!/bin/bash
#SBATCH -J v2_entropy
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57268549fall2026
#SBATCH -G a100:1
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 15:55:00
#SBATCH -o /scratch/%u/logs/v2_entropy_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --export=NONE
module load mamba/latest
source activate verl-agent
module load gcc/12.5.0 cuda-12.8.1-gcc-12.1.0
cd ~/verl-agent
export HF_HOME=/scratch/$USER/hf_cache
export WANDB_MODE=offline
export WANDB_DIR=/scratch/$USER/wandb
mkdir -p "$WANDB_DIR"
ray stop --force 2>/dev/null
RUN=v2_grpoA_seed0
VAL_SIZE=140 bash examples/grpo_trainer/run_alfworld_sol.sh vllm \
  trainer.logger="['console','wandb']" \
  trainer.experiment_name=$RUN \
  trainer.default_local_dir=/scratch/$USER/ckpt/$RUN \
  trainer.val_before_train=False \
  trainer.save_freq=5 \
  env.seed=0 \
  +algorithm.uncertainty_method=entropy \
  +algorithm.uncertainty_c=1.0
