#!/bin/bash
# Copy this file for your version, e.g. cp 6_job_templates/TEMPLATE_train.sbatch 3_V2_methodA_Ankith/v2_methodA.sbatch
# Then change ONLY: the -J name, the -o log name, RUN, SEED, and EXTRA (your method's settings).
# Everything else must stay identical to Version 1 so the comparison is fair.
#SBATCH -J vN_name
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57268549fall2026
#SBATCH -G a100:1
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 15:55:00
#SBATCH -o /scratch/%u/logs/vN_name_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --export=NONE
RUN=vN_name_seed0
SEED=0
EXTRA=""
module load mamba/latest
source activate verl-agent
module load gcc/12.5.0 cuda-12.8.1-gcc-12.1.0
cd ~/verl-agent
export HF_HOME=/scratch/$USER/hf_cache
export WANDB_MODE=offline
export WANDB_DIR=/scratch/$USER/wandb
mkdir -p "$WANDB_DIR"
ray stop --force 2>/dev/null
VAL_SIZE=140 bash examples/grpo_trainer/run_alfworld_sol.sh vllm trainer.logger="['console','wandb']" trainer.experiment_name=$RUN trainer.default_local_dir=/scratch/$USER/ckpt/$RUN trainer.val_before_train=False trainer.save_freq=5 env.seed=$SEED actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 $EXTRA
