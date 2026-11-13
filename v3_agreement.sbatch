#!/bin/bash
# Version 3: GRPO + Method B (agreement uncertainty). Copied from 6_job_templates/TEMPLATE_train.sbatch;
# only -J, -o, RUN, SEED and EXTRA differ. Do NOT launch until the team has agreed on the V2 collapse fix.
#SBATCH -J v3_agreement
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57268549fall2026
#SBATCH -G a100:1
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 15:55:00
#SBATCH -o /scratch/%u/logs/v3_agreement_%j.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=%u@asu.edu
#SBATCH --export=NONE
RUN=v3_grpoB_seed0
SEED=0
EXTRA="+algorithm.uncertainty_method=agreement +algorithm.uncertainty_c=1.0 +algorithm.agreement_k=5"
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
