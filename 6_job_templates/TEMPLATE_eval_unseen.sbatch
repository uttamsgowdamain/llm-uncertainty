#!/bin/bash
# Test a finished run's latest checkpoint on the 134 unseen games (no training).
# Set RUN to the run you want to test. Result: grep "val/.*success_rate" in the log.
#SBATCH -J eval_unseen
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57268549fall2026
#SBATCH -G a100:1
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 01:30:00
#SBATCH -o /scratch/%u/logs/eval_unseen_%j.out
#SBATCH --export=NONE
RUN=v1_grpo_seed0
module load mamba/latest
source activate verl-agent
module load gcc/12.5.0 cuda-12.8.1-gcc-12.1.0
cd ~/verl-agent
export HF_HOME=/scratch/$USER/hf_cache
ray stop --force 2>/dev/null
VAL_SIZE=134 bash examples/grpo_trainer/run_alfworld_sol.sh vllm trainer.logger="['console']" trainer.default_local_dir=/scratch/$USER/ckpt/$RUN trainer.val_before_train=True trainer.val_only=True env.alfworld.eval_dataset=eval_out_of_distribution actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16
