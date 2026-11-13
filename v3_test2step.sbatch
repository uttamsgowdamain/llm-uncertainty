#!/bin/bash
# Version 3 (GRPO + Method B): 2-step training test. Run this before any full run.
# Passes if the log ends with V3_TEST_DONE, shows "uncertainty/" metrics, and
# /scratch/$USER/ckpt/v3_test/global_step_2 exists.
#SBATCH -J v3_test2step
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_cse57268549fall2026
#SBATCH -G a100:1
#SBATCH -c 16
#SBATCH --mem=192G
#SBATCH -t 4:00:00
#SBATCH -o /scratch/%u/logs/v3_test2step_%j.out
#SBATCH --export=NONE
set -e
module load mamba/latest
source activate verl-agent
module load gcc/12.5.0 cuda-12.8.1-gcc-12.1.0
cd ~/verl-agent
export HF_HOME=/scratch/$USER/hf_cache
ray stop --force 2>/dev/null || true
bash examples/grpo_trainer/run_alfworld_sol.sh vllm trainer.total_epochs=2 trainer.val_before_train=False trainer.save_freq=2 trainer.default_local_dir=/scratch/$USER/ckpt/v3_test actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=16 actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=16 actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=16 +algorithm.uncertainty_method=agreement +algorithm.uncertainty_c=1.0 +algorithm.agreement_k=5
echo V3_TEST_DONE
