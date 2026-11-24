# Aigerim — V1 Extra Seeds

## Completed
- Connected to ASU Sol and verified the verl-agent environment.
- Installed the missing gymnasium dependency.
- Cloned the team repository and installed run_alfworld_sol.sh.
- Prepared separate V1 seed 1 training and Slurm scripts.
- Verified access to the CSE 572 class account and QOS.
- Identified missing Flash Attention and attempted installation.

## Current Status
- Flash Attention installation is not yet complete.
- Compute-node session requested: Job 2331558.
- Initial smoke test 2328063 was cancelled before execution because its resource configuration needed correction.
- V1 seed 1 training and unseen evaluation have not started.

## Files
- run_alfworld_v1_seed1.sh: preliminary training configuration.
- alfworld_v1_seed1.sbatch: preliminary Slurm configuration.

## Next Steps
- Complete Flash Attention installation.
- Update scripts to match the team's validated Setup Kit.
- Run the 2-step smoke test.
- Train V1 seed 1 and evaluate on unseen tasks.

Note: These scripts are drafts and have not yet passed GPU validation.
