# Version 1: plain GRPO (baseline), Uttam

No code changes to verl-agent: V1 uses `1_setup_Sol/run_alfworld_sol.sh` as is.

| File | What it is |
|---|---|
| `v1_grpo.sbatch` | The exact training job (150 steps, seed 0) |
| `v1_eval_unseen.sbatch` | Tests the final checkpoint on the 134 unseen games |

Result: **70.7% seen / 76.9% unseen** (untrained: 9.3% / 6.0%). Full numbers in `7_results/`.
