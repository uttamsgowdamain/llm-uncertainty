# Version 4: GRPO + A + B, Yugant

Average of Method A and Method B uncertainty. If A and B change the same file, merge them by hand here.

Put in this folder:

| What | Where |
|---|---|
| Your job files | here, e.g. `v4_combined.sbatch` (copy `6_job_templates/TEMPLATE_train.sbatch`) |
| verl-agent files you changed | `code/`, same folder layout as verl-agent (e.g. `code/verl/trainer/ppo/uncertainty.py`) |
| Your results | add a row to `7_results/summary.csv` and your curve as `7_results/v4_seen_curve.csv` |

Others get your code with: `bash 1_setup_Sol/apply_code.sh 5_V4_combined_Yugant`
