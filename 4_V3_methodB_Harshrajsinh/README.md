# Version 3: GRPO + Method B, Harshrajsinh

Agreement uncertainty: fraction of 5 sampled actions that disagree with the most common one.

Put in this folder:

| What | Where |
|---|---|
| Your job files | here, e.g. `v3_methodB.sbatch` (copy `6_job_templates/TEMPLATE_train.sbatch`) |
| verl-agent files you changed | `code/`, same folder layout as verl-agent (e.g. `code/verl/trainer/ppo/uncertainty.py`) |
| Your results | add a row to `7_results/summary.csv` and your curve as `7_results/v3_seen_curve.csv` |

Others get your code with: `bash 1_setup_Sol/apply_code.sh 4_V3_methodB_Harshrajsinh`
