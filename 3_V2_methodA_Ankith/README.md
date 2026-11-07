# Version 2: GRPO + Method A, Ankith

Single-attempt uncertainty: mean normalized token entropy over the attempt's steps (response tokens only).

Put in this folder:

| What | Where |
|---|---|
| Your job files | here, e.g. `v2_methodA.sbatch` (copy `6_job_templates/TEMPLATE_train.sbatch`) |
| verl-agent files you changed | `code/`, same folder layout as verl-agent (e.g. `code/verl/trainer/ppo/uncertainty.py`) |
| Your results | add a row to `7_results/summary.csv` and your curve as `7_results/v2_seen_curve.csv` |

Others get your code with: `bash 1_setup_Sol/apply_code.sh 3_V2_methodA_Ankith`
