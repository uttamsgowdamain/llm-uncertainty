# Version 3: GRPO + Method B, Harshrajsinh

Agreement uncertainty: at each step, sample K = 5 extra actions for the same state and measure the
fraction that disagree with the most common one, u_t = 1 - max_a #{k : a_k = a} / K. U_B is the mean of
u_t over the attempt's steps (report Eq. 4), passed as `step_u` into Ankith's shared
`apply_failure_uncertainty_bonus`, so a failed attempt gets c x U_B (c = 1.0). Everything else matches V1.

## Changed verl-agent files (`code/`)

| File | Change |
|---|---|
| `verl/trainer/ppo/uncertainty.py` | Ankith's file unchanged, plus `normalize_action`, `agreement_uncertainty`, `agreement_step_uncertainty` |
| `verl/trainer/ppo/ray_trainer.py` | Ankith's file, with `uncertainty_method` = `entropy` (V2), `agreement` (V3) or `both` (V4: per-step average of A and B) |
| `agent_system/multi_turn_rollout/rollout_loop.py` | Samples K extra responses per active environment per step and stores `agreement_u` per row |

This code/ folder is a superset of Method A's: applying it keeps V2 working (`uncertainty_method=entropy`),
and with no `uncertainty_method` set it behaves exactly like V1.

## Settings

    +algorithm.uncertainty_method=agreement   # entropy | agreement | both
    +algorithm.uncertainty_c=1.0
    +algorithm.agreement_k=5
    +algorithm.agreement_in_val=True          # optional: also sample during validation (AUROC test-only pass)

## Design choices to note in Methods

- The K samples are extra draws with the same sampling settings as the executed action; the executed action is not one of the K.
- Normalization: first `<action>...</action>` block, lowercased, whitespace collapsed, trailing punctuation dropped.
  A sample without an action block maps to one shared `<invalid>` label.
- Extra samples are only drawn for environments that are still active, and only during training by default.
- U_B is in [0, 1 - 1/K] = [0, 0.8], while Method A's U is in [0, 1].

## Run

    cd ~/group38-llm-uncertainty && git pull
    bash 1_setup_Sol/apply_code.sh 4_V3_methodB_Harshrajsinh
    sbatch 4_V3_methodB_Harshrajsinh/v3_test2step.sbatch       # 2-step test first
    # full run (3+ chained jobs) only after the V2 fix is agreed:
    # J1=$(sbatch --parsable 4_V3_methodB_Harshrajsinh/v3_agreement.sbatch) ...
