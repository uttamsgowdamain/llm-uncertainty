Changed verl-agent files for Version 2: GRPO + Method A go here, keeping verl-agent's folder layout.


Update (Nov 27): V2 fix. Method A bonus is skipped in all-fail groups (+algorithm.uncertainty_skip_all_fail_groups=True), as agreed for V2, V3 and V4.

Now includes agent_system/multi_turn_rollout/rollout_loop.py alongside ray_trainer.py and uncertainty.py. v2fix.patch is the diff against the last commit.

Job file: ../v2fix_entropy.sbatch (entropy, c=1.0, seed 0).
