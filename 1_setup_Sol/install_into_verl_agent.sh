#!/bin/bash
# Copy our Sol training script into your verl-agent folder (default ~/verl-agent).
# Usage (from the repo folder): bash 1_setup_Sol/install_into_verl_agent.sh
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
VA="${1:-$HOME/verl-agent}"
[ -d "$VA/verl" ] || { echo "No verl-agent found at $VA (clone it first, see README)"; exit 1; }
cp "$HERE/run_alfworld_sol.sh" "$VA/examples/grpo_trainer/run_alfworld_sol.sh"
echo "Copied run_alfworld_sol.sh into $VA/examples/grpo_trainer/"
