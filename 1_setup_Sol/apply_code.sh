#!/bin/bash
# Copy one version's changed verl-agent files into your verl-agent folder.
# Each version's code/ folder uses verl-agent's own folder layout, e.g.
#   3_V2_methodA_Ankith/code/verl/trainer/ppo/uncertainty.py -> ~/verl-agent/verl/trainer/ppo/uncertainty.py
# Usage (from the repo folder): bash 1_setup_Sol/apply_code.sh 3_V2_methodA_Ankith
set -e
NAME="${1%/}"; [ -n "$NAME" ] || { echo "Usage: bash 1_setup_Sol/apply_code.sh <version folder, e.g. 3_V2_methodA_Ankith>"; exit 1; }
REPO="$(cd "$(dirname "$0")/.." && pwd)"
SRC="$REPO/$NAME/code"
VA="${2:-$HOME/verl-agent}"
[ -d "$SRC" ] || { echo "No folder $NAME/code"; exit 1; }
[ -d "$VA/verl" ] || { echo "No verl-agent found at $VA"; exit 1; }
cd "$SRC"
find . -type f ! -name README.md | while read -r f; do
  mkdir -p "$VA/$(dirname "$f")"; cp "$f" "$VA/$f"; echo "copied $f"
done
