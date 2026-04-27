#!/bin/bash
set -euo pipefail

CONFIG_PATH="${1:-production_config.yaml}"
shift || true

if [[ $# -gt 0 ]]; then
    repeat_dirs=("$@")
else
    repeat_dirs=(sim_1 sim_2 sim_3)
fi

for repeat_dir in "${repeat_dirs[@]}"; do
    sbatch submit_production_repeat.sh "$repeat_dir" "$CONFIG_PATH"
done
