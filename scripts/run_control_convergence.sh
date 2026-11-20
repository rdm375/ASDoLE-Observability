#!/usr/bin/env bash
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_dir}/build/asdole_decay_control_convergence}"
output_dir="${2:-${repo_dir}/results/control-convergence}"

if [[ ! -x "$binary" ]]; then
    printf 'Executable not found: %s\nBuild it first with CMake.\n' "$binary" >&2
    exit 1
fi

mkdir -p "$output_dir"
"$binary" > "$output_dir/control_convergence.csv"
python3 "$repo_dir/scripts/plot_control_convergence.py" \
    "$output_dir/control_convergence.csv" \
    --output "$output_dir/control_convergence.png"
printf 'CSV: %s\nPlot: %s\n' \
    "$output_dir/control_convergence.csv" "$output_dir/control_convergence.png"
