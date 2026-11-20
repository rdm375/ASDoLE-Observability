#!/usr/bin/env bash
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_dir}/build/asdole_decay}"
output_dir="${2:-${repo_dir}/results/convergence}"

if [[ ! -x "$binary" ]]; then
    printf 'Executable not found: %s\nBuild it first with CMake.\n' "$binary" >&2
    exit 1
fi

mkdir -p "$output_dir"
"$binary" > "$output_dir/convergence.csv"
python3 "$repo_dir/scripts/plot_convergence.py" \
    "$output_dir/convergence.csv" \
    --output "$output_dir/convergence.png"
printf 'CSV: %s\nPlot: %s\n' \
    "$output_dir/convergence.csv" "$output_dir/convergence.png"
