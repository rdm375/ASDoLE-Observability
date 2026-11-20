#!/usr/bin/env bash
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_dir}/build/asdole_stability_sweep}"
output_dir="${2:-${repo_dir}/results/stability}"

if [[ ! -x "$binary" ]]; then
    printf 'Executable not found: %s\nBuild it first with CMake.\n' "$binary" >&2
    exit 1
fi

mkdir -p "$output_dir"
"$binary" > "$output_dir/stability_sweep.csv" 2> "$output_dir/summary.txt"
python3 "$repo_dir/scripts/plot_stability_sweep.py" \
    "$output_dir/stability_sweep.csv" \
    --output "$output_dir/stability_sweep.png"
printf 'CSV: %s\nPlot: %s\nSummary: %s\n' \
    "$output_dir/stability_sweep.csv" "$output_dir/stability_sweep.png" \
    "$output_dir/summary.txt"
