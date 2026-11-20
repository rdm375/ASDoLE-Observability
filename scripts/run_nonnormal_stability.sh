#!/usr/bin/env bash
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_dir}/build/asdole_nonnormal_stability}"
output_dir="${2:-${repo_dir}/results/nonnormal-stability}"

if [[ ! -x "$binary" ]]; then
    printf 'Executable not found: %s\nBuild it first with CMake.\n' "$binary" >&2
    exit 1
fi

mkdir -p "$output_dir"
"$binary" > "$output_dir/nonnormal_stability.csv" 2> "$output_dir/summary.txt"
python3 "$repo_dir/scripts/plot_nonnormal_stability.py" \
    "$output_dir/nonnormal_stability.csv" \
    --output "$output_dir/nonnormal_stability.png" \
    --components-output "$output_dir/nonnormal_components.png"
printf 'CSV: %s\nNorm plot: %s\nComponent plot: %s\nSummary: %s\n' \
    "$output_dir/nonnormal_stability.csv" "$output_dir/nonnormal_stability.png" \
    "$output_dir/nonnormal_components.png" "$output_dir/summary.txt"
