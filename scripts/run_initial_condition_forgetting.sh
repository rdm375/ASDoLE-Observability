#!/usr/bin/env bash
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_dir}/build/asdole_initial_condition_forgetting}"
output_dir="${2:-${repo_dir}/results/initial-condition-forgetting}"
if [[ ! -x "$binary" ]]; then
    printf 'Executable not found: %s\nBuild it first with CMake.\n' "$binary" >&2
    exit 1
fi
mkdir -p "$output_dir"
csv="$output_dir/forgetting.csv"
summary="$output_dir/summary.txt"
plot="$output_dir/initial_condition_forgetting.png"
"$binary" "$csv" | tee "$summary"
python3 "$repo_dir/scripts/plot_initial_condition_forgetting.py" "$csv" \
    --summary "$summary" --output "$plot"
printf 'CSV: %s\nSummary: %s\nPlot: %s\n' "$csv" "$summary" "$plot"
