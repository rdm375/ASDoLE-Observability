#!/usr/bin/env bash
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_dir}/build/asdole_observation_gls}"
output_dir="${2:-${repo_dir}/results/observation-gls}"
if [[ ! -x "$binary" ]]; then
    printf 'Executable not found: %s\nBuild it first with CMake.\n' "$binary" >&2
    exit 1
fi
mkdir -p "$output_dir"
csv="$output_dir/observation_gls.csv"
summary="$output_dir/summary.txt"
plot="$output_dir/observation_gls.png"
"$binary" "$csv" | tee "$summary"
python3 "$repo_dir/scripts/plot_observation_gls.py" "$csv" \
    --summary "$summary" --output "$plot"
printf 'CSV: %s\nSummary: %s\nPlot: %s\n' "$csv" "$summary" "$plot"
