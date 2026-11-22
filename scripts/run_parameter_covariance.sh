#!/usr/bin/env bash
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_dir}/build/asdole_parameter_covariance_transport}"
output_dir="${2:-${repo_dir}/results/parameter-covariance}"
theta="${3:-0.7}"
if [[ ! -x "$binary" ]]; then
    printf 'Executable not found: %s\nBuild it first with CMake.\n' "$binary" >&2
    exit 1
fi
mkdir -p "$output_dir"
"$binary" "$theta" > "$output_dir/parameter_covariance.csv"
python3 "$repo_dir/scripts/plot_parameter_covariance.py" \
    "$output_dir/parameter_covariance.csv" --output "$output_dir/parameter_covariance.png"
printf 'CSV: %s\nPlot: %s\n' \
    "$output_dir/parameter_covariance.csv" "$output_dir/parameter_covariance.png"
