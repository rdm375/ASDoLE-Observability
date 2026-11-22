#!/usr/bin/env bash
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_dir}/build/asdole_error_source_comparison}"
output_dir="${2:-${repo_dir}/results/error-source-comparison}"
if [[ ! -x "$binary" ]]; then
    printf 'Executable not found: %s\nBuild it first with CMake.\n' "$binary" >&2
    exit 1
fi
mkdir -p "$output_dir"
"$binary" > "$output_dir/error_source_comparison.csv"
python3 "$repo_dir/scripts/plot_error_source_comparison.py" \
    "$output_dir/error_source_comparison.csv" \
    --output "$output_dir/error_source_comparison.png"
printf 'CSV: %s\nPlot: %s\n' \
    "$output_dir/error_source_comparison.csv" \
    "$output_dir/error_source_comparison.png"
