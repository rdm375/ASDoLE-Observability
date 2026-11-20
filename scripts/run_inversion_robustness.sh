#!/usr/bin/env bash
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_dir}/build/asdole_inversion_robustness}"
output_dir="${2:-${repo_dir}/results/inversion-robustness}"

if [[ ! -x "$binary" ]]; then
    printf 'Executable not found: %s\nBuild it first with CMake.\n' "$binary" >&2
    exit 1
fi

mkdir -p "$output_dir"
"$binary" "$output_dir" | tee "$output_dir/summary.txt"
python3 "$repo_dir/scripts/plot_inversion_robustness.py" \
    "$output_dir/inversion_robustness.csv" \
    --output "$output_dir/inversion_robustness.png" \
    --summary-output "$output_dir/inversion_robustness_summary.csv"
