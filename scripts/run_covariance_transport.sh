#!/usr/bin/env bash
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
set -euo pipefail

repo_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
binary="${1:-${repo_dir}/build/asdole_covariance_transport}"
output_dir="${2:-${repo_dir}/results/covariance-transport}"
if [[ ! -x "$binary" ]]; then
    printf 'Executable not found: %s\nBuild it first with CMake.\n' "$binary" >&2
    exit 1
fi
mkdir -p "$output_dir"
output_csv="$output_dir/covariance_transport.csv"
tmp_csv="$output_dir/.covariance_transport.tmp.csv"
first=1
for theta in 0.5 0.6 0.7 1.0; do
    for steps in 10 20 40 80 160; do
        "$binary" "$theta" "$steps" > "$tmp_csv"
        if (( first )); then
            cat "$tmp_csv" > "$output_csv"
            first=0
        else
            tail -n +2 "$tmp_csv" >> "$output_csv"
        fi
    done
done
rm -f "$tmp_csv"
python3 "$repo_dir/scripts/plot_covariance_transport.py" \
    "$output_csv" --output "$output_dir/covariance_transport.png"
printf 'CSV: %s\nPlot: %s\n' "$output_csv" "$output_dir/covariance_transport.png"
