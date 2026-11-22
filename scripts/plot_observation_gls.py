#!/usr/bin/env python3
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
"""Plot temporally correlated observation uncertainty and a GLS bias fit."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIELDS = ("time", "residual", "state_variance", "observation_variance",
          "covariance_with_first", "correlation_with_first")


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(FIELDS)}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                parsed = {field: float(row[field]) for field in FIELDS}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid CSV line {line_number}: {exc}") from exc
            if not all(math.isfinite(value) for value in parsed.values()):
                raise ValueError(f"non-finite data on CSV line {line_number}")
            rows.append(parsed)
    if not rows:
        raise ValueError("CSV contains no observations")
    return rows


def make_plot(rows, output: Path, gls_bias: float, gls_sigma: float,
              ordinary_bias: float, ordinary_sigma: float):
    times = [row["time"] for row in rows]
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.8))
    axes[0].plot(times, [row["state_variance"] for row in rows],
                 label="propagated state variance")
    axes[0].plot(times, [row["observation_variance"] for row in rows],
                 label="total observation variance")
    axes[0].set_title("Observation uncertainty by time")
    axes[0].set_ylabel("Variance")
    axes[0].legend(frameon=False)
    ax_corr = axes[0].twinx()
    ax_corr.plot(times, [row["correlation_with_first"] for row in rows],
                 color="tab:green", linestyle="--", label="correlation with first sample")
    ax_corr.set_ylabel("Correlation")
    ax_corr.set_ylim(-1.05, 1.05)
    ax_corr.legend(frameon=False, loc="center right", fontsize=8)

    residual = [row["residual"] for row in rows]
    axes[1].plot(times, residual, "o", markersize=3, label="synthetic residual")
    axes[1].axhline(gls_bias, color="tab:red", label="GLS fitted bias")
    axes[1].fill_between(times, [gls_bias - gls_sigma] * len(times),
                         [gls_bias + gls_sigma] * len(times), color="tab:red",
                         alpha=0.15, label="GLS 1-sigma")
    axes[1].axhline(ordinary_bias, color="tab:purple", linestyle="--",
                    label="ordinary least-squares bias")
    axes[1].set_title("Constant sensor-bias fit")
    axes[1].set_ylabel("Residual")
    axes[1].legend(frameon=False, fontsize=8)

    for axis in axes:
        axis.set_xlabel("Time")
        axis.grid(True, alpha=0.25)
    fig.suptitle("Observation covariance and generalized least squares")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("observation_gls.png"))
    parser.add_argument("--summary", type=Path, required=True,
                        help="text output from asdole_observation_gls")
    args = parser.parse_args()
    summary = args.summary.read_text(encoding="utf-8")
    values = {}
    # Parse the stable key=value diagnostics without depending on field order.
    for line in summary.splitlines():
        for token in line.split():
            if "=" in token:
                key, value = token.split("=", 1)
                try:
                    values[key] = float(value)
                except ValueError:
                    continue
    make_plot(read_rows(args.csv), args.output,
              values["gls_bias"], values["gls_formal_sigma"],
              values["ordinary_least_squares_bias"],
              values["ordinary_formal_sigma"])
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
