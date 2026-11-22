#!/usr/bin/env python3
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
"""Plot initial-condition error and covariance forgetting diagnostics."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIELDS = (
    "time", "error_x", "error_y", "error_norm", "tlm_gain_2norm",
    "var_x", "cov_xy", "var_y", "trace_covariance", "sigma_x", "sigma_y",
    "correlation_xy", "local_error_decay_rate", "local_covariance_decay_rate",
    "information_xx", "information_xy", "information_yy",
    "information_eigen_min", "information_eigen_max", "information_rate_y",
)


def read_rows(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError("CSV is missing expected forgetting-analysis columns")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                item = {key: float(row[key]) for key in FIELDS}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid CSV row {line_number}: {exc}") from exc
            if not all(math.isfinite(value) for value in item.values()):
                raise ValueError(f"non-finite value on CSV line {line_number}")
            rows.append(item)
    if not rows:
        raise ValueError("CSV contains no data rows")
    return rows


def read_summary(path: Path):
    result = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        for token in line.split():
            if "=" in token:
                key, value = token.split("=", 1)
                try:
                    result[key] = float(value)
                except ValueError:
                    pass
    return result


def make_plot(rows, summary, output: Path):
    t = [row["time"] for row in rows]
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.0))

    ax = axes[0, 0]
    ax.semilogy(t, [row["error_norm"] for row in rows], label="selected error norm")
    ax.semilogy(t, [row["tlm_gain_2norm"] for row in rows], label="worst-case TLM gain")
    ax.set_title("Initial-condition perturbation")
    ax.set_ylabel("Norm / gain")
    ax.legend(frameon=False)

    ax = axes[0, 1]
    ax.semilogy(t, [row["var_x"] for row in rows], label=r"$P_{xx}$")
    ax.semilogy(t, [row["var_y"] for row in rows], label=r"$P_{yy}$")
    positive_cross = [max(abs(row["cov_xy"]), 1.0e-300) for row in rows]
    ax.semilogy(t, positive_cross, label=r"$|P_{xy}|$")
    ax.set_title("Absolute covariance decays")
    ax.set_ylabel("Variance / covariance")
    ax.legend(frameon=False)

    ax = axes[1, 0]
    ax.plot(t, [row["correlation_xy"] for row in rows], color="tab:purple")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("Normalized correlation can rise")
    ax.set_ylabel(r"$\rho_{xy}$")

    ax = axes[1, 1]
    ax.plot(t, [row["information_eigen_min"] for row in rows],
            label="small information eigenvalue")
    ax.plot(t, [row["information_eigen_max"] for row in rows],
            label="large information eigenvalue")
    ax.set_title("Information about the initial state accumulates")
    ax.set_ylabel("Cumulative information eigenvalue")
    ax2 = ax.twinx()
    ax2.semilogy(t[1:], [row["information_rate_y"] for row in rows[1:]],
                 color="tab:red", linestyle="--", label="instantaneous information rate")
    ax2.set_ylabel("Information rate")
    ax.legend(frameon=False, fontsize=8, loc="upper left")
    ax2.legend(frameon=False, fontsize=8, loc="center right")

    for ax in axes.flat:
        ax.set_xlabel("Time")
        ax.grid(True, alpha=0.25)
    fig.suptitle(
        "Initial-condition forgetting: state sensitivity, covariance, and information\n"
        f"late-window fitted decay rates: state {summary['fitted_error_decay_rate']:.3f}, "
        f"covariance trace {summary['fitted_covariance_trace_decay_rate']:.3f}")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("forgetting.png"))
    args = parser.parse_args()
    make_plot(read_rows(args.csv), read_summary(args.summary), args.output)
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
