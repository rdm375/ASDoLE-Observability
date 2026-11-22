#!/usr/bin/env python3
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
"""Plot state uncertainty induced by initial-state and parameter uncertainty."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("parameter_covariance.png"))
    args = parser.parse_args()
    with args.csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"time", "var_x", "cov_xy", "var_y", "cov_xk", "cov_yk", "var_k"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        rows = [{key: float(row[key]) for key in required} for row in reader]
    if not rows:
        raise ValueError("CSV contains no rows")

    time = [row["time"] for row in rows]
    sigma_x = [math.sqrt(max(0.0, row["var_x"])) for row in rows]
    sigma_y = [math.sqrt(max(0.0, row["var_y"])) for row in rows]
    corr_xk = [row["cov_xk"] / math.sqrt(row["var_x"] * row["var_k"])
               if row["var_x"] > 0 and row["var_k"] > 0 else float("nan") for row in rows]
    corr_yk = [row["cov_yk"] / math.sqrt(row["var_y"] * row["var_k"])
               if row["var_y"] > 0 and row["var_k"] > 0 else float("nan") for row in rows]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 7.5))
    axes[0, 0].plot(time, sigma_x, label=r"$\sigma_x$")
    axes[0, 0].plot(time, sigma_y, label=r"$\sigma_y$")
    axes[0, 0].set_title("Marginal state uncertainty")
    axes[0, 0].set_ylabel("Standard deviation")
    axes[0, 1].plot(time, [row["cov_xy"] for row in rows], color="tab:purple",
                    label=r"$\operatorname{Cov}(x,y)$")
    axes[0, 1].set_title("State cross-covariance")
    axes[0, 1].set_ylabel(r"$\operatorname{Cov}(x,y)$")
    axes[1, 0].plot(time, [row["cov_xk"] for row in rows], label=r"$\operatorname{Cov}(x,k)$")
    axes[1, 0].plot(time, [row["cov_yk"] for row in rows], label=r"$\operatorname{Cov}(y,k)$")
    axes[1, 0].set_title("State/parameter cross-covariance")
    axes[1, 0].set_ylabel("Cross-covariance")
    axes[1, 1].plot(time, corr_xk, label=r"$\operatorname{Corr}(x,k)$")
    axes[1, 1].plot(time, corr_yk, label=r"$\operatorname{Corr}(y,k)$")
    axes[1, 1].set_title("Normalized parameter correlations")
    axes[1, 1].set_ylabel("Correlation")
    axes[1, 1].set_ylim(-1.05, 1.05)
    for axis in axes.flat:
        axis.set_xlabel("Time")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Propagation of initial-state and coupling-parameter uncertainty")
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
