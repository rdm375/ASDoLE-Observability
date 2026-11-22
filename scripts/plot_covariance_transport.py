#!/usr/bin/env python3
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
"""Plot discrete ASDoLE state-covariance propagation and refinement results."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REQUIRED = {
    "theta", "steps", "dt", "time", "var_x", "cov_xy", "var_y",
    "correlation_xy", "state_error", "covariance_frobenius_error",
}


def read_rows(path: Path):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not REQUIRED.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(REQUIRED))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                item = {name: float(row[name]) for name in REQUIRED - {"steps"}}
                item["steps"] = int(row["steps"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid value on CSV line {line_number}: {exc}") from exc
            # A correlation is mathematically undefined when either variance
            # is zero. The C++ diagnostic represents that case as NaN; keep
            # it as a plot gap while rejecting non-finite physical quantities.
            non_finite = [name for name, value in item.items() if not math.isfinite(value)]
            unexpected = [name for name in non_finite
                          if name != "correlation_xy" or not math.isnan(item[name])]
            if unexpected:
                raise ValueError(
                    f"non-finite value in {', '.join(unexpected)} on CSV line {line_number}"
                )
            rows.append(item)
    if not rows:
        raise ValueError("CSV contains no data")
    return rows


def make_plot(rows, output: Path):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["theta"], row["steps"])].append(row)
    for history in grouped.values():
        history.sort(key=lambda row: row["time"])

    thetas = sorted({key[0] for key in grouped})
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 8.2))

    for i, theta in enumerate(thetas):
        runs = sorted((key[1], data) for key, data in grouped.items() if key[0] == theta)
        final_rows = [data[-1] for _, data in runs]
        dt = [row["dt"] for row in final_rows]
        state_error = [row["state_error"] for row in final_rows]
        covariance_error = [row["covariance_frobenius_error"] for row in final_rows]
        color = colors(i % 10)
        axes[0, 0].loglog(dt, state_error, marker="o", color=color,
                          label=rf"state, $\theta={theta:g}$")
        axes[0, 0].loglog(dt, covariance_error, marker="s", linestyle="--", color=color,
                          label=rf"covariance, $\theta={theta:g}$")

        orders = []
        for coarse, fine in zip(final_rows, final_rows[1:]):
            e0 = coarse["covariance_frobenius_error"]
            e1 = fine["covariance_frobenius_error"]
            orders.append((fine["dt"], math.log(e0 / e1, 2.0) if e0 > 0 and e1 > 0 else float("nan")))
        if orders:
            axes[0, 1].semilogx([item[0] for item in orders], [item[1] for item in orders],
                                marker="o", color=color, label=rf"$\theta={theta:g}$")

        finest = runs[-1][1]
        axes[1, 0].plot([r["time"] for r in finest], [r["cov_xy"] for r in finest],
                        color=color, label=rf"$\theta={theta:g}$")
        # Recompute correlation from the covariance columns. This both makes
        # the plotted quantity auditable and avoids relying on a possibly
        # undefined/exported correlation column.
        correlation_rows = []
        for row in finest:
            denominator_sq = row["var_x"] * row["var_y"]
            if denominator_sq > 0.0:
                rho = row["cov_xy"] / math.sqrt(denominator_sq)
                if math.isfinite(rho):
                    correlation_rows.append((row["time"], rho))
        if correlation_rows:
            axes[1, 1].plot([point[0] for point in correlation_rows],
                            [point[1] for point in correlation_rows],
                            color=color, label=rf"$\theta={theta:g}$")
        else:
            raise ValueError(
                f"no defined correlation values for theta={theta:g}; "
                "check var_x, cov_xy, and var_y in the CSV"
            )

    axes[0, 0].set_title("Final-time error versus step size")
    axes[0, 0].set_xlabel("Step size")
    axes[0, 0].set_ylabel("Absolute error")
    axes[0, 1].set_title("Observed covariance order")
    axes[0, 1].set_xlabel("Step size")
    axes[0, 1].set_ylabel("Observed order")
    axes[0, 1].axhline(2.0, color="0.25", linestyle="--", linewidth=1)
    axes[1, 0].set_title("Induced cross-covariance")
    axes[1, 0].set_xlabel("Time")
    axes[1, 0].set_ylabel(r"$P_{xy}(t)$")
    axes[1, 1].set_title("Normalized error correlation")
    axes[1, 1].set_xlabel("Time")
    axes[1, 1].set_ylabel(r"$\rho_{xy}(t)$")
    axes[1, 1].set_ylim(-1.05, 1.05)
    for axis in axes.flat:
        axis.grid(True, which="both", alpha=0.25)
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0], axes[1, 1]):
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("ASDoLE dynamical error propagation and cross-covariance")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("covariance_transport.png"))
    args = parser.parse_args()
    make_plot(read_rows(args.csv), args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
