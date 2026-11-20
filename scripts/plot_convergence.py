#!/usr/bin/env python3
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
"""Plot convergence CSV emitted by asdole_decay."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


METHODS = {
    "theta_method": ("Theta method", "#1f77b4", "-"),
    "affine_doubled_step": ("Doubled affine step D", "#ff7f0e", "--"),
    "asdole_extrapolated": ("ASDoLE 2D-S", "#2ca02c", "-"),
}


def read_data(csv_path: Path):
    grouped = defaultdict(list)
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"method", "theta", "dt", "global_error", "observed_order"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                key = (row["method"], float(row["theta"]))
                dt = float(row["dt"])
                error = float(row["global_error"])
                order = None if row["observed_order"].strip().upper() == "NA" else float(row["observed_order"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid data on CSV line {line_number}: {exc}") from exc
            if not (math.isfinite(dt) and dt > 0 and math.isfinite(error) and error > 0):
                raise ValueError(f"dt and global_error must be finite and positive (line {line_number})")
            if order is not None and not math.isfinite(order):
                raise ValueError(f"observed_order must be finite or NA (line {line_number})")
            grouped[key].append({"dt": dt, "error": error, "order": order})
    if not grouped:
        raise ValueError("CSV contains no data rows")
    return grouped


def make_plot(data, output_path: Path):
    thetas = sorted({theta for _, theta in data})
    fig, axes = plt.subplots(
        2,
        len(thetas),
        figsize=(4.0 * len(thetas), 7.0),
        squeeze=False,
        sharex="col",
    )

    for column, theta in enumerate(thetas):
        error_axis = axes[0, column]
        order_axis = axes[1, column]
        error_axis.set_title(rf"$\theta = {theta:g}$")

        for method, (label, color, linestyle) in METHODS.items():
            points = sorted(data.get((method, theta), []), key=lambda item: item["dt"])
            if not points:
                continue
            dts = [item["dt"] for item in points]
            errors = [item["error"] for item in points]
            orders = [(item["dt"], item["order"]) for item in points if item["order"] is not None]

            error_axis.loglog(
                dts,
                errors,
                marker="o",
                markersize=4,
                linewidth=1.6,
                linestyle=linestyle,
                color=color,
                label=label,
            )
            if orders:
                order_axis.semilogx(
                    [item[0] for item in orders],
                    [item[1] for item in orders],
                    marker="o",
                    markersize=4,
                    linewidth=1.6,
                    linestyle=linestyle,
                    color=color,
                    label=label,
                )

        error_axis.grid(True, which="both", alpha=0.25)
        order_axis.grid(True, which="both", alpha=0.25)
        error_axis.invert_xaxis()
        order_axis.invert_xaxis()
        order_axis.axhline(1.0, color="0.35", linewidth=1, linestyle=":")
        order_axis.axhline(2.0, color="0.35", linewidth=1, linestyle=":")
        order_axis.set_ylim(bottom=0)
        order_axis.set_xlabel("Time step size, dt")

    axes[0, 0].set_ylabel("Global error at final time")
    axes[1, 0].set_ylabel("Observed convergence order")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.04),
        ncol=len(handles),
        frameon=False,
    )
    fig.suptitle("Decay-equation convergence", y=1.13, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.90))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV output from asdole_decay")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("convergence.png"),
        help="output PNG path (default: convergence.png)",
    )
    args = parser.parse_args()

    make_plot(read_data(args.csv), args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
