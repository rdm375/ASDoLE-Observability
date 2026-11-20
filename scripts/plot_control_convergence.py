#!/usr/bin/env python3
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
"""Plot state-control convergence errors and observed orders."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


QUANTITIES = (
    ("state_error", "State max error"),
    ("tlm_error", "TLM state max error"),
    ("costate_error", "Costate max error"),
    ("objective_error", "Objective error"),
)


def read_data(path: Path):
    grouped = defaultdict(list)
    required = {"theta", "dt", "steps", *(key for key, _ in QUANTITIES)}
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                theta = float(row["theta"])
                dt = float(row["dt"])
                values = {key: float(row[key]) for key, _ in QUANTITIES}
                orders = {
                    key: None if row[f"{key.removesuffix('_error')}_order"].strip().upper() == "NA"
                    else float(row[f"{key.removesuffix('_error')}_order"])
                    for key, _ in QUANTITIES
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise ValueError(f"Invalid data on CSV line {line_number}: {exc}") from exc
            if not math.isfinite(dt) or dt <= 0:
                raise ValueError(f"dt must be finite and positive (line {line_number})")
            if any(not math.isfinite(value) or value <= 0 for value in values.values()):
                raise ValueError(f"errors must be finite and positive (line {line_number})")
            grouped[theta].append({"dt": dt, "values": values, "orders": orders})
    if not grouped:
        raise ValueError("CSV contains no data rows")
    for rows in grouped.values():
        rows.sort(key=lambda item: item["dt"])
    return grouped


def make_plot(data, output: Path):
    thetas = sorted(data)
    colors = plt.get_cmap("tab10")
    fig, axes = plt.subplots(2, 4, figsize=(15.5, 7.7), squeeze=False)

    for column, (key, label) in enumerate(QUANTITIES):
        error_axis = axes[0, column]
        order_axis = axes[1, column]
        error_axis.set_title(label)
        for index, theta in enumerate(thetas):
            rows = data[theta]
            color = colors(index % 10)
            error_axis.loglog(
                [row["dt"] for row in rows],
                [row["values"][key] for row in rows],
                marker="o",
                markersize=3.5,
                linewidth=1.5,
                color=color,
                label=rf"$\theta={theta:g}$",
            )
            order_rows = [row for row in rows if row["orders"][key] is not None]
            order_axis.semilogx(
                [row["dt"] for row in order_rows],
                [row["orders"][key] for row in order_rows],
                marker="o",
                markersize=3.5,
                linewidth=1.5,
                color=color,
            )

        error_axis.grid(True, which="both", alpha=0.25)
        order_axis.grid(True, which="both", alpha=0.25)
        order_axis.axhline(2.0, color="0.3", linewidth=1.1, linestyle="--")
        order_axis.set_ylim(bottom=0)
        error_axis.invert_xaxis()
        order_axis.invert_xaxis()
        order_axis.set_xlabel("Time step size, dt")

    axes[0, 0].set_ylabel("Absolute error")
    axes[1, 0].set_ylabel("Observed order")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 1.02),
               ncol=len(handles), frameon=False)
    fig.suptitle("ASDoLE discrete state-control convergence", y=1.10, fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="CSV emitted by asdole_decay_control_convergence")
    parser.add_argument("--output", type=Path, default=Path("control_convergence.png"))
    args = parser.parse_args()
    make_plot(read_data(args.csv), args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
