#!/usr/bin/env python3
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
"""Compare covariance caused by a fixed parameter and stepwise forcing noise."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FIELDS = (
    "time", "param_var_x", "param_cov_xy", "param_var_y",
    "forcing_var_x", "forcing_cov_xy", "forcing_var_y", "forcing_intensity",
)


def read_csv(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        if not reader.fieldnames or not set(FIELDS).issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(FIELDS)}")
        rows = []
        for line_number, row in enumerate(reader, start=2):
            try:
                parsed = {key: float(row[key]) for key in FIELDS}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"invalid value on CSV line {line_number}: {exc}") from exc
            if not all(math.isfinite(value) for value in parsed.values()):
                raise ValueError(f"non-finite data on CSV line {line_number}")
            rows.append(parsed)
    if not rows:
        raise ValueError("CSV contains no rows")
    return rows


def plot(rows, output: Path):
    time = [row["time"] for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(15.0, 4.8))
    axes[0].plot(time, [row["param_var_y"] for row in rows],
                 label="fixed uncertain coupling")
    axes[0].plot(time, [row["forcing_var_y"] for row in rows],
                 label="independent forcing increments", linestyle="--")
    axes[0].set_title(r"Added variance in $y$")
    axes[0].set_ylabel(r"$\Delta P_{yy}$")

    axes[1].plot(time, [row["param_var_x"] for row in rows],
                 label="fixed uncertain coupling")
    axes[1].plot(time, [row["forcing_var_x"] for row in rows],
                 label="independent forcing increments", linestyle="--")
    axes[1].set_title(r"Added variance in $x$")
    axes[1].set_ylabel(r"$\Delta P_{xx}$")

    axes[2].plot(time, [row["param_cov_xy"] for row in rows],
                 label="fixed uncertain coupling")
    axes[2].plot(time, [row["forcing_cov_xy"] for row in rows],
                 label="independent forcing increments", linestyle="--")
    axes[2].set_title("Added state cross-covariance")
    axes[2].set_ylabel(r"$\Delta P_{xy}$")

    for axis in axes:
        axis.set_xlabel("Time")
        axis.grid(True, alpha=0.25)
        axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Persistent parameter error versus stepwise forcing noise")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("error_source_comparison.png"))
    args = parser.parse_args()
    plot(read_csv(args.csv), args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
