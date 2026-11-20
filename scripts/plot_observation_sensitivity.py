#!/usr/bin/env python3
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
"""Plot TLM and adjoint observation sensitivities from the ASDoLE example."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_history(path: Path):
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = {
        "time", "d_observation_d_initial_y1", "d_observation_d_initial_y2",
        "terminal_adjoint_y1", "terminal_adjoint_y2",
        "objective_costate_y1", "objective_costate_y2",
    }
    if not rows or not required.issubset(rows[0]):
        raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
    return {name: [float(row[name]) for row in rows] for name in required}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("observation_sensitivity.png"))
    args = parser.parse_args()
    data = read_history(args.csv)

    fig, axes = plt.subplots(3, 1, figsize=(9, 10), sharex=True)
    axes[0].plot(data["time"], data["d_observation_d_initial_y1"],
                 label=r"$\partial y_1(t)/\partial y_1(0)$", linewidth=1.8)
    axes[0].plot(data["time"], data["d_observation_d_initial_y2"],
                 label=r"$\partial y_1(t)/\partial y_2(0)$", linewidth=1.8)
    axes[0].set_ylabel("TLM observation sensitivity")
    axes[0].set_title("How initial-state perturbations appear in the sensor")
    axes[0].legend(frameon=False)

    axes[1].plot(data["time"], data["terminal_adjoint_y1"],
                 label=r"terminal $y_1(T)$: state component 1", linewidth=1.8)
    axes[1].plot(data["time"], data["terminal_adjoint_y2"],
                 label=r"terminal $y_1(T)$: state component 2", linewidth=1.8)
    axes[1].set_ylabel("Adjoint sensitivity")
    axes[1].set_title("Sensitivity of terminal observation $y_1(T)$ to state perturbations")
    axes[1].legend(frameon=False, fontsize=8)

    axes[2].plot(data["time"], data["objective_costate_y1"],
                 label="integrated objective: component 1", linewidth=1.8)
    axes[2].plot(data["time"], data["objective_costate_y2"],
                 label="integrated objective: component 2", linewidth=1.8)
    axes[2].set_xlabel("Time")
    axes[2].set_ylabel("Costate")
    axes[2].set_title("Sensitivity of the time-integrated least-squares objective")
    axes[2].legend(frameon=False)

    for axis in axes:
        axis.grid(True, alpha=0.25)
    fig.tight_layout()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
