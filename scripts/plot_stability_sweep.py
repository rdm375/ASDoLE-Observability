#!/usr/bin/env python3
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
"""Plot ASDoLE state, TLM, and adjoint amplification on the negative real axis."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_data(path: Path):
    grouped = defaultdict(lambda: {"x": [], "state": [], "tlm": [], "adjoint": []})
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"theta", "z", "state_gain", "tlm_gain", "adjoint_gain"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            theta = float(row["theta"])
            data = grouped[theta]
            data["x"].append(-float(row["z"]))
            data["state"].append(abs(float(row["state_gain"])))
            data["tlm"].append(abs(float(row["tlm_gain"])))
            data["adjoint"].append(abs(float(row["adjoint_gain"])))
    if not grouped:
        raise ValueError("CSV has no data")
    return grouped


def plot(data, output: Path):
    fig, (amplification_axis, defect_axis) = plt.subplots(
        2, 1, figsize=(9.5, 7.2), sharex=True, gridspec_kw={"height_ratios": [2, 1]}
    )
    colors = plt.get_cmap("tab10")
    for index, theta in enumerate(sorted(data)):
        color = colors(index % 10)
        row = data[theta]
        amplification_axis.loglog(row["x"], row["state"], color=color, linewidth=1.8,
                                  label=rf"$\theta={theta:g}$")
        defect = [max(abs(state - tlm), abs(state - adjoint))
                  for state, tlm, adjoint in zip(row["state"], row["tlm"], row["adjoint"])]
        defect_axis.loglog(row["x"], [max(value, 1.0e-17) for value in defect],
                           color=color, linewidth=1.2, label=rf"$\theta={theta:g}$")
    amplification_axis.axhline(1.0, color="black", linestyle="--", linewidth=1.1,
                               label="stability boundary")
    amplification_axis.set_ylabel("Absolute amplification")
    amplification_axis.set_title("ASDoLE state and sensitivity stability on the negative real axis")
    amplification_axis.grid(True, which="both", alpha=0.25)
    amplification_axis.legend(ncol=5, fontsize=9, frameon=False)
    defect_axis.axhline(2.0e-10, color="black", linestyle="--", linewidth=1.0,
                        label="comparison tolerance")
    defect_axis.set_xlabel(r"Stiffness parameter $-\lambda h$")
    defect_axis.set_ylabel("Max state/TLM/adjoint defect")
    defect_axis.grid(True, which="both", alpha=0.25)
    defect_axis.legend(fontsize=8, frameon=False)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("stability_sweep.png"))
    args = parser.parse_args()
    plot(load_data(args.csv), args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
