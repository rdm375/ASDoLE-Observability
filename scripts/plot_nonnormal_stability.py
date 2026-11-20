#!/usr/bin/env python3
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
"""Plot component histories and norms for the non-normal stability case."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_data(path: Path):
    data = defaultdict(lambda: {
        "t": [], "state": [], "tlm": [], "adjoint": [], "optimal_gain": [],
        "state_x1": [], "state_x2": [], "tlm_x1": [], "tlm_x2": [],
        "adjoint_x1": [], "adjoint_x2": [],
    })
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "theta", "time", "state_norm", "tlm_norm", "adjoint_norm", "optimal_gain",
            "state_x1", "state_x2", "tlm_x1", "tlm_x2",
            "adjoint_x1", "adjoint_x2",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        for row in reader:
            theta = float(row["theta"])
            group = data[theta]
            group["t"].append(float(row["time"]))
            group["state"].append(float(row["state_norm"]))
            group["tlm"].append(float(row["tlm_norm"]))
            group["adjoint"].append(float(row["adjoint_norm"]))
            group["optimal_gain"].append(float(row["optimal_gain"]))
            for key in ("state_x1", "state_x2", "tlm_x1", "tlm_x2",
                        "adjoint_x1", "adjoint_x2"):
                group[key].append(float(row[key]))
    if not data:
        raise ValueError("CSV has no data")
    return data


def plot_norms(data, output: Path):
    thetas = sorted(data)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), sharex=True, sharey=True)
    for axis, theta in zip(axes.flat, thetas):
        group = data[theta]
        axis.plot(group["t"], group["state"], color="black", linewidth=1.7, label="state")
        axis.plot(group["t"], group["tlm"], color="tab:blue", linewidth=1.5, label="TLM perturbation")
        axis.plot(group["t"], group["adjoint"], color="tab:red", linewidth=1.5, label="adjoint covector")
        axis.plot(group["t"], group["optimal_gain"], color="tab:green", linewidth=1.4,
                  linestyle="--", label="worst-case TLM gain")
        axis.set_title(rf"$\theta={theta:g}$")
        axis.grid(True, alpha=0.25)
        axis.set_yscale("log")
    axes[1, 0].set_xlabel("Time")
    axes[1, 1].set_xlabel("Time")
    axes[0, 0].set_ylabel("Euclidean norm")
    axes[1, 0].set_ylabel("Euclidean norm")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.96),
               ncol=4, frameon=False)
    fig.suptitle("Transient growth in a stable non-normal system", y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_components(data, output: Path):
    """Plot signed coordinates so directional growth is visible."""
    thetas = sorted(data)
    fig, axes = plt.subplots(3, 2, figsize=(12, 10), sharex=True)
    rows = (("state", "State", "black"),
            ("tlm", "TLM perturbation", "tab:blue"),
            ("adjoint", "Adjoint covector", "tab:red"))
    for row, (prefix, title, color) in enumerate(rows):
        for component in (1, 2):
            axis = axes[row, component - 1]
            for theta in thetas:
                group = data[theta]
                axis.plot(group["t"], group[f"{prefix}_x{component}"],
                          linewidth=1.5, label=rf"$\theta={theta:g}$")
            axis.axhline(0.0, color="0.4", linewidth=0.7)
            axis.set_title(f"{title}: component {component}")
            axis.grid(True, alpha=0.25)
            axis.set_ylabel("Value")
            if component == 1:
                axis.legend(frameon=False, ncol=2, fontsize=8)
    axes[-1, 0].set_xlabel("Forward time $t$")
    axes[-1, 1].set_xlabel("Forward time $t$")
    fig.suptitle("State, forward TLM perturbation, and backward adjoint history", y=0.995)
    fig.text(0.5, 0.01,
             "TLM starts at t=0; adjoint is initialized at T and propagated backward.",
             ha="center", fontsize=9)
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("nonnormal_stability.png"))
    parser.add_argument("--components-output", type=Path)
    args = parser.parse_args()
    data = read_data(args.csv)
    plot_norms(data, args.output)
    print(f"Wrote {args.output}")
    if args.components_output:
        plot_components(data, args.components_output)
        print(f"Wrote {args.components_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
