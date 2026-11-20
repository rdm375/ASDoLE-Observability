#!/usr/bin/env python3
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
"""Summarize parameter recovery versus observation noise and grid spacing."""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {
            "schedule", "steps", "dt", "relative_noise", "regularization",
            "error_y10", "error_k", "scaled_parameter_error", "residual_rms", "converged",
        }
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        rows = list(reader)
    if not rows:
        raise ValueError("CSV has no fit results")
    return rows


def summarize(rows: list[dict[str, str]]) -> list[dict[str, float | int | str]]:
    groups: dict[tuple[str, int, float, float], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        key = (row["schedule"], int(row["steps"]), float(row["relative_noise"]),
               float(row["regularization"]))
        groups[key].append(row)

    summaries = []
    for (schedule, steps, noise, regularization), group in sorted(groups.items()):
        count = len(group)
        summaries.append({
            "schedule": schedule,
            "steps": steps,
            "dt": float(group[0]["dt"]),
            "relative_noise": noise,
            "regularization": regularization,
            "realizations": count,
            "scaled_parameter_rmse": math.sqrt(sum(float(r["scaled_parameter_error"]) ** 2 for r in group) / count),
            "y10_rmse": math.sqrt(sum(float(r["error_y10"]) ** 2 for r in group) / count),
            "k_rmse": math.sqrt(sum(float(r["error_k"]) ** 2 for r in group) / count),
            "mean_residual_rms": sum(float(r["residual_rms"]) for r in group) / count,
            "convergence_rate": sum(int(r["converged"]) for r in group) / count,
        })
    return summaries


def write_summary(summaries: list[dict[str, float | int | str]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summaries[0]))
        writer.writeheader()
        writer.writerows(summaries)


def make_plot(summaries: list[dict[str, float | int | str]], output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    colors = {20: "tab:blue", 40: "tab:orange", 80: "tab:green"}
    styles = {0.0: "-", 1.0: "--", 10.0: ":"}
    for axis, schedule in zip(axes[:2], ("full_window", "early_window")):
        local = [r for r in summaries if r["schedule"] == schedule and r["relative_noise"] > 0.0]
        for steps in sorted({int(r["steps"]) for r in local}):
            for regularization in sorted({float(r["regularization"]) for r in local}):
                curve = sorted(
                    (r for r in local if int(r["steps"]) == steps
                     and float(r["regularization"]) == regularization),
                    key=lambda row: float(row["relative_noise"]),
                )
                if not curve:
                    continue
                axis.plot(
                    [float(r["relative_noise"]) for r in curve],
                    [float(r["scaled_parameter_rmse"]) for r in curve],
                    color=colors.get(steps), linestyle=styles.get(regularization, "-"),
                    marker="o", markersize=3.5,
                    label=rf"$\Delta t={4.0 / steps:g}$, $\lambda={regularization:g}$",
                )
        axis.set_xscale("log")
        axis.set_yscale("log")
        axis.set_title(schedule.replace("_", " ").title())
        axis.set_xlabel("Relative observation-noise standard deviation")
        axis.grid(True, which="both", alpha=0.25)
        axis.legend(frameon=False, fontsize=6.5, ncol=2)
    axes[0].set_ylabel("Scaled parameter-recovery RMSE")

    axis = axes[2]
    for schedule, color in (("full_window", "tab:blue"), ("early_window", "tab:orange")):
        curve = sorted(
            (r for r in summaries if r["schedule"] == schedule
             and float(r["relative_noise"]) == 0.0
             and float(r["regularization"]) == 0.0),
            key=lambda row: float(row["dt"]),
        )
        if curve:
            dts = [float(r["dt"]) for r in curve]
            errors = [float(r["scaled_parameter_rmse"]) for r in curve]
            axis.loglog(dts, errors, color=color, marker="o",
                        label=schedule.replace("_", " "))
            if schedule == "full_window":
                reference = [errors[0] * (dt / dts[0]) ** 2 for dt in dts]
                axis.loglog(dts, reference, color="black", linestyle="--",
                            linewidth=1.0, label=r"reference $O(\Delta t^2)$")
    axis.set_title("Noiseless discretization bias")
    axis.set_xlabel("Time step $\\Delta t$")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(frameon=False, fontsize=8)
    fig.suptitle("Inverse robustness: noise, observation window, regularization, and time step")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("inversion_robustness.png"))
    parser.add_argument("--summary-output", type=Path)
    args = parser.parse_args()
    summaries = summarize(read_rows(args.csv))
    summary_path = args.summary_output or args.output.with_name("inversion_robustness_summary.csv")
    write_summary(summaries, summary_path)
    make_plot(summaries, args.output)
    print(f"Summary CSV: {summary_path}")
    print(f"Plot: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
