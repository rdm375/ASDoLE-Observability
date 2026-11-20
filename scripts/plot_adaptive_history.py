#!/usr/bin/env python3
# Copyright (c) 2026 Richard Douglas Myers
# SPDX-License-Identifier: AGPL-3.0-only
"""Plot accepted and rejected ASDoLE adaptive step attempts."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def read_rows(path: Path):
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        required = {"theta", "t_start", "h", "error_norm", "accepted", "h_next"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"CSV must contain columns: {', '.join(sorted(required))}")
        for line_number, row in enumerate(reader, start=2):
            try:
                parsed = {
                    "t": float(row["t_start"]),
                    "h": float(row["h"]),
                    "error": float(row["error_norm"]),
                    "accepted": bool(int(row["accepted"])),
                }
                theta = float(row["theta"])
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid CSV row {line_number}: {exc}") from exc
            grouped[theta].append(parsed)
    if not grouped:
        raise ValueError("CSV contains no step attempts")
    return grouped


def plot_history(grouped, output: Path):
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    colors = plt.get_cmap("tab10")
    for index, theta in enumerate(sorted(grouped)):
        rows = grouped[theta]
        color = colors(index % 10)
        accepted = [row for row in rows if row["accepted"]]
        rejected = [row for row in rows if not row["accepted"]]
        axes[0].semilogy([r["t"] + 0.5*r["h"] for r in accepted], [r["h"] for r in accepted],
                         ".-", markersize=3, linewidth=1.0, color=color, label=rf"$\theta={theta:g}$")
        if rejected:
            axes[0].scatter([r["t"] + 0.5*r["h"] for r in rejected], [r["h"] for r in rejected],
                            marker="x", s=22, color=color)
        axes[1].semilogy([r["t"] + 0.5*r["h"] for r in accepted], [r["error"] for r in accepted],
                         ".", markersize=3, color=color, label=rf"$\theta={theta:g}$")
        if rejected:
            axes[1].scatter([r["t"] + 0.5*r["h"] for r in rejected], [r["error"] for r in rejected],
                            marker="x", s=22, color=color)

    for axis in axes:
        axis.axvspan(0.42, 0.58, color="gold", alpha=0.18, label="high-rate transient")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Attempted step size")
    axes[0].set_title("ASDoLE adaptive step-size history (× marks rejected steps)")
    axes[1].set_ylabel("Scaled error estimate")
    axes[1].axhline(1.0, color="black", linestyle="--", linewidth=1.0)
    axes[1].set_xlabel("Time")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=5, frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180, bbox_inches="tight")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("adaptive_history.png"))
    args = parser.parse_args()
    plot_history(read_rows(args.csv), args.output)
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
