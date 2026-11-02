#!/usr/bin/env python3
"""Render the benchmark figures from `results/rtx4090.json`.

The plots are generated from the committed measurement file rather than drawn by
hand, so re-running the suite and overwriting that JSON refreshes every figure.
Outputs SVG for the README (crisp at any zoom) and PNG for quick inspection.

Usage:
    python3 scripts/plot_results.py
"""

import json
import math
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "results", "rtx4090.json")
FIG_DIR = os.path.join(ROOT, "figs")

# One colour per optimisation stage, reused across figures.
COLOURS = {
    "k0": "#9e9e9e",
    "k1": "#c0392b",
    "k2": "#d97b23",
    "k3": "#c9a227",
    "k4": "#4f9d3a",
    "k5": "#2c7fb8",
    "k6": "#34568b",
    "k10": "#5b2c83",
}


def style_axes(ax):
    """Apply a restrained, print-friendly style."""
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color("#bbbbbb")
    ax.spines["bottom"].set_color("#bbbbbb")
    ax.tick_params(colors="#444444", labelsize=9)
    ax.grid(True, which="major", color="#e6e6e6", linewidth=0.8)
    ax.set_axisbelow(True)


def plot_vs_size(data):
    """GFLOPS against problem size, one line per kernel."""
    fig, ax = plt.subplots(figsize=(8.4, 4.4), dpi=160)
    style_axes(ax)

    sizes = data["sizes"]
    # The legend is ordered by final throughput, which is the order a reader
    # scans the right-hand edge in; labels on the lines would collide up there.
    ordered = sorted(data["kernels"], key=lambda item: item["gflops"][-1], reverse=True)
    for kernel in ordered:
        colour = COLOURS.get(kernel["id"], "#333333")
        solid = kernel["id"] != "k0"
        ax.plot(
            sizes,
            kernel["gflops"],
            marker="o",
            markersize=3.2,
            linewidth=1.9 if solid else 1.4,
            linestyle="-" if solid else "--",
            color=colour,
            label=f"{kernel['id']}  {kernel['short']}",
        )

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xticks(sizes)
    ax.set_xticklabels([str(size) for size in sizes])
    ax.set_xlabel("Matrix size (M = N = K)", fontsize=10)
    ax.set_ylabel("GFLOPS (log scale)", fontsize=10)
    ax.set_title(
        f"SGEMM throughput vs. problem size\n{data['meta']['gpu']} (FP32)",
        fontsize=11,
        pad=10,
    )
    ax.legend(
        fontsize=8,
        loc="center left",
        bbox_to_anchor=(1.015, 0.5),
        frameon=False,
        title="4096³ GFLOPS",
        title_fontsize=8,
    )
    ax.set_xlim(sizes[0] * 0.85, sizes[-1] * 1.12)

    fig.tight_layout()
    return fig


def plot_at_4096(data):
    """Horizontal bars at 4096 with the fraction of cuBLAS annotated."""
    fig, ax = plt.subplots(figsize=(7.2, 4.0), dpi=160)
    style_axes(ax)
    ax.grid(True, axis="x", color="#e6e6e6", linewidth=0.8)
    ax.grid(False, axis="y")

    kernels = list(reversed(data["kernels"]))
    labels = [f"{k['id']}  {k['short']}" for k in kernels]
    values = [k["gflops"][-1] for k in kernels]
    reference = values[-1]  # the first element of the reversed list is cuBLAS
    colours = [COLOURS.get(k["id"], "#333333") for k in kernels]

    bars = ax.barh(labels, values, color=colours, height=0.62)
    for bar, value in zip(bars, values):
        ax.text(
            value + reference * 0.012,
            bar.get_y() + bar.get_height() / 2,
            f"{value:,.0f}  ({value / reference * 100:.1f}%)",
            va="center",
            fontsize=8,
            color="#333333",
        )

    peak = data["meta"]["nominal_peak_tflops"] * 1000
    ax.set_xlim(0, max(reference * 1.34, peak * 1.10))
    ax.axvline(peak, color="#888888", linestyle=":", linewidth=1.1)
    ax.text(peak * 1.012, len(kernels) - 0.55,
            f"nominal FP32 peak  {peak / 1000:.1f} TFLOPS",
            fontsize=7.5, color="#666666", va="top", rotation=0)

    ax.xaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda value, _pos: f"{value:,.0f}"))
    ax.set_xlabel("GFLOPS at 4096 x 4096 x 4096", fontsize=10)
    ax.set_title(
        f"FP32 SGEMM at 4096³ vs. cuBLAS on {data['meta']['gpu']}",
        fontsize=11,
        pad=10,
    )

    fig.tight_layout()
    return fig


def save(figure, name):
    """Write one figure as SVG (README) and PNG (inspection)."""
    os.makedirs(FIG_DIR, exist_ok=True)
    svg = os.path.join(FIG_DIR, f"{name}.svg")
    png = os.path.join(FIG_DIR, f"{name}.png")
    figure.savefig(svg, format="svg", bbox_inches="tight")
    figure.savefig(png, format="png", dpi=160, bbox_inches="tight")
    plt.close(figure)
    print(f"wrote {svg}")


def main():
    if not os.path.exists(DATA):
        print(f"missing {DATA}", file=sys.stderr)
        return 1
    with open(DATA, "r", encoding="utf-8") as handle:
        data = json.load(handle)

    save(plot_vs_size(data), "performance_vs_size")
    save(plot_at_4096(data), "performance_at_4096")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
