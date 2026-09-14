#!/usr/bin/env python3
"""Plot aggregate O3 downscaling skill metrics."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

ORDER = [
    "LR-inter",
    "O3_retrain_reg",
    "O3_retrain_final",
    "PM2O3_frozen_reg",
    "PM2O3_frozen_final",
]
LABELS = ["Bilinear", "Reg", "CorrDiff", "Fine-tuned\n(Reg)", "Fine-tuned\n(CorrDiff)"]
COLORS = ["#9aa0a6", "#7fb0e0", "#1f5fa8", "#f3c08a", "#d2521f"]
METRICS = ["RMSE", "MAE", "R2", "R"]
TITLES = [r"RMSE ($\mu$g · m$^{-3}$)", r"MAE ($\mu$g · m$^{-3}$)", r"R$^2$", "Correlation R"]
YLIMITS = [(0, 3.3), (0, 2.3), (0.80, 1.0), (0.80, 1.0)]

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 28,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "savefig.bbox": "tight",
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot canonical model metrics.")
    parser.add_argument("--input", type=Path, default=Path("canonical_metrics.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--prefix", default="F1_aggregate_skill_v2")
    return parser.parse_args()


def main():
    args = parse_args()
    metrics = pd.read_csv(args.input, index_col="Model")
    values_by_metric = {
        metric: [metrics.loc[model, metric] for model in ORDER]
        for metric in METRICS
    }

    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    x = np.arange(len(LABELS))
    for ax, metric, title, ylim in zip(axes.flat, METRICS, TITLES, YLIMITS):
        values = values_by_metric[metric]
        bars = ax.bar(x, values, color=COLORS, edgecolor="white", linewidth=0.8, zorder=3)
        ax.set_title(title, fontsize=24)
        ax.set_xticks(x)
        ax.set_xticklabels(LABELS, fontsize=22, rotation=-30, ha="center")
        ax.set_ylim(ylim)
        ax.yaxis.set_major_locator(MaxNLocator(5))
        ax.tick_params(labelsize=24)
        ax.grid(axis="y", color="#E2E8F0", lw=0.6, zorder=0)
        offset = (ylim[1] - ylim[0]) * 0.02
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value + offset,
                f"{value:.2f}",
                ha="center",
                va="bottom",
                fontsize=22,
            )

    fig.tight_layout()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"{args.prefix}.{extension}")
    plt.close(fig)


if __name__ == "__main__":
    main()
