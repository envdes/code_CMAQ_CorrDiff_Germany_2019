#!/usr/bin/env python3
"""Plot transfer-learning performance as a function of training data."""

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

plt.rcParams.update(
    {
        "font.size": 28,
        "font.family": "DejaVu Sans",
        "axes.edgecolor": "#333333",
        "axes.linewidth": 0.9,
    }
)


def parse_args():
    parser = argparse.ArgumentParser(description="Plot data-efficiency results.")
    parser.add_argument("--input", type=Path, default=Path("data_efficiency.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("."))
    parser.add_argument("--prefix", default="figure9_transfer_learning_efficiency")
    return parser.parse_args()


def main():
    args = parse_args()
    data = pd.read_csv(args.input)
    labels = [f"{int(value):,}" for value in data["samples"]]
    x = np.arange(len(labels))
    width = 0.36

    rmse_finetuned = data["rmse_ft"].to_numpy()
    rmse_retrained = data["rmse_rt"].to_numpy()
    r2_finetuned = data["r2_ft"].to_numpy()
    r2_retrained = data["r2_rt"].to_numpy()

    color_finetuned = "#D9622B"
    color_retrained = "#2E75B6"
    negative_band = "#F7DED1"
    positive_band = "#E7EEF5"
    edge_color = "#4d4d4d"
    boundary = min(1.5, len(labels) - 0.5)

    fig, axes = plt.subplots(2, 1, figsize=(18, 16), sharex=True)

    def add_bands(axis):
        axis.axvspan(-0.5, boundary, color=negative_band, zorder=0, lw=0)
        axis.axvspan(boundary, len(labels) - 0.5, color=positive_band, zorder=0, lw=0)
        axis.axvline(boundary, color="#888888", ls=(0, (4, 3)), lw=1.0, zorder=1)

    def add_bars(axis, finetuned, retrained, offset):
        first = axis.bar(
            x - width / 2,
            finetuned,
            width=width,
            color=color_finetuned,
            edgecolor=edge_color,
            linewidth=0.6,
            zorder=3,
            label="Fine-tuned (frozen encoder, 100K steps)",
        )
        second = axis.bar(
            x + width / 2,
            retrained,
            width=width,
            color=color_retrained,
            edgecolor=edge_color,
            linewidth=0.6,
            zorder=3,
            label="Retrain from scratch (100K steps)",
        )
        for bars, color in ((first, color_finetuned), (second, color_retrained)):
            for bar in bars:
                axis.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + offset,
                    f"{bar.get_height():.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=22,
                    color=color,
                    fontweight="bold",
                )

    add_bands(axes[0])
    add_bars(axes[0], rmse_finetuned, rmse_retrained, 0.05)
    axes[0].set_ylabel(r"RMSE ($\mu$g · m$^{-3}$)", fontsize=28)
    axes[0].set_ylim(0, 5.6)
    axes[0].yaxis.set_minor_locator(MultipleLocator(0.5))
    axes[0].set_title("(a) RMSE", loc="left", fontsize=28)
    axes[0].grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
    axes[0].tick_params(labelsize=24)

    add_bands(axes[1])
    add_bars(axes[1], r2_finetuned, r2_retrained, 0.01)
    axes[1].set_ylabel(r"$R^2$", fontsize=28)
    axes[1].set_ylim(0.45, 1.02)
    axes[1].yaxis.set_minor_locator(MultipleLocator(0.05))
    axes[1].set_title(r"(b) $R^2$", loc="left", fontsize=28)
    axes[1].grid(axis="y", color="#dddddd", lw=0.6, zorder=0)
    axes[1].tick_params(labelsize=24)
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=22)
    axes[1].set_xlabel("O$_3$ training samples", fontsize=28)

    axes[0].text(
        np.mean((-0.5, boundary)),
        axes[0].get_ylim()[1] * 0.97,
        "Negative transfer",
        ha="center",
        va="top",
        fontsize=22,
        color="#a8481c",
        style="italic",
    )
    axes[0].text(
        np.mean((boundary, len(labels) - 0.5)),
        axes[0].get_ylim()[1] * 0.97,
        "Warm start advantageous",
        ha="center",
        va="top",
        fontsize=22,
        color="#1c4f7a",
        style="italic",
    )

    handles = [
        patches.Patch(
            facecolor=color_finetuned,
            edgecolor=edge_color,
            label="Fine-tuned (frozen encoder, 100K steps)",
        ),
        patches.Patch(
            facecolor=color_retrained,
            edgecolor=edge_color,
            label="Retrain from scratch (100K steps)",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=2, bbox_to_anchor=(0.5, 0.01), frameon=False, fontsize=24)
    for axis in axes:
        axis.set_axisbelow(True)

    plt.tight_layout(rect=[0, 0.04, 1, 1])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for extension in ("png", "pdf"):
        fig.savefig(args.output_dir / f"{args.prefix}.{extension}", bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
