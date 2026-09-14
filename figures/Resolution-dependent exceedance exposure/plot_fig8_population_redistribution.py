
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

from figure_style_config import (
    FIG_WIDTH_IN as DEFAULT_FIG_WIDTH_IN,
    NEGATIVE_COLORS, POSITIVE_COLORS, NEGATIVE_BACKGROUND, POSITIVE_BACKGROUND,
    apply_style, save_exact_size, print_size_report,
)
from exposure_helpers import BINS, bin_population, load_data, masked_population

FONT_FAMILY = "Times New Roman"
FONT_SERIF_FALLBACKS = [
    FONT_FAMILY,
    "Times",
    "Liberation Serif",
    "DejaVu Serif",
]

FIG_WIDTH_IN = DEFAULT_FIG_WIDTH_IN
FIG_HEIGHT_IN = 5.55

PANEL_TITLE_PT = 21
AXIS_LABEL_PT_LOCAL = 19
TICK_PT_LOCAL = 19
CATEGORY_HEADER_PT = 19
LEGEND_PT_LOCAL = 19

AXIS_LINEWIDTH = 1.25
TICK_WIDTH = 1.20
TICK_LENGTH = 5.0
GRID_LINEWIDTH = 0.55
GRID_ALPHA = 0.34
BAR_EDGE_LINEWIDTH = 0.75
WHISKER_LINEWIDTH = 1.25

LEFT = 0.105
RIGHT = 0.985
BOTTOM = 0.335
TOP = 0.815
WSPACE = 0.055

BAR_HEIGHT = 0.68
TITLE_PAD = 10
X_LABEL_PAD = 9
CATEGORY_HEADER_Y = 0.925
LEGEND_LOC = "lower center"
LEGEND_BBOX = (0.50, 0.045)
OUTPUT_BASENAME = "fig8_population_redistribution_shared_y_modular_local_style"


def apply_local_style() -> None:
    apply_style()
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_SERIF_FALLBACKS,
        "font.size": TICK_PT_LOCAL,
        "axes.titlesize": PANEL_TITLE_PT,
        "axes.labelsize": AXIS_LABEL_PT_LOCAL,
        "xtick.labelsize": TICK_PT_LOCAL,
        "ytick.labelsize": TICK_PT_LOCAL,
        "legend.fontsize": LEGEND_PT_LOCAL,
        "axes.linewidth": AXIS_LINEWIDTH,
        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def draw_minmax(ax, y: float, low: float, high: float) -> None:
    cap = 0.12
    ax.plot([low, high], [y, y], color="black", linestyle="--",
            linewidth=WHISKER_LINEWIDTH, zorder=6)
    ax.plot([low, low], [y-cap, y+cap], color="black",
            linewidth=WHISKER_LINEWIDTH, zorder=6)
    ax.plot([high, high], [y-cap, y+cap], color="black",
            linewidth=WHISKER_LINEWIDTH, zorder=6)


def main() -> None:
    apply_local_style()
    data = load_data()

    mask = data["mask"].astype(bool)
    population = masked_population(data["pop"].astype(float), mask)
    hr = data["delta_hr_lr"].astype(float)
    members = data["delta_cd_member"].astype(float)

    hr_positive, hr_negative = bin_population(hr, population)
    cd_positive = np.zeros((members.shape[0], len(BINS)))
    cd_negative = np.zeros_like(cd_positive)
    for member in range(members.shape[0]):
        cd_positive[member], cd_negative[member] = bin_population(
            members[member], population,
        )

    cd_positive_mean = cd_positive.mean(axis=0)
    cd_negative_mean = cd_negative.mean(axis=0)
    cd_positive_min = cd_positive.min(axis=0)
    cd_positive_max = cd_positive.max(axis=0)
    cd_negative_min = cd_negative.min(axis=0)
    cd_negative_max = cd_negative.max(axis=0)

    axis_max = 1.12 * max(
        hr_positive.max(), hr_negative.max(),
        cd_positive_max.max(), cd_negative_max.max(),
    )

    labels = [item[0] for item in BINS]
    y = np.arange(len(labels))

    fig, axes = plt.subplots(
        1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharey=True,
    )
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP, wspace=WSPACE)

    for ax in axes:
        ax.axvspan(-axis_max, 0, color=NEGATIVE_BACKGROUND, alpha=0.90, zorder=0)
        ax.axvspan(0, axis_max, color=POSITIVE_BACKGROUND, alpha=0.90, zorder=0)

    axes[0].barh(y, -hr_negative, color=NEGATIVE_COLORS, edgecolor="0.25",
                 linewidth=BAR_EDGE_LINEWIDTH, height=BAR_HEIGHT, zorder=3)
    axes[0].barh(y, hr_positive, color=POSITIVE_COLORS, edgecolor="0.25",
                 linewidth=BAR_EDGE_LINEWIDTH, height=BAR_HEIGHT, zorder=3)
    axes[1].barh(y, -cd_negative_mean, color=NEGATIVE_COLORS, edgecolor="0.25",
                 linewidth=BAR_EDGE_LINEWIDTH, height=BAR_HEIGHT, zorder=3)
    axes[1].barh(y, cd_positive_mean, color=POSITIVE_COLORS, edgecolor="0.25",
                 linewidth=BAR_EDGE_LINEWIDTH, height=BAR_HEIGHT, zorder=3)

    for i in range(len(BINS)):
        draw_minmax(axes[1], y[i], -cd_negative_max[i], -cd_negative_min[i])
        draw_minmax(axes[1], y[i], cd_positive_min[i], cd_positive_max[i])

    titles = ["(a) HR (4 km) - CMAQ (36 km)", "(b) CorrDiff - CMAQ (36 km)"]
    for ax, title in zip(axes, titles):
        ax.axvline(0, color="black", linewidth=1.15, zorder=5)
        ax.set_xlim(-axis_max, axis_max)
        ax.set_title(title, fontsize=PANEL_TITLE_PT, pad=TITLE_PAD)
        ax.set_xlabel("Population (million)", fontsize=AXIS_LABEL_PT_LOCAL, labelpad=X_LABEL_PAD)
        ax.tick_params(
            axis="both",
            labelsize=TICK_PT_LOCAL,
            width=TICK_WIDTH,
            length=TICK_LENGTH,
            pad=4,
        )
        ax.grid(
            axis="x",
            linestyle="--",
            linewidth=GRID_LINEWIDTH,
            alpha=GRID_ALPHA,
        )
        ax.set_axisbelow(True)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.xaxis.set_major_formatter(lambda value, _: f"{abs(value):g}")

    axes[0].set_yticks(y, labels)
    axes[0].set_ylabel("")
    axes[1].tick_params(axis="y", which="both", left=False, labelleft=False)

    fig.text(
        0.50, CATEGORY_HEADER_Y,
        r"Absolute $\Delta\mathrm{DOT}_{25}$ category (days)",
        ha="center", va="center", fontsize=CATEGORY_HEADER_PT,
    )

    handles = [
        Patch(facecolor="#2171b5", edgecolor="0.25",
              label=r"Negative $\Delta\mathrm{DOT}_{25}$"),
        Patch(facecolor="#e34a33", edgecolor="0.25",
              label=r"Positive $\Delta\mathrm{DOT}_{25}$"),
    ]
    fig.legend(
        handles=handles, loc=LEGEND_LOC, bbox_to_anchor=LEGEND_BBOX,
        frameon=False, fontsize=LEGEND_PT_LOCAL, ncol=2,
        handlelength=1.25, handleheight=0.75,
        columnspacing=1.5, labelspacing=0.30, borderaxespad=0.0,
    )

    png, pdf = save_exact_size(fig, OUTPUT_BASENAME)
    plt.close(fig)

    print_size_report(FIG_HEIGHT_IN)
    print("Figure 8 uses sharey=True and hides duplicate right-panel category labels")
    print(f"Saved: {png.name}, {pdf.name}")


if __name__ == "__main__":
    main()
