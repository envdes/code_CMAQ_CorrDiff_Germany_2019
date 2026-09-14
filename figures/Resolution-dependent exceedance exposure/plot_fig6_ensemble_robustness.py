
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, LinearSegmentedColormap, ListedColormap
import numpy as np

from figure_style_config import (
    FIG_WIDTH_IN,
    TITLE_PT as DEFAULT_TITLE_PT,
    AXIS_LABEL_PT as DEFAULT_AXIS_LABEL_PT,
    TICK_PT as DEFAULT_TICK_PT,
    COLORBAR_LABEL_PT as DEFAULT_COLORBAR_LABEL_PT,
    COLORBAR_TICK_PT as DEFAULT_COLORBAR_TICK_PT,
    SEQUENTIAL_BLUE, SEQUENTIAL_WARM, TICK_WIDTH, TICK_LENGTH,
    apply_style, save_exact_size, print_size_report,
)
from exposure_helpers import load_data, configure_map_axis

FIG_HEIGHT_IN = 4.55

AX1_POS = [0.150, 0.18, 0.30, 0.66]
CB1_POS = [0.462, 0.2857, 0.015, 0.4486]
AX2_POS = [0.585, 0.18, 0.30, 0.66]
CB2_POS = [0.895, 0.2857, 0.015, 0.4486]

TITLE_PAD = 10
COLORBAR_LABEL_PAD = 8
ROBUST_SPREAD_PERCENTILE = 99
OUTPUT_BASENAME = "fig6_ensemble_robustness_modular"

PANEL_TITLE_PT = 21
AXIS_LABEL_PT_LOCAL = 21
TICK_PT_LOCAL = 21
COLORBAR_LABEL_PT_LOCAL = 21
COLORBAR_TICK_PT_LOCAL = 21


def apply_local_style() -> None:
    apply_style()
    matplotlib.rcParams.update({
        "font.size": TICK_PT_LOCAL,
        "axes.titlesize": PANEL_TITLE_PT,
        "axes.labelsize": AXIS_LABEL_PT_LOCAL,
        "xtick.labelsize": TICK_PT_LOCAL,
        "ytick.labelsize": TICK_PT_LOCAL,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def style_map_axis(ax) -> None:
    ax.tick_params(
        axis="both", labelsize=TICK_PT_LOCAL,
        width=TICK_WIDTH, length=TICK_LENGTH, pad=5,
    )
    ax.xaxis.label.set_size(AXIS_LABEL_PT_LOCAL)
    ax.yaxis.label.set_size(AXIS_LABEL_PT_LOCAL)


def main() -> None:
    apply_local_style()
    data = load_data()

    mask = data["mask"].astype(bool)
    delta = data["delta_cd_member"].astype(float)
    lats = data["lats"]
    lons = data["lons"]

    n_positive = np.count_nonzero(delta > 0, axis=0)
    n_negative = np.count_nonzero(delta < 0, axis=0)
    n_zero = np.count_nonzero(delta == 0, axis=0)
    majority_count = np.maximum.reduce([n_positive, n_negative, n_zero])
    agreement = majority_count / delta.shape[0]
    spread = np.std(delta, axis=0, ddof=0)

    agreement_plot = np.where(mask, agreement, np.nan)
    spread_plot = np.where(mask, spread, np.nan)

    agreement_levels = np.arange(3, 9) / 8
    agreement_boundaries = np.arange(2.5, 9.0, 1.0) / 8
    agreement_cmap = ListedColormap(SEQUENTIAL_BLUE[1:7], name="member_agreement")
    agreement_norm = BoundaryNorm(agreement_boundaries, agreement_cmap.N)
    spread_cmap = LinearSegmentedColormap.from_list(
        "correction_spread", SEQUENTIAL_WARM, N=256,
    )
    spread_limit = float(np.percentile(spread[mask], ROBUST_SPREAD_PERCENTILE))

    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax1 = fig.add_axes(AX1_POS)
    cax1 = fig.add_axes(CB1_POS)
    ax2 = fig.add_axes(AX2_POS)
    cax2 = fig.add_axes(CB2_POS)

    configure_map_axis(ax1, lons, lats, mask, show_ylabel=True)
    style_map_axis(ax1)
    m1 = ax1.pcolormesh(
        lons, lats, agreement_plot,
        cmap=agreement_cmap, norm=agreement_norm,
        shading="auto", edgecolors="face", linewidth=0.10,
        rasterized=True, zorder=1,
    )
    ax1.set_title("(a) Directional agreement", fontsize=PANEL_TITLE_PT, pad=TITLE_PAD)
    cb1 = fig.colorbar(
        m1, cax=cax1, orientation="vertical",
        boundaries=agreement_boundaries, ticks=agreement_levels,
    )
    cb1.set_ticklabels([f"{level:.3f}" for level in agreement_levels])
    cb1.set_label("Member fraction", fontsize=COLORBAR_LABEL_PT_LOCAL,
                  labelpad=COLORBAR_LABEL_PAD)
    cb1.ax.tick_params(labelsize=COLORBAR_TICK_PT_LOCAL, width=TICK_WIDTH,
                       length=TICK_LENGTH, pad=4)

    configure_map_axis(ax2, lons, lats, mask, show_ylabel=False)
    style_map_axis(ax2)
    m2 = ax2.pcolormesh(
        lons, lats, spread_plot,
        cmap=spread_cmap, vmin=0.0, vmax=spread_limit,
        shading="auto", edgecolors="face", linewidth=0.10,
        rasterized=True, zorder=1,
    )
    ax2.set_title("(b) Ensemble spread", fontsize=PANEL_TITLE_PT, pad=TITLE_PAD)
    cb2 = fig.colorbar(m2, cax=cax2, orientation="vertical", extend="max")
    cb2.set_label(r"SD $\Delta\mathrm{DOT}_{25}$ (days)",
                  fontsize=COLORBAR_LABEL_PT_LOCAL, labelpad=COLORBAR_LABEL_PAD)
    cb2.ax.tick_params(labelsize=COLORBAR_TICK_PT_LOCAL, width=TICK_WIDTH,
                       length=TICK_LENGTH, pad=4)

    png, pdf = save_exact_size(fig, OUTPUT_BASENAME)
    plt.close(fig)

    print_size_report(FIG_HEIGHT_IN)
    print(f"Spread display upper limit: {spread_limit:.3f} days")
    print(f"Saved: {png.name}, {pdf.name}")


if __name__ == "__main__":
    main()
