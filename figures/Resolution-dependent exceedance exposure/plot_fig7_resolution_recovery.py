
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
import numpy as np

from figure_style_config import (
    FIG_WIDTH_IN as DEFAULT_FIG_WIDTH_IN,
    DIVERGING_COLORS,
    apply_style, save_exact_size,
)
from exposure_helpers import load_data, configure_map_axis

FONT_FAMILY = "Times New Roman"
FONT_SERIF_FALLBACKS = [
    FONT_FAMILY,
    "Times",
    "Liberation Serif",
    "DejaVu Serif",
]

FIG_WIDTH_IN = DEFAULT_FIG_WIDTH_IN
FIG_HEIGHT_IN = 4.35

PANEL_TITLE_PT = 21
AXIS_LABEL_PT_LOCAL = 21
TICK_PT_LOCAL = 21
COLORBAR_LABEL_PT_LOCAL = 21
COLORBAR_TICK_PT_LOCAL = 21

AXIS_LINEWIDTH = 1.25
TICK_WIDTH = 1.20
TICK_LENGTH = 5.0
GRID_LINEWIDTH = 0.55
GRID_ALPHA = 0.34
MAP_BORDER_LINEWIDTH = 1.25

LEFT = 0.145
RIGHT = 0.895
BOTTOM = 0.125
TOP = 0.895
WSPACE = 0.065

COLORBAR_GAP = 0.01
COLORBAR_WIDTH = 0.015
COLORBAR_VERTICAL_INSET = 0.000
TITLE_PAD = 7
COLORBAR_LABEL_PAD = -0.2
MAP_LABEL_PAD = 5
Y_LABEL_PAD = 8
MAP_TICK_PAD = 4
ROBUST_PERCENTILE = 99.5
OUTPUT_BASENAME = "fig7_resolution_recovery_modular_local_style_v5_gap"


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
        "axes.linewidth": AXIS_LINEWIDTH,
        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def print_local_size_report() -> None:
    print(f"Source figure size: {FIG_WIDTH_IN:.3f} x {FIG_HEIGHT_IN:.3f} in")
    print(f"Panel title: {PANEL_TITLE_PT} pt")
    print(f"Map labels/colorbar label: {AXIS_LABEL_PT_LOCAL} / {COLORBAR_LABEL_PT_LOCAL} pt")
    print(f"Map ticks/colorbar ticks: {TICK_PT_LOCAL} / {COLORBAR_TICK_PT_LOCAL} pt")


def style_map_axis(ax, *, show_ylabel: bool) -> None:
    ax.tick_params(
        axis="both",
        labelsize=TICK_PT_LOCAL,
        width=TICK_WIDTH,
        length=TICK_LENGTH,
        pad=MAP_TICK_PAD,
    )
    ax.set_xlabel("Longitude", fontsize=AXIS_LABEL_PT_LOCAL, labelpad=MAP_LABEL_PAD)
    if show_ylabel:
        ax.set_ylabel("Latitude", fontsize=AXIS_LABEL_PT_LOCAL, labelpad=MAP_LABEL_PAD)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA, zorder=2)
    for spine in ax.spines.values():
        spine.set_linewidth(MAP_BORDER_LINEWIDTH)


def main() -> None:
    apply_local_style()
    data = load_data()

    mask = data["mask"].astype(bool)
    hr = data["delta_hr_lr"].astype(float)
    cd = data["delta_cd_mean"].astype(float)
    lats = data["lats"]
    lons = data["lons"]

    fields = [np.where(mask, hr, np.nan), np.where(mask, cd, np.nan)]
    joint_absolute = np.concatenate([np.abs(hr[mask]), np.abs(cd[mask])])
    limit = float(np.percentile(joint_absolute, ROBUST_PERCENTILE))
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    cmap = LinearSegmentedColormap.from_list(
        "dot25_resolution_difference", DIVERGING_COLORS, N=256,
    )

    fig, axes = plt.subplots(1, 2, figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), sharey=True)
    fig.subplots_adjust(left=LEFT, right=RIGHT, bottom=BOTTOM, top=TOP, wspace=WSPACE)

    titles = ["(a) HR (4 km) - CMAQ (36 km)", "(b) CorrDiff - CMAQ (36 km)"]
    mesh = None
    for i, (ax, field, title) in enumerate(zip(axes, fields, titles)):
        configure_map_axis(ax, lons, lats, mask, show_ylabel=(i == 0))
        style_map_axis(ax, show_ylabel=(i == 0))
        ax.set_title(title, fontsize=PANEL_TITLE_PT, pad=TITLE_PAD)
        mesh = ax.pcolormesh(
            lons, lats, field, cmap=cmap, norm=norm,
            shading="auto", edgecolors="face", linewidth=0.10,
            rasterized=True, zorder=1,
        )

    axes[1].tick_params(axis="y", which="both", left=False, labelleft=False)
    axes[1].set_ylabel("")

    fig.canvas.draw()
    right_box = axes[1].get_position()
    cax = fig.add_axes([
        right_box.x1 + COLORBAR_GAP,
        right_box.y0 + COLORBAR_VERTICAL_INSET,
        COLORBAR_WIDTH,
        right_box.height - 2 * COLORBAR_VERTICAL_INSET,
    ])
    cb = fig.colorbar(mesh, cax=cax, orientation="vertical", extend="both")
    cb.set_label(r"$\Delta\mathrm{DOT}_{25}$ (days)",
                 fontsize=COLORBAR_LABEL_PT_LOCAL, labelpad=COLORBAR_LABEL_PAD)
    cb.ax.tick_params(labelsize=COLORBAR_TICK_PT_LOCAL, width=TICK_WIDTH,
                      length=TICK_LENGTH, pad=4)

    png, pdf = save_exact_size(fig, OUTPUT_BASENAME)
    plt.close(fig)

    print_local_size_report()
    print(f"Shared symmetric range: {-limit:.3f} to {limit:.3f} days")
    print(f"Saved: {png.name}, {pdf.name}")


if __name__ == "__main__":
    main()
