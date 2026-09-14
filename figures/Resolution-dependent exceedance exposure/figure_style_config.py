
from __future__ import annotations

from pathlib import Path
import matplotlib

HERE = Path(__file__).resolve().parent
DATA_PATH = HERE / "dot_results.npz"

LATEX_TEXTWIDTH_PT = 390.0
TEX_POINTS_PER_INCH = 72.27
OVERLEAF_WIDTH_IN = LATEX_TEXTWIDTH_PT / TEX_POINTS_PER_INCH
TARGET_FINAL_FONT_PT = 12.0

SOURCE_BASE_FONT_PT = 24.0
FIG_WIDTH_IN = OVERLEAF_WIDTH_IN * SOURCE_BASE_FONT_PT / TARGET_FINAL_FONT_PT

FONT_FAMILY = "Times New Roman"
FONT_SERIF_FALLBACKS = [
    FONT_FAMILY,
    "Times",
    "Liberation Serif",
    "DejaVu Serif",
]

TITLE_PT = 24
AXIS_LABEL_PT = 24
TICK_PT = 22
COLORBAR_LABEL_PT = 24
COLORBAR_TICK_PT = 22
LEGEND_PT = 22

AXIS_LINEWIDTH = 1.25
TICK_WIDTH = 1.20
TICK_LENGTH = 5.0
GRID_LINEWIDTH = 0.55
GRID_ALPHA = 0.34
MAP_BORDER_LINEWIDTH = 1.25
BAR_EDGE_LINEWIDTH = 0.75
WHISKER_LINEWIDTH = 1.25

PNG_DPI = 300
PDF_FONT_TYPE = 42

DIVERGING_COLORS = [
    "#2166ac", "#67a9cf", "#d1e5f0", "#f7f7f7",
    "#fddbc7", "#ef8a62", "#b2182b",
]
SEQUENTIAL_BLUE = [
    "#f7fbff", "#deebf7", "#c6dbef", "#9ecae1",
    "#6baed6", "#4292c6", "#2171b5", "#084594",
]
SEQUENTIAL_WARM = ["#fff7ec", "#fee8c8", "#fdbb84", "#e34a33", "#7f0000"]
NEGATIVE_COLORS = ["#eff3ff", "#bdd7e7", "#6baed6", "#2171b5", "#08306b"]
POSITIVE_COLORS = ["#fee8c8", "#fdbb84", "#fc8d59", "#e34a33", "#990000"]
NEGATIVE_BACKGROUND = "#eef5ff"
POSITIVE_BACKGROUND = "#fff2e8"


def apply_style() -> None:
    matplotlib.rcParams.update({
        "font.family": "serif",
        "font.serif": FONT_SERIF_FALLBACKS,
        "font.size": TICK_PT,
        "axes.titlesize": TITLE_PT,
        "axes.labelsize": AXIS_LABEL_PT,
        "xtick.labelsize": TICK_PT,
        "ytick.labelsize": TICK_PT,
        "legend.fontsize": LEGEND_PT,
        "axes.linewidth": AXIS_LINEWIDTH,
        "xtick.major.width": TICK_WIDTH,
        "ytick.major.width": TICK_WIDTH,
        "xtick.major.size": TICK_LENGTH,
        "ytick.major.size": TICK_LENGTH,
        "figure.dpi": 120,
        "savefig.dpi": PNG_DPI,
        "pdf.fonttype": PDF_FONT_TYPE,
        "ps.fonttype": PDF_FONT_TYPE,
        "mathtext.fontset": "stix",
    })


def effective_final_font_pt(source_font_pt: float, figure_width_in: float = FIG_WIDTH_IN) -> float:
    return source_font_pt * OVERLEAF_WIDTH_IN / figure_width_in


def save_exact_size(fig, basename: str) -> tuple[Path, Path]:
    png = HERE / f"{basename}.png"
    pdf = HERE / f"{basename}.pdf"
    fig.savefig(png, dpi=PNG_DPI, facecolor="white", bbox_inches=None, pad_inches=0)
    fig.savefig(pdf, facecolor="white", bbox_inches=None, pad_inches=0)
    return png, pdf


def print_size_report(fig_height_in: float) -> None:
    print(f"Overleaf text width: {OVERLEAF_WIDTH_IN:.3f} in ({LATEX_TEXTWIDTH_PT:.0f} pt)")
    print(f"Source figure size: {FIG_WIDTH_IN:.3f} x {fig_height_in:.3f} in")
    print(f"24 pt source text -> {effective_final_font_pt(24):.2f} pt in manuscript")
    print(f"22 pt source text -> {effective_final_font_pt(22):.2f} pt in manuscript")
