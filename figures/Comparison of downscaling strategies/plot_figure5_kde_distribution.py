import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

HERE = Path(__file__).parent.absolute()
DATA_DIR = HERE.parent / 'data'
CURVES_FILE = DATA_DIR / 'figure5_kde_curves.csv'
WASSERSTEIN_FILE = DATA_DIR / 'figure5_wasserstein.csv'
OUTDIR = str(HERE)

OUTPUT_BASE = f"{OUTDIR}/figure_kde_distribution"

FIG_WIDTH_IN = 4.65
FIG_HEIGHT_IN = 3.35
FONT_SIZE = 10.0
LEGEND_FS = 9.0

UNIT = r"PM$_{2.5}$ ($\mu$g m$^{-3}$)"

plt.rcParams.update({
    "font.family": "STIXGeneral",
    "mathtext.fontset": "stix",
    "font.size": FONT_SIZE,
    "axes.labelsize": FONT_SIZE,
    "xtick.labelsize": FONT_SIZE,
    "ytick.labelsize": FONT_SIZE,
    "legend.fontsize": FONT_SIZE,
    "axes.linewidth": 0.8,
    "lines.linewidth": 1.35,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

C_HR = "#000000"

COLORS = {
    "Bilinear":  "#E69F00",
    "FLAML":     "#CC79A7",
    "RF":        "#882255",
    "Reg":       "#009E73",
    "Diffusion": "#999999",
    "CorrDiff":  "#0072B2",
}

STYLES = {
    "Bilinear": ":",
    "FLAML": "--",
    "RF": "-.",
    "Reg": "--",
    "Diffusion": "-",
    "CorrDiff": "-",
}

DISPLAY_NAMES = {
    "Bilinear": "LR-interp",
    "FLAML": "FLAML",
    "RF": "RF",
    "Reg": "Reg",
    "Diffusion": "Diffusion",
    "CorrDiff": "CorrDiff",
}

PLOT_ORDER = ["Bilinear", "FLAML", "RF", "Reg", "Diffusion", "CorrDiff"]

EPS = 1e-12

print("=" * 70)
print("Figure 5: KDE Distribution (Publication Layout)")
print("=" * 70)
print()

if not CURVES_FILE.exists():
    raise FileNotFoundError(
        f"KDE curves not found: {CURVES_FILE}\n"
        f"Please run: .claude/scripts/export_figure5_data.py"
    )

if not WASSERSTEIN_FILE.exists():
    raise FileNotFoundError(
        f"Wasserstein distances not found: {WASSERSTEIN_FILE}\n"
        f"Please run: .claude/scripts/export_figure5_data.py"
    )

print(f"Loading precomputed KDE curves from: {CURVES_FILE.name}")
curves = pd.read_csv(CURVES_FILE)
print(f"  Shape: {curves.shape}")
print(f"  Columns: {list(curves.columns)}")
print()

print(f"Loading Wasserstein distances from: {WASSERSTEIN_FILE.name}")
wasserstein = pd.read_csv(WASSERSTEIN_FILE)
w_dist = dict(zip(wasserstein['model'], wasserstein['wasserstein']))

print("  Wasserstein distances:")
for model, w in w_dist.items():
    print(f"    {model:12s}: {w:.4f}")
print()

print("Generating figure...")

xgrid = curves['pm25'].values
kde_hr = curves['HR'].values

kde_models = {
    'Bilinear': curves['LR-interp'].values,
    'FLAML': curves['FLAML'].values,
    'RF': curves['RF'].values,
    'Reg': curves['Reg'].values,
    'Diffusion': curves['Diffusion'].values,
    'CorrDiff': curves['CorrDiff'].values,
}

fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))

ax = fig.add_axes([0.16, 0.36, 0.80, 0.60])
leg_ax = fig.add_axes([0.04, 0.025, 0.92, 0.24])
leg_ax.axis("off")

hr_line, = ax.plot(
    xgrid,
    kde_hr + EPS,
    color=C_HR,
    linewidth=1.45,
    linestyle="-",
    label="HR (4 km)",
    zorder=7,
)

line_handles = {"HR": hr_line}

for name in PLOT_ORDER:
    lw = 1.60 if name == "CorrDiff" else 1.05
    z = 8 if name == "CorrDiff" else 5

    line, = ax.plot(
        xgrid,
        kde_models[name] + EPS,
        color=COLORS[name],
        linewidth=lw,
        linestyle=STYLES[name],
        label=DISPLAY_NAMES[name],
        zorder=z,
    )
    line_handles[name] = line

ax.set_yscale("log")
ax.set_xlabel(UNIT, labelpad=2)
ax.set_ylabel("Probability density", labelpad=2)

ax.set_ylim(1e-12, 2e-1)
ax.set_yticks([1e-1, 1e-3, 1e-5, 1e-7, 1e-9, 1e-11])

ax.grid(
    True,
    which="major",
    linestyle="--",
    linewidth=0.55,
    alpha=0.25,
    color="gray",
)
ax.grid(False, which="minor")

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.tick_params(axis="both", which="major", labelsize=FONT_SIZE, pad=1.5)


def legend_label(key):
    if key == "HR":
        return "HR (4 km)"
    w_value = w_dist.get(DISPLAY_NAMES.get(key, key), w_dist.get(key, 0.0))
    return f"{DISPLAY_NAMES[key]} ({w_value:.2f})"


row1 = ["HR", "CorrDiff", "Diffusion"]
row2 = ["Reg", "Bilinear", "FLAML", "RF"]

legend_common = {
    "frameon": False,
    "fontsize": LEGEND_FS,
    "handlelength": 1.7,
    "handletextpad": 0.45,
    "columnspacing": 1.30,
    "borderaxespad": 0.0,
}

legend_top = leg_ax.legend(
    [line_handles[key] for key in row1],
    [legend_label(key) for key in row1],
    loc="center",
    bbox_to_anchor=(0.50, 0.70),
    ncol=3,
    **legend_common,
)
leg_ax.add_artist(legend_top)

leg_ax.legend(
    [line_handles[key] for key in row2],
    [legend_label(key) for key in row2],
    loc="center",
    bbox_to_anchor=(0.50, 0.20),
    ncol=4,
    **legend_common,
)

pdf_path = f"{OUTPUT_BASE}.pdf"
png_path = f"{OUTPUT_BASE}.png"

fig.savefig(
    pdf_path,
    dpi=300,
    bbox_inches=None,
    pad_inches=0,
)
fig.savefig(
    png_path,
    dpi=300,
    bbox_inches=None,
    pad_inches=0,
)

plt.close(fig)

print(f"✓ Saved: {pdf_path}")
print(f"✓ Saved: {png_path}")
print()
print("=" * 70)
print("✓ Figure 5 generated successfully!")
print("=" * 70)
print()
print("LaTeX: insert the PDF at natural size:")
print(r"  \includegraphics{figure_kde_distribution.pdf}")
print("Do not specify width=... or scale=...")
