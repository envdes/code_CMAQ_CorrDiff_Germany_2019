import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import os
from pathlib import Path

HERE = Path(__file__).parent.absolute()
DATA_FILE = HERE.parent / 'data' / 'figure2_monthly_metrics.csv'

FIG_WIDTH, FIG_HEIGHT = 5.40, 3.50
FONT_SIZE, FONT_FAMILY = 11, 'STIXGeneral'

MAP = [
    ('LR-interp', 'Bilinear', '#E69F00'),
    ('FLAML/XGBoost', 'FLAML', '#CC79A7'),
    ('RF', 'RF (md12)', '#882255'),
    ('Reg', 'CorrDiff-v1 reg', '#009E73'),
    ('Diffusion', 'Diffusion diff (D)', '#999999'),
    ('CorrDiff', 'CorrDiff-v1 diff', '#0072B2'),
]

COL = {m[0]: m[2] for m in MAP}
SHORT = {'FLAML/XGBoost': 'FLAML'}
MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_FILE}\n"
        f"Please run: .claude/scripts/export_figure2_data.py"
    )

m = pd.read_csv(DATA_FILE)

def series(metric_name, model_name):
    model_data = m[m['model'] == model_name].set_index('month')
    return [float(model_data.loc[mo, metric_name]) for mo in MON]

plt.rcParams.update({
    'font.family': FONT_FAMILY, 'mathtext.fontset': 'stix', 'font.size': FONT_SIZE,
    'axes.labelsize': FONT_SIZE, 'axes.titlesize': FONT_SIZE, 'xtick.labelsize': FONT_SIZE,
    'ytick.labelsize': FONT_SIZE, 'legend.fontsize': 9, 'axes.linewidth': 0.8,
    'pdf.fonttype': 42, 'ps.fonttype': 42,
})

def _clean(ax):
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.grid(True, color='#E6E6E6', linestyle='-', linewidth=0.6)
    ax.set_axisbelow(True)

fig, axes = plt.subplots(2, 1, figsize=(FIG_WIDTH, FIG_HEIGHT), sharex=True)
handles, labels = [], []

for internal_name, display_name, _ in MAP:
    lw = 1.3 if internal_name == 'CorrDiff' else 1.0
    ms = 3.5 if internal_name == 'CorrDiff' else 3.0
    z = 5 if internal_name == 'CorrDiff' else 3

    for ax, metric in zip(axes, ('r2', 'rmse')):
        ln, = ax.plot(MON, series(metric, internal_name), marker='o', markersize=ms,
                     linewidth=lw, color=COL[internal_name], label=SHORT.get(internal_name, display_name),
                     zorder=z, clip_on=False)
    handles.append(ln)
    labels.append(SHORT.get(internal_name, display_name))

axes[0].axhline(0, color='0.25', linestyle='--', linewidth=1.0, zorder=1)
axes[1].set_xlabel('Month (2019)', labelpad=2)

for ax in axes:
    _clean(ax)
    ax.margins(x=0.03, y=0.08)
    ax.yaxis.set_major_locator(MaxNLocator(nbins=4))

axes[0].text(0.0, 1.02, r'(a) $R^2$', transform=axes[0].transAxes, ha='left', va='bottom')
axes[1].text(0.0, 1.02, r'(b) RMSE ($\mu$g m$^{-3}$)', transform=axes[1].transAxes, ha='left', va='bottom')

fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.50, -0.01), ncol=6,
          frameon=False, handlelength=1.35, columnspacing=0.65, handletextpad=0.35)

fig.subplots_adjust(left=0.103, right=0.897, top=0.90, bottom=0.20, hspace=0.38)

fig.savefig(f'{HERE}/figure_monthly_performance.pdf', dpi=300, bbox_inches=None, pad_inches=0)
fig.savefig(f'{HERE}/figure_monthly_performance.png', dpi=300, bbox_inches=None, pad_inches=0)
plt.close(fig)
print(f'Generated: figure_monthly_performance.png/pdf')
