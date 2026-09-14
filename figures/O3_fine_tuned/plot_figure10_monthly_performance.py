import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from pathlib import Path

HERE = Path(__file__).parent.absolute()
DATA_FILE = HERE.parent / 'data' / 'figure10_monthly_metrics.csv'

MON = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

MCOL = {
    'LR-interp':             '#9aa0a6',
    'O3 Reg':                '#7fb0e0',
    'O3 CorrDiff':           '#1f5fa8',
    'Fine-tuned Reg':        '#f3c08a',
    'Fine-tuned CorrDiff':   '#d2521f',
}

MODEL_ORDER = ['LR-interp', 'O3 Reg', 'O3 CorrDiff', 'Fine-tuned Reg', 'Fine-tuned CorrDiff']

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 28,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'figure.dpi': 150,
    'savefig.dpi': 200,
    'savefig.bbox': 'tight',
})

print("=" * 70)
print("Figure 10: O3 Monthly Temporal Performance")
print("=" * 70)
print()

if not DATA_FILE.exists():
    raise FileNotFoundError(
        f"Data file not found: {DATA_FILE}\n"
        f"Please run: .claude/scripts/export_figure10_data.py"
    )

print(f"Loading data from: {DATA_FILE.name}")
df = pd.read_csv(DATA_FILE)
print(f"  Shape: {df.shape}")
print(f"  Models: {df['model'].unique().tolist()}")
print()

rmse_dict = {}
r2_dict = {}

for model in MODEL_ORDER:
    model_data = df[df['model'] == model].set_index('month')
    rmse_dict[model] = [float(model_data.loc[mo, 'rmse']) for mo in MON]
    r2_dict[model] = [float(model_data.loc[mo, 'r2']) for mo in MON]

print("Generating figure...")

fig, axes = plt.subplots(2, 1, figsize=(18, 14), sharex=True)
r2_ylim = (0.15, 1.0)

for name in MODEL_ORDER:
    if name not in r2_dict:
        continue
    ls = '--' if 'Reg' in name else '-'
    lw = 2.5 if 'Reg' in name else 2.8
    axes[0].plot(MON, r2_dict[name], marker='o', ms=10, lw=lw, ls=ls,
                 color=MCOL[name], label=name, zorder=2)

axes[0].axhline(0, color='#333', ls='--', lw=1.5)
axes[0].set_title('R²', loc='center', fontweight='normal', fontsize=28, pad=10)
axes[0].set_ylim(r2_ylim)
axes[0].grid(color='#EEE', lw=0.8)
axes[0].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
axes[0].tick_params(labelsize=24)

for name in MODEL_ORDER:
    if name not in rmse_dict:
        continue
    ls = '--' if 'Reg' in name else '-'
    lw = 2.5 if 'Reg' in name else 2.8
    axes[1].plot(MON, rmse_dict[name], marker='o', ms=10, lw=lw, ls=ls,
                 color=MCOL[name], label=name, zorder=2)

axes[1].set_title('RMSE (μg · m⁻³)', loc='center', fontweight='normal', fontsize=28, pad=10)
axes[1].grid(color='#EEE', lw=0.8)
axes[1].yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(5))
axes[1].tick_params(labelsize=24)

xticks = np.arange(len(MON))
axes[1].set_xticks(xticks)
axes[1].set_xticklabels(MON, fontsize=22)
axes[1].set_xlabel("O$_3$ month (2019)", fontsize=28)

handles, labels = axes[1].get_legend_handles_labels()
fig.legend(handles, labels, frameon=False, ncol=3,
           loc='lower center', bbox_to_anchor=(0.5, -0.03), fontsize=24)
fig.tight_layout(rect=[0, 0.04, 1, 1])

for a in axes:
    a.set_axisbelow(True)

png_path = str(HERE / 'figure10_monthly_temporal_performance.png')
pdf_path = str(HERE / 'figure10_monthly_temporal_performance.pdf')

fig.savefig(png_path)
fig.savefig(pdf_path)
plt.close(fig)

print(f"✓ Saved: {png_path}")
print(f"✓ Saved: {pdf_path}")
print()
print("=" * 70)
print("✓ Figure 10 completed successfully!")
print("=" * 70)
