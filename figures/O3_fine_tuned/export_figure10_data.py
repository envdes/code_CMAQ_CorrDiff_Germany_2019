import os
import sys
import glob
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
GIT_DRAW_DIR = SCRIPT_DIR.parent.parent
O3_DIR = GIT_DRAW_DIR / "O3FINETUNE"
OUTPUT_DIR = GIT_DRAW_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "figure10_monthly_metrics.csv"

DIR_O3 = "/home/zhoujunan/results/corrdiff_O3_2019daily/inference_npz/arrays"
DIR_PM = "/home/zhoujunan/corrdiff3.0/corrdiff_O3/O3_results_frozenenc_diff_daily"
LR_INTERP_PATH = "/home/zhoujunan/results/eval_results/lrinterp_cmaq36km_daily_o3.npz"

VMIN, VMAX = 1.49e-7, 161.81
SCALE = VMAX - VMIN

DAILY_DAYS = [30, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 30]
MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

MODEL_MAP = {
    'LR-interp': 'LR-interp',
    'O3 Reg': 'O3 CorrDiff-reg',
    'O3 CorrDiff': 'O3 CorrDiff',
    'Fine-tuned Reg': 'PM2O3 CorrDiff-reg',
    'Fine-tuned CorrDiff': 'PM2O3 CorrDiff',
}


def month_index():
    doff = np.cumsum([0] + DAILY_DAYS[:-1])
    idx = np.zeros(sum(DAILY_DAYS), dtype=int)
    for m in range(12):
        idx[doff[m]:doff[m] + DAILY_DAYS[m]] = m
    return idx


def denorm_lr(x):
    return (x + 1.0) / 2.0 * SCALE + VMIN


def auto_denorm(x):
    x = np.asarray(x, dtype=np.float64)
    if x.max() <= 1.5 and x.min() >= -1.5:
        return denorm_lr(x)
    return x


def monthly_rmse_r2(pred, truth, midx):
    rmse, r2 = [], []
    for m in range(12):
        mask = midx == m
        pm, tm = pred[mask], truth[mask]
        T_m = pm.shape[0]
        rmse_t = np.zeros(T_m)
        r2_t = np.zeros(T_m)
        for t in range(T_m):
            p, tr = pm[t].ravel(), tm[t].ravel()
            err = p - tr
            rmse_t[t] = np.sqrt(np.mean(err**2))
            ss_res = np.sum(err**2)
            ss_tot = np.sum((tr - tr.mean())**2)
            r2_t[t] = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0
        rmse.append(float(np.mean(rmse_t)))
        r2.append(float(np.mean(r2_t)))
    return np.array(rmse), np.array(r2)


def main():
    print("=" * 70)
    print("Figure 10 O3 Monthly Temporal Performance Data Export")
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Checking data paths...")
    if not Path(DIR_O3).exists():
        print(f"ERROR: O3 retrain directory not found: {DIR_O3}")
        print("Please update DIR_O3 in this script to point to actual data location.")
        return 1

    if not Path(DIR_PM).exists():
        print(f"ERROR: PM2O3 finetune directory not found: {DIR_PM}")
        print("Please update DIR_PM in this script to point to actual data location.")
        return 1

    if not Path(LR_INTERP_PATH).exists():
        print(f"ERROR: LR-interp file not found: {LR_INTERP_PATH}")
        print("Please update LR_INTERP_PATH in this script to point to actual data location.")
        return 1

    print()

    print("Loading data...")
    o3_files = sorted(glob.glob(str(Path(DIR_O3) / 'day_*.npz')))
    pm_final_files = sorted(glob.glob(str(Path(DIR_PM) / 'final_*.npy')))
    pm_reg_files = sorted(glob.glob(str(Path(DIR_PM) / 'reg_*.npy')))
    n_days = len(o3_files)

    print(f"  O3 retrain files: {n_days}")
    print(f"  PM2O3 final files: {len(pm_final_files)}")
    print(f"  PM2O3 reg files: {len(pm_reg_files)}")

    if n_days == 0:
        print("ERROR: No O3 daily files found.")
        return 1

    lr_interp_o3 = np.load(LR_INTERP_PATH)['pred_maps'].astype(np.float64)
    print(f"  LR-interp: {lr_interp_o3.shape}")
    print()

    print("Loading daily data...")
    truth_all, lr_all, o3r_all, o3f_all, pmr_all, pmf_all = [], [], [], [], [], []

    for day_idx in range(n_days):
        if day_idx % 50 == 0:
            print(f"  Processing day {day_idx}/{n_days}...")

        data_o3 = np.load(o3_files[day_idx])

        hr = data_o3['hr'].astype(np.float64).squeeze(axis=0)
        hr = auto_denorm(hr)
        truth_all.append(hr)

        lr_all.append(lr_interp_o3[day_idx])

        reg_o3 = data_o3['reg'].astype(np.float64).squeeze(axis=0)
        reg_o3 = auto_denorm(reg_o3)
        o3r_all.append(reg_o3)

        final_o3 = data_o3['final'].astype(np.float64)
        if final_o3.ndim == 4:
            final_o3 = final_o3.squeeze(axis=0).mean(axis=0)
        elif final_o3.ndim == 3:
            final_o3 = final_o3.mean(axis=0)
        final_o3 = auto_denorm(final_o3)
        o3f_all.append(final_o3)

        reg_pm = np.load(pm_reg_files[day_idx]).astype(np.float64).squeeze()
        reg_pm = auto_denorm(reg_pm)
        pmr_all.append(reg_pm)

        final_pm = np.load(pm_final_files[day_idx]).astype(np.float64).squeeze()
        if final_pm.ndim == 3:
            final_pm = final_pm.mean(axis=0)
        final_pm = auto_denorm(final_pm)
        pmf_all.append(final_pm)

    truth_all = np.stack(truth_all)
    lr_all = np.stack(lr_all)
    o3r_all = np.stack(o3r_all)
    o3f_all = np.stack(o3f_all)
    pmr_all = np.stack(pmr_all)
    pmf_all = np.stack(pmf_all)

    print(f"\n  Truth: {truth_all.shape}")
    print(f"  LR-interp: {lr_all.shape}")
    print(f"  O3 Reg: {o3r_all.shape}")
    print(f"  O3 CorrDiff: {o3f_all.shape}")
    print(f"  Fine-tuned Reg: {pmr_all.shape}")
    print(f"  Fine-tuned CorrDiff: {pmf_all.shape}")
    print()

    models = {
        'LR-interp': lr_all,
        'O3 CorrDiff-reg': o3r_all,
        'O3 CorrDiff': o3f_all,
        'PM2O3 CorrDiff-reg': pmr_all,
        'PM2O3 CorrDiff': pmf_all,
    }

    midx = month_index()
    print(f"Month index: {len(midx)} days → 12 months")
    print()

    rows = []

    for display_name, internal_key in MODEL_MAP.items():
        print(f"  Processing {display_name}...", end=" ", flush=True)

        pred = models[internal_key]
        rmse, r2 = monthly_rmse_r2(pred, truth_all, midx)

        for m in range(12):
            rows.append({
                'month': MONTHS[m],
                'model': display_name,
                'rmse': round(float(rmse[m]), 6),
                'r2': round(float(r2[m]), 6)
            })

        print(f"✓ (RMSE range: [{rmse.min():.2f}, {rmse.max():.2f}])")

    print()

    df = pd.DataFrame(rows)

    print("Verifying data...")
    n_models = len(MODEL_MAP)
    n_months = 12
    expected_rows = n_models * n_months
    actual_rows = len(df)

    print(f"  Expected rows: {expected_rows} ({n_models} models × {n_months} months)")
    print(f"  Actual rows: {actual_rows}")

    if actual_rows != expected_rows:
        print(f"  WARNING: Row count mismatch!")

    if df.isnull().any().any():
        print(f"  WARNING: CSV contains missing values!")

    dup_count = df.duplicated(subset=['month', 'model']).sum()
    if dup_count > 0:
        print(f"  WARNING: {dup_count} duplicate combinations found!")

    print()

    df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved: {OUTPUT_FILE}")
    print(f"  Columns: {list(df.columns)}")
    print(f"  Total rows: {len(df)}")
    print()

    print("Sample data (first 6 rows):")
    print(df.head(6).to_string(index=False))
    print()

    print("Summary statistics by model:")
    summary = df.groupby('model')[['rmse', 'r2']].agg(['min', 'max', 'mean'])
    print(summary.round(4))
    print()

    print("=" * 70)
    print("✓ Export completed successfully!")
    print("=" * 70)
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
