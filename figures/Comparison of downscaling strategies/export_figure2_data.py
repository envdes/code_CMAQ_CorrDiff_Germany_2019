import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
GIT_DRAW_DIR = SCRIPT_DIR.parent.parent
STAT_PERF_DIR = GIT_DRAW_DIR / "Statistical_Performance"

OUTPUT_DIR = GIT_DRAW_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "figure2_monthly_metrics.csv"

MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
          "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
MONTH_DAYS = [30, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 28]

MODEL_MAP = {
    'LR-interp': 'pm25_lr_interp.npz',
    'FLAML/XGBoost': 'pm25_flaml.npz',
    'RF': 'pm25_rf.npz',
    'Reg': 'pm25_reg.npz',
    'Diffusion': 'pm25_diffusion.npz',
    'CorrDiff': 'pm25_corrdiff.npz',
}


def month_index():
    idx = np.zeros(sum(MONTH_DAYS), dtype=int)
    s = 0
    for m, n in enumerate(MONTH_DAYS):
        idx[s:s + n] = m
        s += n
    return idx


def per_timestep(pred, truth, midx):
    rmse, r2 = [], []
    for m in range(12):
        p = pred[midx == m].astype(np.float64)
        t = truth[midx == m].astype(np.float64)
        rs, r2s = [], []
        for i in range(len(p)):
            pi, ti = p[i].ravel(), t[i].ravel()
            e = pi - ti
            rs.append(np.sqrt(np.mean(e**2)))
            ss_res = np.sum(e**2)
            ss_tot = np.sum((ti - ti.mean())**2)
            r2s.append(1 - ss_res / ss_tot if ss_tot > 0 else 0.0)
        rmse.append(float(np.mean(rs)))
        r2.append(float(np.mean(r2s)))
    return np.array(rmse), np.array(r2)


def main():
    print("=" * 70)
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ground truth...")
    truth_file = STAT_PERF_DIR / "pm25_truth.npz"
    if not truth_file.exists():
        print(f"Error: Truth file not found: {truth_file}")
        return 1

    with np.load(truth_file) as f:
        truth = f['data']

    print(f"  Truth shape: {truth.shape}")
    print(f"  Truth range: [{truth.min():.2f}, {truth.max():.2f}] μg/m³")
    print()

    midx = month_index()
    print(f"Month index: {len(midx)} days → 12 months")
    print()

    rows = []

    for model_name, npz_file in MODEL_MAP.items():
        print(f"  Processing {model_name}...", end=" ", flush=True)

        pred_file = STAT_PERF_DIR / npz_file
        if not pred_file.exists():
            print(f"WARNING: File not found: {pred_file}")
            continue

        with np.load(pred_file) as f:
            pred = f['data']

        rmse, r2 = per_timestep(pred, truth, midx)

        for m in range(12):
            rows.append({
                'month': MONTHS[m],
                'model': model_name,
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
