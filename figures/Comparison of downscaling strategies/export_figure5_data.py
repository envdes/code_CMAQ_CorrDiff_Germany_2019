import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wasserstein_distance

SCRIPT_DIR = Path(__file__).parent.absolute()
GIT_DRAW_DIR = SCRIPT_DIR.parent.parent
STAT_PERF_DIR = GIT_DRAW_DIR / "Statistical_Performance"
OUTPUT_DIR = GIT_DRAW_DIR / "data"

OUTPUT_CURVES = OUTPUT_DIR / "figure5_kde_curves.csv"
OUTPUT_WASSERSTEIN = OUTPUT_DIR / "figure5_wasserstein.csv"
OUTPUT_METADATA = OUTPUT_DIR / "figure5_metadata.json"

MODEL_FILES = {
    'truth': 'pm25_truth.npz',
    'Bilinear': 'pm25_lr_interp.npz',
    'FLAML': 'pm25_flaml.npz',
    'RF': 'pm25_rf.npz',
    'Reg': 'pm25_reg.npz',
    'Diffusion': 'pm25_diffusion.npz',
    'CorrDiff': 'pm25_corrdiff.npz',
}

N_SUBSAMPLE = 3_000_000
RNG_SEED = 0
KDE_POINTS = 500
KDE_CHUNK = 25_000


def load_model_data(name, npz_file):
    path = STAT_PERF_DIR / npz_file
    if not path.exists():
        raise FileNotFoundError(f"Model data not found: {path}")

    with np.load(path) as f:
        data = f['data'].astype(np.float32)

    print(f"  Loaded {name}: {data.shape}, range [{data.min():.2f}, {data.max():.2f}]")
    return data


def kde_1d(samples, xgrid, chunk_size=KDE_CHUNK):
    x = np.asarray(samples, dtype=np.float64)
    x = x[np.isfinite(x)]

    n = len(x)
    if n < 2:
        return np.full_like(xgrid, np.nan, dtype=np.float64)

    std = max(np.std(x, ddof=1), 1e-6)
    h = max(1.06 * std * (n ** (-1 / 5)), 1e-3)

    accum = np.zeros_like(xgrid, dtype=np.float64)

    for start in range(0, n, chunk_size):
        block = x[start:start + chunk_size]
        u = (xgrid[:, None] - block[None, :]) / h
        accum += np.exp(-0.5 * u**2).sum(axis=1)

    return accum / (n * np.sqrt(2 * np.pi) * h)


def main():
    print("=" * 70)
    print("Figure 5 KDE Distribution Data Export")
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading standardized PM2.5 NPZ files...")
    models = {}
    for name, npz_file in MODEL_FILES.items():
        models[name] = load_model_data(name, npz_file)

    print()

    truth = models['truth']
    hr_shape = truth.shape

    for name, arr in models.items():
        if arr.shape != hr_shape:
            raise ValueError(f"Shape mismatch: truth {hr_shape}, {name} {arr.shape}")

    print(f"All arrays aligned: {hr_shape}")
    print()

    print("Flattening all arrays...")
    hr_flat = truth.ravel()
    model_flats = {name: models[name].ravel() for name in MODEL_FILES.keys() if name != 'truth'}

    n_total = len(hr_flat)
    n_sample = min(N_SUBSAMPLE, n_total)

    print(f"  Total pixels: {n_total:,}")
    print(f"  Subsample size: {n_sample:,}")
    print()

    print(f"Generating random subsample (seed={RNG_SEED})...")
    rng = np.random.default_rng(RNG_SEED)
    idx = rng.choice(n_total, size=n_sample, replace=False)

    hr_sub = hr_flat[idx]
    model_sub = {name: model_flats[name][idx] for name in model_flats.keys()}

    print(f"  HR: {hr_sub.shape}, range [{hr_sub.min():.2f}, {hr_sub.max():.2f}]")
    for name in ['Bilinear', 'FLAML', 'RF', 'Reg', 'Diffusion', 'CorrDiff']:
        arr = model_sub[name]
        print(f"  {name}: {arr.shape}, range [{arr.min():.2f}, {arr.max():.2f}]")
    print()

    print("Computing Wasserstein distances...")
    w_dist = {}
    for name in ['Bilinear', 'FLAML', 'RF', 'Reg', 'Diffusion', 'CorrDiff']:
        w = wasserstein_distance(model_sub[name], hr_sub)
        w_dist[name] = w
        print(f"  {name:12s}: W = {w:.6f}")
    print()

    print("Constructing KDE x-grid...")
    sample_groups = [hr_sub] + [model_sub[name] for name in ['Bilinear', 'FLAML', 'RF', 'Reg', 'Diffusion', 'CorrDiff']]
    all_vals = np.concatenate(sample_groups)

    xmin = max(0.0, np.percentile(all_vals, 0.1))
    xmax = np.percentile(all_vals, 99.9)
    xgrid = np.linspace(xmin, xmax, KDE_POINTS)

    print(f"  xmin (0.1 percentile): {xmin:.6f}")
    print(f"  xmax (99.9 percentile): {xmax:.6f}")
    print(f"  KDE points: {KDE_POINTS}")
    print()

    print("Computing KDE curves (Silverman bandwidth)...")
    kde_hr = kde_1d(hr_sub, xgrid)
    print(f"  HR: computed")

    kde_models = {}
    for name in ['Bilinear', 'FLAML', 'RF', 'Reg', 'Diffusion', 'CorrDiff']:
        kde_models[name] = kde_1d(model_sub[name], xgrid)
        print(f"  {name}: computed")
    print()

    print("Saving KDE curves to CSV...")
    curves_df = pd.DataFrame({
        'pm25': xgrid,
        'HR': kde_hr,
        'CorrDiff': kde_models['CorrDiff'],
        'Diffusion': kde_models['Diffusion'],
        'Reg': kde_models['Reg'],
        'LR-interp': kde_models['Bilinear'],
        'FLAML': kde_models['FLAML'],
        'RF': kde_models['RF'],
    })

    curves_df.to_csv(OUTPUT_CURVES, index=False, float_format='%.12e')
    print(f"✓ Saved: {OUTPUT_CURVES}")
    print(f"  Shape: {curves_df.shape}")
    print(f"  Size: {OUTPUT_CURVES.stat().st_size / 1024:.2f} KB")
    print()

    print("Saving Wasserstein distances to CSV...")
    wasserstein_df = pd.DataFrame({
        'model': ['CorrDiff', 'Diffusion', 'Reg', 'LR-interp', 'FLAML', 'RF'],
        'wasserstein': [
            w_dist['CorrDiff'],
            w_dist['Diffusion'],
            w_dist['Reg'],
            w_dist['Bilinear'],
            w_dist['FLAML'],
            w_dist['RF'],
        ]
    })

    wasserstein_df.to_csv(OUTPUT_WASSERSTEIN, index=False, float_format='%.6f')
    print(f"✓ Saved: {OUTPUT_WASSERSTEIN}")
    print(f"  Shape: {wasserstein_df.shape}")
    print()

    print("Wasserstein distances:")
    for _, row in wasserstein_df.iterrows():
        print(f"  {row['model']:12s}: {row['wasserstein']:.6f}")
    print()

    print("Saving metadata...")
    import json
    metadata = {
        'sample_size': int(n_sample),
        'rng_seed': int(RNG_SEED),
        'kde_points': int(KDE_POINTS),
        'percentile_min': float(xmin),
        'percentile_max': float(xmax),
        'kde_method': 'Gaussian kernel with Silverman bandwidth',
        'bandwidth_formula': '1.06 * std * n^(-1/5)',
        'source_data': 'Standardized pm25_*.npz files',
        'total_pixels': int(n_total),
        'data_shape': list(hr_shape),
    }

    with open(OUTPUT_METADATA, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✓ Saved: {OUTPUT_METADATA}")
    print()

    print("=" * 70)
    print("✓ Export completed successfully!")
    print("=" * 70)
    print()
    print("Summary:")
    print(f"  - Sample size: {n_sample:,} pixels (from {n_total:,} total)")
    print(f"  - RNG seed: {RNG_SEED}")
    print(f"  - KDE points: {KDE_POINTS}")
    print(f"  - Output files:")
    print(f"    • {OUTPUT_CURVES.name} ({OUTPUT_CURVES.stat().st_size / 1024:.2f} KB)")
    print(f"    • {OUTPUT_WASSERSTEIN.name} ({OUTPUT_WASSERSTEIN.stat().st_size} bytes)")
    print(f"    • {OUTPUT_METADATA.name} ({OUTPUT_METADATA.stat().st_size} bytes)")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
