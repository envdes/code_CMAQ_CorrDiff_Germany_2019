#!/usr/bin/env python3
"""Compute canonical O3 downscaling metrics from saved predictions."""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

H_O3_VMIN = 4.586369261802559e-13
H_O3_VMAX = 121.23634338378906


def denorm(values, vmin=H_O3_VMIN, vmax=H_O3_VMAX):
    return (values + 1.0) / 2.0 * (vmax - vmin) + vmin


def is_norm_space(values):
    return values.min() >= -1.5 and values.max() <= 1.5


def load_o3_results(directory):
    files = sorted(Path(directory).glob("day_*.npz"))
    if not files:
        raise FileNotFoundError(f"No day_*.npz files found in {directory}")

    truth, regression, final = [], [], []
    for path in files:
        with np.load(path) as data:
            hr = np.asarray(data["hr"], dtype=np.float64).squeeze()
            reg = np.asarray(data["reg"], dtype=np.float64).squeeze()
            sample_final = np.asarray(data["final"], dtype=np.float64).squeeze()

        if sample_final.ndim > 2:
            sample_final = sample_final.mean(axis=0)
        truth.append(denorm(hr) if is_norm_space(hr) else hr)
        regression.append(denorm(reg) if is_norm_space(reg) else reg)
        final.append(
            denorm(sample_final) if is_norm_space(sample_final) else sample_final
        )

    return np.stack(truth), np.stack(regression), np.stack(final)


def load_pm_regression(directory, alternate_directory=None):
    primary = sorted(Path(directory).glob("day_*.npz"))
    values = []

    if primary:
        for path in primary:
            with np.load(path) as data:
                regression = np.asarray(data["reg"], dtype=np.float64)
            regression = np.squeeze(regression)
            if regression.ndim == 2:
                values.append(regression)
            else:
                values.extend(regression[index] for index in range(regression.shape[0]))
    elif alternate_directory is not None:
        files = sorted(Path(alternate_directory).glob("reg_*.npy"))
        for path in files:
            regression = np.asarray(np.load(path), dtype=np.float64).squeeze()
            values.append(denorm(regression) if is_norm_space(regression) else regression)
    else:
        files = []

    if not values:
        searched = [str(directory)]
        if alternate_directory is not None:
            searched.append(str(alternate_directory))
        raise FileNotFoundError(f"No PM2O3 regression predictions found in {searched}")

    result = np.stack(values)
    return denorm(result) if is_norm_space(result) else result


def load_pm_final(directory):
    files = sorted(Path(directory).glob("final_*.npy"))
    if not files:
        raise FileNotFoundError(f"No final_*.npy files found in {directory}")

    values = []
    for path in files:
        final = np.asarray(np.load(path), dtype=np.float64).squeeze()
        if final.ndim > 2:
            final = final.mean(axis=0)
        values.append(denorm(final) if is_norm_space(final) else final)
    return np.stack(values)


def radial_psd(map2d):
    from numpy.fft import fft2, fftshift

    height, width = map2d.shape
    spectrum = fftshift(fft2(map2d.astype(np.float64)))
    power = np.abs(spectrum) ** 2
    center_y, center_x = height // 2, width // 2
    y, x = np.indices((height, width))
    radius = np.sqrt((y - center_y) ** 2 + (x - center_x) ** 2).astype(int)
    summed = np.bincount(radius.ravel(), power.ravel())
    counts = np.bincount(radius.ravel())
    return (summed / counts)[1:]


def high_k_energy(maps):
    spectrum = np.mean([radial_psd(sample) for sample in maps], axis=0)
    return float(spectrum[len(spectrum) * 2 // 3 :].sum())


def compute_metrics(prediction, truth):
    predicted = prediction.ravel().astype(np.float64)
    observed = truth.ravel().astype(np.float64)
    error = predicted - observed
    ss_res = np.sum(error**2)
    ss_tot = np.sum((observed - observed.mean()) ** 2)
    predicted_centered = predicted - predicted.mean()
    observed_centered = observed - observed.mean()
    correlation_denominator = np.sqrt(
        np.sum(predicted_centered**2) * np.sum(observed_centered**2)
    )
    truth_high_k = high_k_energy(truth)

    return {
        "RMSE": float(np.sqrt(np.mean(error**2))),
        "MAE": float(np.mean(np.abs(error))),
        "Bias": float(np.mean(error)),
        "R": float(
            np.sum(predicted_centered * observed_centered)
            / correlation_denominator
        )
        if correlation_denominator > 0
        else 0.0,
        "R2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0,
        "std_ratio": float(np.std(predicted) / np.std(observed))
        if np.std(observed) > 0
        else 0.0,
        "high_k_energy_ratio": high_k_energy(prediction) / truth_high_k
        if truth_high_k > 0
        else 0.0,
    }


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate O3 predictions against a common high-resolution truth set."
    )
    parser.add_argument("--o3-dir", type=Path, required=True)
    parser.add_argument("--bilinear-file", type=Path, required=True)
    parser.add_argument("--pm-reg-dir", type=Path, required=True)
    parser.add_argument("--pm-final-dir", type=Path, required=True)
    parser.add_argument("--pm-reg-alt-dir", type=Path)
    parser.add_argument("--output", type=Path, default=Path("canonical_metrics.csv"))
    return parser.parse_args()


def main():
    args = parse_args()
    truth, o3_reg, o3_final = load_o3_results(args.o3_dir)

    with np.load(args.bilinear_file) as data:
        bilinear = np.asarray(data["pred_maps"], dtype=np.float64)

    pm_reg = load_pm_regression(args.pm_reg_dir, args.pm_reg_alt_dir)
    pm_final = load_pm_final(args.pm_final_dir)

    models = {
        "LR-inter": bilinear,
        "O3_retrain_reg": o3_reg,
        "O3_retrain_final": o3_final,
        "PM2O3_frozen_reg": pm_reg,
        "PM2O3_frozen_final": pm_final,
    }

    rows = []
    for name, prediction in models.items():
        if prediction.shape != truth.shape:
            raise ValueError(
                f"{name}: shape mismatch {prediction.shape} versus {truth.shape}"
            )
        metrics = compute_metrics(prediction, truth)
        metrics["Model"] = name
        rows.append(metrics)

    columns = [
        "RMSE",
        "MAE",
        "Bias",
        "R",
        "R2",
        "std_ratio",
        "high_k_energy_ratio",
    ]
    result = pd.DataFrame(rows).set_index("Model")[columns]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(args.output)
    print(args.output)


if __name__ == "__main__":
    main()
