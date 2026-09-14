"""Compute sigma_data (global and per-channel std) from the WeatherDataset HR targets.

Usage:
    /path/to/conda/envs/nemo/bin/python scripts/compute_sigma_data.py
"""

import sys
sys.path.insert(0, "/path/to/data")

import numpy as np
import torch

from datasets.ctm_local import WeatherDataset, OUTPUT_NAMES


def main():
    ds = WeatherDataset(
        data_path="/path/to/data/newgrid/traindata",
        datatype="h5",
        pipeline={"normal_type": "min_max"},
        phase="train",
    )

    n_total = len(ds)
    # Sample evenly across the dataset
    n_sample = min(500, n_total)
    step = max(1, n_total // n_sample)

    all_values = []
    per_channel = {name: [] for name in OUTPUT_NAMES}

    for i in range(0, n_total, step):
        hr, _ = ds[i]  # hr: (C, H, W), already normalised to [-1, 1]
        hr_np = hr.numpy()
        all_values.append(hr_np.flatten())
        for c, name in enumerate(OUTPUT_NAMES):
            per_channel[name].append(hr_np[c].flatten())

    all_values = np.concatenate(all_values)
    global_std = float(np.std(all_values))

    print(f"Sampled {len(all_values) // (ds.hr_shape[0] * ds.hr_shape[1])} frames ({len(all_values)} pixels)")
    print(f"\n{'='*50}")
    print(f"Global std (single-channel, hPM25-log): {global_std:.6f}")
    print(f"\nPer-channel std:")
    for name in OUTPUT_NAMES:
        ch_data = np.concatenate(per_channel[name])
        ch_std = float(np.std(ch_data))
        print(f"  {name:25s}: {ch_std:.6f}")
    print(f"{'='*50}")
    print(f"\nRecommended sigma_data for diffusion.yaml: {global_std:.6f}")
    print(f"(If per-channel stds differ greatly, consider per-channel weighting.)")


if __name__ == "__main__":
    main()
