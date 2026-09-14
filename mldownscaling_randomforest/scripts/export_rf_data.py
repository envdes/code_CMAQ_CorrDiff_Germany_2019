"""Export RF training/test data from WeatherDataset to npz (nemo env).

Usage:
    # Step 1: export data
    python scripts/export_rf_data.py

    # Step 3 (after Phase 2): assemble final NetCDF
    python scripts/export_rf_data.py --assemble-netcdf outputs/rf_pred.npz \
        --output-dir outputs -o rf_pred_2019.nc
"""

import argparse
import os
import sys
from pathlib import Path

import h5py
import numpy as np
import xarray as xr

# Allow import of datasets.ctm from the project root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from datasets.ctm import WeatherDataset  # noqa: E402


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Export RF data from WeatherDataset, or assemble NetCDF"
    )
    parser.add_argument(
        "--train-data",
        default="/path/to/data/newgrid/traindata",
        help="Path to 2018 h5 files + minmax.txt",
    )
    parser.add_argument(
        "--test-data",
        default="/path/to/data/generate2019/Dmean_array",
        help="Path to 2019 h5 files",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Directory for intermediate npz files",
    )
    parser.add_argument(
        "--max-train-rows",
        type=int,
        default=2_000_000,
        help="Max rows in training table",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--assemble-netcdf",
        default=None,
        help="Path to rf_pred.npz from Phase 2 (triggers assembly mode)",
    )
    parser.add_argument(
        "--coords-npz",
        default=None,
        help="Path to coords npz (default: <output-dir>/rf_coords.npz)",
    )
    parser.add_argument(
        "-o",
        "--output-nc",
        default="rf_pred_2019.nc",
        help="Output NetCDF path (assembly mode)",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Dataset loader
# ---------------------------------------------------------------------------

def load_ds(data_path, phase="train", data_len=-1):
    """Instantiate WeatherDataset and print diagnostics.

    Args:
        data_len: Override dataset length. -1 uses WeatherDataset default.
                  Pass the actual h5 timestep count to avoid the built-in -2 truncation.
    """
    ds = WeatherDataset(
        data_path=data_path,
        datatype="h5",
        pipeline={"normal_type": "min_max"},
        phase=phase,
        data_len=data_len,
    )

    hr, lr = ds[0]
    print(f"Return order check: ds[0] returns (hr, lr) — shapes: "
          f"hr={tuple(hr.shape)}, lr={tuple(lr.shape)}")
    print(f"lr channels (10): {ds.low_keys}")
    print(f"hr channels  (4): {ds.high_keys}")

    hpm25_lo, hpm25_hi = ds.minmax["hPM25"]
    print(f"hPM25 minmax: vmin={hpm25_lo:.6f}, vmax={hpm25_hi:.4f}")
    print(f"len(ds) = {len(ds)}")
    return ds


# ---------------------------------------------------------------------------
# Export: training table
# ---------------------------------------------------------------------------

def export_train_data(ds, hpm25_min, hpm25_max, max_train_rows, seed, output_dir):
    """Build subsampled training table and save to npz."""
    n_frames = len(ds)
    rows_per_frame = 144 * 192  # 27648
    per_frame_quota = max(1, max_train_rows // n_frames)
    rng = np.random.default_rng(seed)

    total_rows = n_frames * per_frame_quota
    X_train = np.zeros((total_rows, 10), dtype=np.float32)
    y_train = np.zeros((total_rows,), dtype=np.float32)

    print(f"\n=== Export Training Data ===")
    print(f"n_frames={n_frames}, per_frame_quota={per_frame_quota}, "
          f"total_rows={total_rows}, seed={seed}")

    for i in range(n_frames):
        hr, lr = ds[i]  # hr: (4,144,192), lr: (10,144,192)

        # Flatten lr: (10,144,192) → (27648,10)
        lr_np = lr.numpy().astype(np.float32)
        lr_flat = lr_np.reshape(10, rows_per_frame).T

        # Denormalize hPM25 to physical µg/m³
        hr_np = hr.numpy().astype(np.float32)
        hpm25_norm = hr_np[0]  # (144,192)
        y_phys = (hpm25_norm + 1.0) * 0.5 * (hpm25_max - hpm25_min + 1e-6) + hpm25_min
        y_flat = y_phys.ravel()

        # Per-frame random subsample
        indices = rng.choice(rows_per_frame, size=per_frame_quota, replace=False)
        start = i * per_frame_quota
        end = start + per_frame_quota
        X_train[start:end] = lr_flat[indices]
        y_train[start:end] = y_flat[indices]

        if (i + 1) % 500 == 0:
            print(f"  processed frame {i+1}/{n_frames}")

    out_path = os.path.join(output_dir, "rf_train.npz")
    np.savez_compressed(out_path, X=X_train, y=y_train)
    print(f"Saved: {out_path}")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")


# ---------------------------------------------------------------------------
# Export: test table (2019, full grid, no subsample)
# ---------------------------------------------------------------------------

def export_test_data(ds, hpm25_min, hpm25_max, output_dir):
    """Export full 2019 test data (all pixels, all frames) to npz."""
    n_frames = len(ds)
    rows_per_frame = 144 * 192  # 27648

    print(f"\n=== Export Test Data ===")
    print(f"n_frames={n_frames}")

    X_blocks = []
    true_blocks = []

    for i in range(n_frames):
        hr, lr = ds[i]

        lr_np = lr.numpy().astype(np.float32)
        lr_flat = lr_np.reshape(10, rows_per_frame).T  # (27648, 10)
        X_blocks.append(lr_flat)

        hr_np = hr.numpy().astype(np.float32)
        hpm25_norm = hr_np[0]  # (144, 192)
        true_phys = (hpm25_norm + 1.0) * 0.5 * (hpm25_max - hpm25_min + 1e-6) + hpm25_min
        true_blocks.append(true_phys.ravel())

        if (i + 1) % 100 == 0:
            print(f"  processed frame {i+1}/{n_frames}")

    X_test = np.concatenate(X_blocks, axis=0).astype(np.float32)
    true_test = np.concatenate(true_blocks, axis=0).astype(np.float32)

    out_path = os.path.join(output_dir, "rf_test.npz")
    np.savez_compressed(
        out_path,
        X=X_test,
        true_flat=true_test,
        n_frames=n_frames,
        H=144,
        W=192,
    )
    print(f"Saved: {out_path}")
    print(f"X_test: {X_test.shape}, true_flat: {true_test.shape}")


# ---------------------------------------------------------------------------
# Export: coordinates from output.h5
# ---------------------------------------------------------------------------

def export_coords(test_data_path, n_frames, output_dir):
    """Read lat/lon/time from output.h5 and save to npz."""
    print(f"\n=== Export Coordinates ===")

    with h5py.File(os.path.join(test_data_path, "output.h5"), "r") as f:
        print(f"h5 keys in output.h5: {list(f.keys())}")
        lat = np.array(f["lat"][:], dtype=np.float64)
        lon = np.array(f["lon"][:], dtype=np.float64)
        time = np.array(f["time"][:n_frames], dtype=np.float64)

    out_path = os.path.join(output_dir, "rf_coords.npz")
    np.savez(out_path, lat=lat, lon=lon, time=time)
    print(f"lat: {lat.shape}, lon: {lon.shape}, time: {time.shape}")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# Assembly: pred.npz + coords → NetCDF with groups
# ---------------------------------------------------------------------------

def assemble_netcdf(pred_npz_path, output_dir, output_nc_path, coords_npz=None):
    """Read rf_pred.npz and coords, write NetCDF with prediction/truth groups."""
    print(f"\n=== Assemble NetCDF ===")

    if coords_npz is None:
        coords_npz = os.path.join(output_dir, "rf_coords.npz")
    pred_data = np.load(pred_npz_path)
    coords = np.load(coords_npz)

    pred_maps = pred_data["pred_maps"]  # (n_frames, H, W)
    true_maps = pred_data["true_maps"]  # (n_frames, H, W)
    lat = coords["lat"]
    lon = coords["lon"]
    time = coords["time"]

    n_frames, H, W = pred_maps.shape
    print(f"pred_maps: {pred_maps.shape}, true_maps: {true_maps.shape}")

    # Build DataArrays with dimension names matching score_samples.py convention
    pred_da = xr.DataArray(
        pred_maps[:, np.newaxis, :, :],
        dims=["time", "ensemble", "y", "x"],
        name="hPM25",
    )
    true_da = xr.DataArray(
        true_maps,
        dims=["time", "y", "x"],
        name="hPM25",
    )
    lat_da = xr.DataArray(lat, dims=["y"], name="lat")
    lon_da = xr.DataArray(lon, dims=["x"], name="lon")
    time_da = xr.DataArray(time, dims=["time"], name="time")

    # Root group: coordinates
    root_ds = xr.Dataset({"lat": lat_da, "lon": lon_da, "time": time_da})
    root_ds.to_netcdf(output_nc_path, mode="w")

    # Prediction group: (time, ensemble, y, x)
    pred_ds = xr.Dataset({"hPM25": pred_da})
    pred_ds.to_netcdf(output_nc_path, mode="a", group="prediction")

    # Truth group: (time, y, x)
    true_ds = xr.Dataset({"hPM25": true_da})
    true_ds.to_netcdf(output_nc_path, mode="a", group="truth")

    print(f"NetCDF saved: {output_nc_path}")
    print(f"  prediction/hPM25: (time={n_frames}, ensemble=1, y={H}, x={W})")
    print(f"  truth/hPM25: (time={n_frames}, y={H}, x={W})")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # --- Assembly mode (Step 3) ---
    if args.assemble_netcdf:
        assemble_netcdf(args.assemble_netcdf, args.output_dir,
                        args.output_nc, args.coords_npz)
        print("Done.")
        return

    # --- Export mode (Step 1) ---
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Print all output file paths before starting
    print("=== Output File Plan ===")
    print(f"  {output_dir / 'rf_train.npz'}")
    print(f"  {output_dir / 'rf_test.npz'}")
    print(f"  {output_dir / 'rf_coords.npz'}")
    print()

    # 2018 training data
    print("=== Phase A: Data Loading (2018 train) ===")
    train_ds = load_ds(args.train_data, phase="train")
    hpm25_min, hpm25_max = train_ds.minmax["hPM25"]

    export_train_data(
        train_ds, hpm25_min, hpm25_max,
        args.max_train_rows, args.seed, output_dir,
    )

    # 2019 test data
    # NOTE: use phase="train" (NOT "test") because WeatherDataset hardcodes
    # test-phase length to 2 frames. Also, WeatherDataset subtracts 2 from
    # high_u_length in train phase, so we read the actual h5 timestep count
    # and pass it as data_len to override the built-in truncation.
    print("\n=== Phase A: Data Loading (2019 test) ===")
    with h5py.File(os.path.join(args.test_data, "output.h5"), "r") as f:
        actual_test_frames = f["hPM25"].shape[0]
    print(f"Actual h5 timesteps in 2019 output.h5: {actual_test_frames}")
    test_ds = load_ds(args.test_data, phase="train", data_len=actual_test_frames)

    export_test_data(test_ds, hpm25_min, hpm25_max, output_dir)
    export_coords(args.test_data, len(test_ds), output_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
