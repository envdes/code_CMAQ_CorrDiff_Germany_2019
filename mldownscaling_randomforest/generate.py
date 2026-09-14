"""Inference-only: apply trained RF to 2019 hourly data, frame-by-frame.

Pure numpy + h5py — no torch dependency. Reads h5 directly, does bilinear
upsample + standard_norm in numpy (pixel-identical to WeatherDataset).

Supports two target transforms (must match the model's training transform):
  --target-transform none   : prediction is directly physical μg/m³
  --target-transform log1p  : model output is log1p; expm1+clip to restore physical

Runtime environment: flaml_ds (/path/to/home/anaconda3/envs/flaml_ds/bin/python)
Dependencies: numpy, joblib, h5py, xgboost (no torch, no xarray)

Usage:
    python predict_hourly.py                                          # baseline
    python predict_hourly.py --target-transform log1p --output-suffix logy \
        --model-path models/rf_hpm25_logy.joblib                     # logy ablation
"""

import argparse
import os
import time
from pathlib import Path

import h5py
import joblib
import numpy as np


# ---------------------------------------------------------------------------
# Pure-numpy transforms (pixel-identical to datasets/ctm.py + torch)
# ---------------------------------------------------------------------------

def parse_minmax_txt(path):
    """Parse minmax.txt, return {varname: (lo, hi)}."""
    result = {}
    with open(path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            varname = tokens[0]
            lo_str = line[line.find("mins") + 4:line.find("mine")]
            hi_str = line[line.find("maxs") + 4:line.find("maxe")]
            result[varname] = (float(lo_str), float(hi_str))
    return result


def standard_norm_np(data, lo, hi, eps=1e-6):
    """Physical → [-1, 1]. Byte-identical to datasets/ctm.py::standard_norm."""
    data = np.asarray(data, dtype=np.float32)
    y = (data - lo) / (hi - lo + eps)
    y = y * 2.0 - 1.0
    return y


def bilinear_upsample_np(data, target_shape):
    """Bilinear upsample (C,H_src,W_src) → (C,H_dst,W_dst).

    Pixel-identical to torch F.interpolate(..., mode='bilinear', align_corners=False).
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    C, H_src, W_src = data.shape
    H_dst, W_dst = target_shape

    scale_h = H_dst / H_src
    scale_w = W_dst / W_src

    ys = (np.arange(H_dst, dtype=np.float32) + 0.5) / scale_h - 0.5
    xs = (np.arange(W_dst, dtype=np.float32) + 0.5) / scale_w - 0.5
    ys = np.clip(ys, 0.0, H_src - 1.0)
    xs = np.clip(xs, 0.0, W_src - 1.0)

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, H_src - 1).astype(np.int32)
    x1 = np.minimum(x0 + 1, W_src - 1).astype(np.int32)

    wy = (ys - y0.astype(np.float32)).reshape(-1, 1)
    wx = (xs - x0.astype(np.float32)).reshape(1, -1)

    result = np.zeros((C, H_dst, W_dst), dtype=np.float32)
    for c in range(C):
        src = data[c]
        top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx
        bot = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
        result[c] = (1 - wy) * top + wy * bot

    return result


def denorm(x, lo, hi, eps=1e-6):
    """[-1, 1] → physical. Inverse of standard_norm_np."""
    return (x + 1.0) * 0.5 * (hi - lo + eps) + lo


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Apply trained RF to 2019 hourly data"
    )
    parser.add_argument(
        "--data-path",
        default="/path/to/data/generate2019/Hour_generate_data",
        help="Path to 2019 hourly h5 files",
    )
    parser.add_argument(
        "--model-path",
        default="models/rf_hpm25.joblib",
        help="Path to trained RF model (.joblib)",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Output directory for pred npz",
    )
    parser.add_argument(
        "--train-minmax",
        default="/path/to/data/newgrid/traindata/minmax.txt",
        help="Training minmax.txt (for 10 input channel normalization)",
    )
    parser.add_argument("--target-transform", default="none",
                        choices=["none", "log1p"],
                        help="Target transform matching model training")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix appended to output filename (e.g. 'logy')")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main inference loop
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load model
    print(f"Loading model from {args.model_path}")
    model = joblib.load(args.model_path)
    print(f"Model type: {type(model).__name__}")

    # Channel order (must match training: WeatherDataset.low_keys)
    channel_order = ["PM25", "blh", "tp", "t2m", "u10", "RH", "sp", "v10", "ssrd", "tcc"]
    H, W = 144, 192

    # Parse minmax for input channel normalization (from TRAINING data only)
    train_minmax = parse_minmax_txt(args.train_minmax)

    channel_minmax = {}
    for v in channel_order:
        if v in train_minmax:
            channel_minmax[v] = train_minmax[v]
        else:
            raise KeyError(f"Variable '{v}' not found in training minmax: {args.train_minmax}")

    print(f"Input channel minmax source: {args.train_minmax}")
    print(f"Target transform: {args.target_transform}")

    # Open h5 files
    input_path = os.path.join(args.data_path, "input.h5")
    output_path = os.path.join(args.data_path, "output.h5")

    with h5py.File(input_path, "r") as f_in, h5py.File(output_path, "r") as f_out:
        n_frames = f_out["hPM25"].shape[0]
        print(f"n_frames = {n_frames}")

        # Pre-allocate
        pred_maps = np.zeros((n_frames, H, W), dtype=np.float32)
        true_maps = np.zeros((n_frames, H, W), dtype=np.float32)
        rows_per_frame = H * W  # 27648

        t0 = time.time()
        for i in range(n_frames):
            # --- Build feature vector: 10 channels, normalized, upsampled ---
            channels = []
            for varname in channel_order:
                raw = np.array(f_in[varname][i, 0:17, 0:22], dtype=np.float32)
                lo, hi = channel_minmax[varname]
                norm = standard_norm_np(raw, lo, hi)       # (17, 22), [-1,1]
                up = bilinear_upsample_np(norm, (H, W))     # (1, 144, 192)
                channels.append(up[0])                       # (144, 192)

            lr_stack = np.stack(channels, axis=0)            # (10, 144, 192)
            lr_flat = lr_stack.reshape(10, rows_per_frame).T  # (27648, 10)

            # --- Predict ---
            pred = model.predict(lr_flat).reshape(H, W)

            if args.target_transform == "log1p":
                pred = np.expm1(pred)
                pred = np.clip(pred, 0, None)

            pred_maps[i] = pred

            # --- Truth (hourly h5 stores physical units directly, no denorm needed) ---
            true_maps[i] = np.array(f_out["hPM25"][i, :, :], dtype=np.float32)

            if i % 200 == 0:
                elapsed = time.time() - t0
                fps = (i + 1) / elapsed if elapsed > 0 else 0
                print(f"  frame {i}/{n_frames} ({fps:.1f} frames/s)")

        elapsed = time.time() - t0
        print(f"Prediction done in {elapsed:.1f}s ({n_frames/elapsed:.1f} frames/s)")

    # Save
    suffix = args.output_suffix
    fname = f"rf_pred_hourly{suffix}.npz"
    out_path = os.path.join(args.output_dir, fname)
    np.savez_compressed(out_path, pred_maps=pred_maps, true_maps=true_maps)
    print(f"\nSaved: {out_path}")
    print(f"pred_maps physical: min={pred_maps.min():.4f}, max={pred_maps.max():.4f}, "
          f"mean={pred_maps.mean():.4f}, std={pred_maps.std():.4f}")
    print(f"true_maps physical: min={true_maps.min():.4f}, max={true_maps.max():.4f}, "
          f"mean={true_maps.mean():.4f}, std={true_maps.std():.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
