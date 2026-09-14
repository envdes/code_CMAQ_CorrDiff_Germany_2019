"""Random Forest baseline for PM2.5 downscaling (CorrDiff paper).

Trains one RF (sklearn CPU or xgboost GPU) on pre-exported 2018 training table,
predicts on 2019 test data frame-by-frame, and saves pred_maps as npz.

Supports two target transforms:
  --target-transform none   : y = y_phys (original baseline)
  --target-transform log1p  : y = log1p(y_phys), expm1 at inference

Runtime environment: flaml_ds (/path/to/home/anaconda3/envs/flaml_ds/bin/python)
Dependencies: numpy, scikit-learn, xgboost, joblib (no torch, no xarray)

Usage:
    python build_rf_baseline.py                                          # baseline
    python build_rf_baseline.py --target-transform log1p --output-suffix logy  # logy ablation
"""

import argparse
import os
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


# ---------------------------------------------------------------------------
# argparse
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train RF baseline and predict on 2019 test data"
    )
    parser.add_argument("--train-npz", default="outputs/rf_train.npz")
    parser.add_argument("--test-npz", default="outputs/rf_test.npz")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"],
                        help="Device: cuda (xgboost GPU) or cpu (sklearn)")
    parser.add_argument("--target-transform", default="none",
                        choices=["none", "log1p"],
                        help="Target transform: none=y_phys, log1p=log1p(y_phys)")
    parser.add_argument("--output-suffix", default="",
                        help="Suffix appended to output filenames (e.g. 'logy')")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_data(train_npz, test_npz):
    """Load pre-exported npz files and print shapes + y statistics."""
    print("\n=== Load Data ===")

    train = dict(np.load(train_npz))
    test = dict(np.load(test_npz))

    y = train["y"]
    print(f"Train: X={train['X'].shape} dtype={train['X'].dtype}, "
          f"y={y.shape} dtype={y.dtype}")
    print(f"y_train physical — min={y.min():.4f}, max={y.max():.4f}, "
          f"mean={y.mean():.4f}, std={y.std():.4f}")

    # Safety check: y must be in physical range (μg/m³), not normalized [-1,1]
    if y.max() < 2.0:
        raise ValueError(
            f"y_train max = {y.max():.4f} < 2. "
            f"Target appears to be in normalized range [-1,1] instead of physical μg/m³."
        )

    print(f"Test:  X={test['X'].shape} dtype={test['X'].dtype}, "
          f"true_flat={test['true_flat'].shape} dtype={test['true_flat'].dtype}, "
          f"n_frames={test['n_frames']}, H={test['H']}, W={test['W']}")

    return train, test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_rf(X, y_phys, seed, device, target_transform):
    """Train RF with 5% in-sample holdout sanity check.

    Args:
        X: feature matrix (N, 10), normalized [-1,1].
        y_phys: target in physical μg/m³.
        device: 'cuda' or 'cpu'.
        target_transform: 'none' or 'log1p'.

    Returns:
        (model, r2_physical) — r2_physical is always in physical μg/m³.
    """
    print(f"\n=== Train RF (device={device}, target={target_transform}) ===")

    # Prepare training target
    if target_transform == "log1p":
        y_fit = np.log1p(np.clip(y_phys, 0, None))
        print(f"y_train_log — min={y_fit.min():.4f}, max={y_fit.max():.4f}, "
              f"mean={y_fit.mean():.4f}, std={y_fit.std():.4f}")
    else:
        y_fit = y_phys

    # 5% holdout for in-sample sanity check only
    n_holdout = int(len(y_fit) * 0.05)
    rng = np.random.default_rng(seed)
    indices = rng.permutation(len(y_fit))
    train_idx = indices[n_holdout:]
    holdout_idx = indices[:n_holdout]

    if device == "cuda":
        from xgboost import XGBRFRegressor
        rf = XGBRFRegressor(
            n_estimators=100, learning_rate=1.0, subsample=0.8,
            device="cuda", random_state=seed, verbosity=1,
        )
        print(f"XGBRFRegressor(n_estimators=100, device=cuda, random_state={seed})")
    else:
        rf = RandomForestRegressor(
            n_estimators=100, n_jobs=-1, random_state=seed,
        )
        print(f"RandomForestRegressor(n_estimators=100, n_jobs=-1, random_state={seed})")

    t0 = time.time()
    rf.fit(X[train_idx], y_fit[train_idx])
    elapsed = time.time() - t0

    # Predict on holdout, then restore to physical if log1p
    y_holdout_pred = rf.predict(X[holdout_idx])
    y_holdout_true = y_phys[holdout_idx]  # always physical

    if target_transform == "log1p":
        y_holdout_pred = np.expm1(y_holdout_pred)
        y_holdout_pred = np.clip(y_holdout_pred, 0, None)
        r2_log = r2_score(y_fit[holdout_idx], rf.predict(X[holdout_idx]))
        print(f"In-sample R² (log-space): {r2_log:.4f}")

    r2_phys = r2_score(y_holdout_true, y_holdout_pred)
    print(f"Training time: {elapsed:.1f}s")
    print(f"In-sample R² (5% holdout, physical μg/m³): {r2_phys:.4f}")
    return rf, r2_phys


# ---------------------------------------------------------------------------
# Prediction (frame-by-frame)
# ---------------------------------------------------------------------------

def predict_test(model, X_test, n_frames, H, W, target_transform):
    """Predict frame-by-frame. For log1p, applies expm1+clip to restore physical."""
    print(f"\n=== Predict 2019 (target={target_transform}) ===")
    pred_maps = np.zeros((n_frames, H, W), dtype=np.float32)
    rows_per_frame = H * W

    for i in range(n_frames):
        start = i * rows_per_frame
        end = start + rows_per_frame
        x_frame = X_test[start:end]
        pred = model.predict(x_frame).reshape(H, W)

        if target_transform == "log1p":
            pred = np.expm1(pred)
            pred = np.clip(pred, 0, None)

        pred_maps[i] = pred
        if i % 50 == 0:
            print(f"  predicting frame {i}/{n_frames}")

    return pred_maps


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_pred(pred_maps, true_test, n_frames, H, W, output_dir, suffix):
    """Save predicted and true maps as npz."""
    true_maps = true_test.reshape(n_frames, H, W)
    fname = f"rf_pred{suffix}.npz"
    out_path = os.path.join(output_dir, fname)
    np.savez_compressed(
        out_path,
        pred_maps=pred_maps.astype(np.float32),
        true_maps=true_maps.astype(np.float32),
    )
    print(f"pred_maps: {pred_maps.shape}, true_maps: {true_maps.shape}")
    print(f"Saved: {out_path}")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    suffix = args.output_suffix  # e.g. "" or "_logy"

    Path(args.model_dir).mkdir(parents=True, exist_ok=True)
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Load
    train, test = load_data(args.train_npz, args.test_npz)

    # Train
    model, r2 = train_rf(
        train["X"], train["y"], args.seed, args.device, args.target_transform,
    )

    # Save model
    model_name = f"rf_hpm25{suffix}.joblib"
    model_path = os.path.join(args.model_dir, model_name)
    joblib.dump(model, model_path)
    print(f"Model saved: {model_path}")

    # Predict
    n_frames = int(test["n_frames"])
    H = int(test["H"])
    W = int(test["W"])
    pred_maps = predict_test(model, test["X"], n_frames, H, W, args.target_transform)

    # Save predictions
    save_pred(pred_maps, test["true_flat"], n_frames, H, W, args.output_dir, suffix)

    # Summary
    print(f"\n=== Summary ===")
    print(f"Target transform: {args.target_transform}")
    print(f"Device: {args.device}")
    print(f"Model: {model_path}")
    print(f"Pred npz: outputs/rf_pred{suffix}.npz")
    print(f"pred_maps shape: {pred_maps.shape}")
    print(f"pred_maps physical — min={pred_maps.min():.4f}, max={pred_maps.max():.4f}, "
          f"mean={pred_maps.mean():.4f}, std={pred_maps.std():.4f}")
    print(f"Done.")


if __name__ == "__main__":
    main()
