"""外推推理：用已训练 FLAML 模型对 2019 数据做预测（纯 numpy，不依赖 torch）"""

import argparse
import json
import os
import sys
from pathlib import Path

import joblib
import numpy as np
import yaml

from data.transforms import (
    standard_norm_np,
    denorm,
    build_coord_grid,
    bilinear_upsample_np,
    parse_minmax_txt,
)


def load_config(config_path):
    """从 configs/exp_A_extrap2019.yaml 加载配置（独立，不需 default.yaml 合并）"""
    with open(config_path) as f:
        return yaml.safe_load(f)


def build_features_for_timestep(
    input_data_t,
    train_channel_order,
    train_minmax,
    H,
    W,
    coord,
    landuse_data_t=None,
    landuse_keys=None,
):
    """对单个时间步构建特征矩阵 (H*W, n_features)

    Args:
        input_data_t: dict {varname: (17,22) raw array} 一个时间步的 10 个条件变量
        train_channel_order: 训练通道顺序 ['PM25','blh',...]
        train_minmax: {varname: (lo, hi)} 训练期 minmax
        H, W: 目标空间维度 (144, 192)
        coord: (H*W, 2) 坐标特征
        landuse_data_t: dict {varname: (144,192)} land-use 数据（可选，配置 B 需要）
        landuse_keys: land-use 变量名列表（可选）

    Returns:
        X: (H*W, n_features) float32
    """
    channel_list = []
    for varname in train_channel_order:
        raw = input_data_t[varname]  # (17, 22)
        lo, hi = train_minmax[varname]
        norm = standard_norm_np(raw, lo, hi)  # (17, 22) in [-1,1]
        upsampled = bilinear_upsample_np(norm, (H, W))  # (1, 144, 192)
        channel_list.append(upsampled[0])  # (144, 192)

    # Stack: (10, 144, 192) → transpose → (144*192, 10)
    lr_stack = np.stack(channel_list, axis=0)  # (10, 144, 192)
    lr_flat = lr_stack.reshape(10, -1).T  # (H*W, 10)

    # Add coords
    parts = [lr_flat, coord.astype(np.float32)]

    # Add land-use features (配置 B)
    if landuse_data_t is not None and landuse_keys is not None:
        for k in landuse_keys:
            lu = landuse_data_t[k]  # (144, 192), already at target resolution, in [0,1]
            parts.append(lu.ravel()[:, np.newaxis])

    X = np.concatenate(parts, axis=1)
    return X


def compute_out_of_range(input_data, train_channel_order, train_minmax):
    """计算每个条件变量落在训练 [min,max] 之外的样本比例

    Args:
        input_data: list of dict, 每个元素是一个时间步的 {varname: (17,22)}
        train_channel_order: list of varnames
        train_minmax: {varname: (lo, hi)}

    Returns:
        dict {varname: fraction_out_of_range}
    """
    results = {}
    n_total = 0
    counts = {v: 0 for v in train_channel_order}

    for t, data_t in enumerate(input_data):
        for varname in train_channel_order:
            raw = data_t[varname].ravel()
            lo, hi = train_minmax[varname]
            below = (raw < lo).sum()
            above = (raw > hi).sum()
            counts[varname] += below + above
        n_total += raw.size  # 17*22 = 374 pixels per timestep

    for varname in train_channel_order:
        results[varname] = float(counts[varname] / (n_total))

    return results


def inverse_transform(pred_resid, pm_coarse, target_mode, clip_negative):
    """将模型预测的残差反变换回物理 hPM25"""
    if target_mode == "log_residual":
        hpm_pred = np.expm1(pred_resid + np.log1p(np.clip(pm_coarse, 0, None)))
    elif target_mode == "log_direct":
        hpm_pred = np.expm1(pred_resid)
    elif target_mode == "residual":
        hpm_pred = pred_resid + pm_coarse
    else:
        hpm_pred = pred_resid
    if clip_negative:
        hpm_pred = np.clip(hpm_pred, 0, None)
    return hpm_pred


def main(config_path, smoke=False):
    cfg = load_config(config_path)

    data_path = Path(cfg["paths"]["data_path"])
    model_path = Path(cfg["paths"]["model_path"])
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    H, W = 144, 192
    max_ts = 10 if smoke else None
    use_landuse = cfg["features"].get("use_landuse", False)
    pol = cfg["target"].get("pollutant", "PM25")  # 污染物: PM25 或 O3
    hpol = f"h{pol}"

    # --- 1. 加载训练期特征名 ---
    with open(cfg["paths"]["train_feature_names"]) as f:
        train_feature_names = json.load(f)

    train_channel_order = train_feature_names[:10]  # 10 condition vars
    landuse_keys = None
    if use_landuse:
        landuse_keys = ["frac_urban", "frac_industry_transport", "frac_forest"]
    print(f"Train channel order: {train_channel_order}")
    print(f"Expected feature columns ({len(train_feature_names)}): {train_feature_names}")
    print(f"Land-use: {use_landuse}")

    # --- 2. 加载训练期 minmax ---
    train_minmax = parse_minmax_txt(cfg["paths"]["train_minmax_path"])
    for v in train_channel_order:
        lo, hi = train_minmax[v]
        print(f"  minmax[{v}]: [{lo:.4f}, {hi:.4f}]")

    # --- 3. 读取 2019 input.h5（+ output.h5 的 land-use 如需要） ---
    import h5py

    print(f"\nLoading 2019 data from {data_path}")
    with h5py.File(data_path / "input.h5", "r") as f_in, \
         h5py.File(data_path / "output.h5", "r") as f_out:
        available_vars = list(f_in.keys())
        data_vars = [v for v in available_vars if v not in ("lat", "lon", "time")]
        n_timesteps = f_in[list(f_in.keys())[0]].shape[0]
        print(f"  timesteps={n_timesteps}, data vars={data_vars}")

        if max_ts:
            n_timesteps = min(n_timesteps, max_ts)

        pol_lo, pol_hi = train_minmax[pol]

        all_X = []
        all_pm_coarse = []
        input_snapshots = []

        for t in range(n_timesteps):
            snapshot = {}
            for v in data_vars:
                raw_arr = f_in[v][t, :, :]
                if raw_arr.ndim < 2:
                    raw_arr = f_in[v][t]
                snapshot[v] = np.array(raw_arr, dtype=np.float32)
            input_snapshots.append(snapshot)

        # 读取 land-use（静态，所有时间步相同，直接取第一个时间步）
        if use_landuse:
            landuse_static = {}
            for k in landuse_keys:
                landuse_static[k] = np.array(f_out[k][0, :, :], dtype=np.float32)
            print(f"  Land-use keys: {list(landuse_static.keys())}")

    # --- 4. 对每个时间步构建特征矩阵 ---
    coord = build_coord_grid(H, W)

    for t, snapshot in enumerate(input_snapshots):
        X = build_features_for_timestep(
            snapshot, train_channel_order, train_minmax, H, W, coord,
            landuse_data_t=landuse_static if use_landuse else None,
            landuse_keys=landuse_keys,
        )
        all_X.append(X)

        # pm_coarse：物理量的上采样粗 PM2.5
        raw_pol = snapshot[pol]  # (17, 22)
        norm_pol = standard_norm_np(raw_pol, pol_lo, pol_hi)
        up_pol = bilinear_upsample_np(norm_pol, (H, W))[0]  # (144, 192)
        pm_coarse_phys = denorm(up_pol, pol_lo, pol_hi)
        all_pm_coarse.append(pm_coarse_phys.ravel())

    X_full = np.concatenate(all_X, axis=0)  # (n_ts * H * W, 12)
    pm_coarse_full = np.concatenate(all_pm_coarse, axis=0)

    # --- 5. 断言特征列 ---
    assert X_full.shape[1] == len(train_feature_names), (
        f"X.shape[1]={X_full.shape[1]} != len(feature_names)={len(train_feature_names)}"
    )
    print(f"\nFeature matrix: X={X_full.shape}, columns OK ({X_full.shape[1]})")

    # --- 6. out-of-range 诊断 ---
    oo_range = compute_out_of_range(input_snapshots, train_channel_order, train_minmax)
    print("\nOut-of-range fractions (train minmax coverage):")
    for v in train_channel_order:
        print(f"  {v}: {oo_range[v]*100:.2f}%")

    # --- 7. Load 模型 & 预测 ---
    print(f"\nLoading model from {model_path}")
    automl = joblib.load(model_path)
    print(f"Model: {automl.best_estimator}")

    print("Predicting...")
    pred_resid = automl.predict(X_full)  # (N,)

    # --- 8. 逆变换 ---
    target_mode = cfg["target"]["mode"]
    clip_negative = cfg["target"]["clip_negative"]
    hpm_pred = inverse_transform(pred_resid, pm_coarse_full, target_mode, clip_negative)

    # --- 9. 重建空间场 ---
    pred_maps = hpm_pred.reshape(n_timesteps, H, W)
    bilinear_maps = pm_coarse_full.reshape(n_timesteps, H, W)

    # 读取 output.h5 的真值
    with h5py.File(data_path / "output.h5", "r") as f:
        true_2019 = np.array(f[hpol][:n_timesteps, :, :], dtype=np.float32)
    true_maps = true_2019

    # --- 10. 自检 ---
    assert not np.isnan(pred_maps).any(), "pred_maps contains NaN!"
    assert not np.isinf(pred_maps).any(), "pred_maps contains Inf!"
    assert pred_maps.shape == (n_timesteps, H, W), f"shape {pred_maps.shape} != ({n_timesteps},{H},{W})"
    print(f"pred_maps: {pred_maps.shape} OK, no NaN/Inf")

    # --- 11. 保存 ---
    np.savez_compressed(
        output_dir / "pred_external.npz",
        pred_maps=pred_maps,
        true_maps=true_maps,
        bilinear_maps=bilinear_maps,
        test_t=np.arange(n_timesteps),
    )

    # split_meta (minimal)
    split_meta = {
        "n_timesteps": n_timesteps,
        "H": H,
        "W": W,
        "target_mode": target_mode,
    }
    with open(output_dir / "split_meta.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    # 诊断
    diagnostics = {
        "out_of_range_fractions": oo_range,
        "train_channel_order": train_channel_order,
        "feature_names": train_feature_names,
        "n_timesteps": n_timesteps,
        "model_best_loss": float(automl.best_loss),
    }
    with open(output_dir / "diagnostics.json", "w") as f:
        json.dump(diagnostics, f, indent=2)

    print(f"\nSaved to {output_dir}")
    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, smoke=args.smoke)
