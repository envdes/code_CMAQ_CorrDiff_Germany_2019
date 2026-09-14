"""数据导出：WeatherDataset → 逐像元表格 npz（corrdiff 环境运行）"""

import argparse
import json
import os
import sys
import numpy as np
from pathlib import Path

from data.transforms import denorm, build_coord_grid


def temporal_split(N, val_frac, test_frac):
    """按时间切分 train/val/test 索引（只按时间顺序，绝不随机切像元）"""
    n_test = int(np.ceil(N * test_frac))
    n_val = int(np.ceil(N * val_frac))
    n_train = N - n_val - n_test
    train_t = list(range(0, n_train))
    val_t = list(range(n_train, n_train + n_val))
    test_t = list(range(n_train + n_val, N))
    return train_t, val_t, test_t


def load_config(config_path):
    """加载配置：先读 default.yaml，再用指定 config 覆盖"""
    import yaml

    base_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    with open(config_path) as f:
        override = yaml.safe_load(f) or {}
    _deep_merge(cfg, override)
    return cfg


def _deep_merge(base, override):
    """递归合并 override 到 base（原地修改 base）"""
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def make_rows(ds, timesteps, cfg, full_grid=False):
    """从 WeatherDataset 按时间步取出数据，反归一化，flatten 成表格行。

    Args:
        ds: WeatherDataset 实例（phase='train'）
        timesteps: 时间步索引列表
        cfg: 完整配置 dict
        full_grid: True=取全部像元（测试集），False=随机子采样（训练/验证集）

    Returns:
        dict with keys: X, y, pm_coarse, hpm_true, feature_names
    """
    H, W = cfg["grid"]["H"], cfg["grid"]["W"]
    n_pixels = H * W

    # 污染物变量名（PM25 或 O3）
    pol = cfg["target"].get("pollutant", "PM25")
    pol_lo, pol_hi = ds.minmax[pol]
    hpol_lo, hpol_hi = ds.minmax[f"h{pol}"]

    coord = build_coord_grid(H, W)

    X_list, y_list, pm_coarse_list, hpm_true_list = [], [], [], []
    seed = cfg["subsample"]["seed"]
    pixels_per_ts = cfg["subsample"]["pixels_per_timestep"]

    for idx, t in enumerate(timesteps):
        hr, lr = ds[t]
        hr_np = hr.cpu().numpy().astype(np.float32)
        lr_np = lr.cpu().numpy().astype(np.float32)

        # flatten spatial dims: (C, H, W) → (H*W, C)
        hr_flat = hr_np.reshape(hr_np.shape[0], -1).T  # (H*W, 4)
        lr_flat = lr_np.reshape(lr_np.shape[0], -1).T  # (H*W, 10)

        # 反归一化 PM2.5 到物理单位
        pm_coarse = denorm(lr_flat[:, 0], pol_lo, pol_hi)
        hpm_true = denorm(hr_flat[:, 0], hpol_lo, hpol_hi)

        # 目标变换
        target_mode = cfg["target"]["mode"]
        if target_mode == "log_residual":
            y = np.log1p(np.clip(hpm_true, 0, None)) - np.log1p(np.clip(pm_coarse, 0, None))
        elif target_mode == "log_direct":
            y = np.log1p(np.clip(hpm_true, 0, None))
        elif target_mode == "residual":
            y = hpm_true - pm_coarse
        else:  # direct
            y = hpm_true

        # 特征矩阵
        X = lr_flat.copy()

        if cfg["features"]["use_coords"]:
            X = np.concatenate([X, coord.astype(np.float32)], axis=1)

        if cfg["features"]["use_landuse"]:
            landuse = (hr_flat[:, 1:4] + 1) * 0.5
            X = np.concatenate([X, landuse.astype(np.float32)], axis=1)

        if full_grid:
            X_list.append(X)
            y_list.append(y)
            pm_coarse_list.append(pm_coarse)
            hpm_true_list.append(hpm_true)
        else:
            rng = np.random.RandomState(seed + t)
            n_sample = min(pixels_per_ts, n_pixels)
            indices = rng.choice(n_pixels, size=n_sample, replace=False)
            X_list.append(X[indices])
            y_list.append(y[indices])
            pm_coarse_list.append(pm_coarse[indices])
            hpm_true_list.append(hpm_true[indices])

    feature_names = list(ds.low_keys)
    if cfg["features"]["use_coords"]:
        feature_names += ["lat_norm", "lon_norm"]
    if cfg["features"]["use_landuse"]:
        feature_names += ["frac_urban", "frac_industry_transport", "frac_forest"]

    return {
        "X": np.concatenate(X_list, axis=0).astype(np.float32),
        "y": np.concatenate(y_list, axis=0).astype(np.float32),
        "pm_coarse": np.concatenate(pm_coarse_list, axis=0).astype(np.float32),
        "hpm_true": np.concatenate(hpm_true_list, axis=0).astype(np.float32),
        "feature_names": feature_names,
    }


def main(config_path, smoke=False):
    cfg = load_config(config_path)

    # 添加 corrdiff repo 到 path
    corrdiff_repo = cfg["paths"]["corrdiff_repo"]
    if corrdiff_repo not in sys.path:
        sys.path.insert(0, corrdiff_repo)

    from datasets.ctm import WeatherDataset

    data_path = cfg["paths"]["data_path"]
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    H, W = cfg["grid"]["H"], cfg["grid"]["W"]

    # 用 phase='train' 获取全部 8686 个时间步
    ds = WeatherDataset(
        data_path=data_path,
        datatype="h5",
        pipeline={"normal_type": "min_max"},
        phase="train",
    )

    N = len(ds)
    print(f"Total timesteps: {N}")

    if smoke:
        cfg["subsample"]["n_train_timesteps"] = 20
        cfg["subsample"]["pixels_per_timestep"] = 500
        val_frac = min(8 / N, cfg["split"]["val_frac"])
        test_frac = min(8 / N, cfg["split"]["test_frac"])
    else:
        val_frac = cfg["split"]["val_frac"]
        test_frac = cfg["split"]["test_frac"]

    train_t, val_t, test_t = temporal_split(N, val_frac, test_frac)

    if smoke:
        n_train = cfg["subsample"]["n_train_timesteps"]
        train_t = train_t[:n_train]
        val_t = val_t[:8]
        test_t = test_t[:8]

    print(
        f"Split sizes: train={len(train_t)}, val={len(val_t)}, test={len(test_t)}"
    )

    # 训练集
    print("Exporting train set...")
    train_data = make_rows(ds, train_t, cfg, full_grid=False)
    np.savez_compressed(output_dir / "tabular_train.npz", **train_data)
    print(f"  X shape: {train_data['X'].shape}, y shape: {train_data['y'].shape}")

    # 验证集
    print("Exporting val set...")
    val_data = make_rows(ds, val_t, cfg, full_grid=False)
    np.savez_compressed(output_dir / "tabular_val.npz", **val_data)
    print(f"  X shape: {val_data['X'].shape}, y shape: {val_data['y'].shape}")

    # 测试集 — full_grid=True
    print("Exporting test set...")
    test_data = make_rows(ds, test_t, cfg, full_grid=True)
    np.savez_compressed(output_dir / "tabular_test.npz", **test_data)
    print(f"  X shape: {test_data['X'].shape}, y shape: {test_data['y'].shape}")

    # feature_names.json
    with open(output_dir / "feature_names.json", "w") as f:
        json.dump(test_data["feature_names"], f, indent=2)

    # split_meta.json
    split_meta = {
        "N": N,
        "train_t": train_t,
        "val_t": val_t,
        "test_t": test_t,
        "H": H,
        "W": W,
        "target_mode": cfg["target"]["mode"],
        "use_landuse": cfg["features"]["use_landuse"],
    }
    with open(output_dir / "split_meta.json", "w") as f:
        json.dump(split_meta, f, indent=2)

    # 自检
    print("\n--- Self-check ---")
    for name, data in [
        ("train", train_data),
        ("val", val_data),
        ("test", test_data),
    ]:
        has_nan = np.isnan(data["X"]).any() or np.isnan(data["y"]).any()
        has_inf = np.isinf(data["X"]).any() or np.isinf(data["y"]).any()
        print(f"  {name}: NaN={has_nan}, Inf={has_inf}")
        assert not has_nan, f"{name} contains NaN!"
        assert not has_inf, f"{name} contains Inf!"
        assert data["X"].shape[1] == len(
            data["feature_names"]
        ), f"{name}: X.shape[1]={data['X'].shape[1]} != len(feature_names)={len(data['feature_names'])}"

    expected_test_samples = len(test_t) * H * W
    assert (
        test_data["X"].shape[0] == expected_test_samples
    ), f"test samples {test_data['X'].shape[0]} != {expected_test_samples}"
    print(f"  test samples OK: {test_data['X'].shape[0]} == {len(test_t)}*{H}*{W}")

    print("\nExport complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config YAML")
    parser.add_argument("--smoke", action="store_true", help="Smoke test mode")
    args = parser.parse_args()
    main(args.config, smoke=args.smoke)
