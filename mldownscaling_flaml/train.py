"""FLAML 降尺度：训练 + 预测 + 反变换（flaml 环境运行，不准 import torch）"""

import argparse
import json
import sys
import time
import os
from pathlib import Path

import joblib
import numpy as np
import yaml


def load_config(config_path):
    base_path = Path(__file__).resolve().parent.parent / "configs" / "default.yaml"
    with open(base_path) as f:
        cfg = yaml.safe_load(f)
    with open(config_path) as f:
        override = yaml.safe_load(f) or {}
    _deep_merge(cfg, override)
    return cfg


def _deep_merge(base, override):
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v


def inverse_transform(pred_resid, pm_coarse, target_mode, clip_negative):
    """将模型预测的残差反变换回物理 hPM25"""
    if target_mode == "log_residual":
        hpm_pred = np.expm1(pred_resid + np.log1p(np.clip(pm_coarse, 0, None)))
    elif target_mode == "log_direct":
        hpm_pred = np.expm1(pred_resid)
    elif target_mode == "residual":
        hpm_pred = pred_resid + pm_coarse
    else:  # direct
        hpm_pred = pred_resid

    if clip_negative:
        hpm_pred = np.clip(hpm_pred, 0, None)
    return hpm_pred


def main(config_path, smoke=False):
    cfg = load_config(config_path)
    if smoke:
        cfg["flaml"]["time_budget"] = 60
    output_dir = Path(cfg["paths"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 加载数据
    train = dict(np.load(output_dir / "tabular_train.npz"))
    val = dict(np.load(output_dir / "tabular_val.npz"))
    test = dict(np.load(output_dir / "tabular_test.npz"))
    with open(output_dir / "split_meta.json") as f:
        split_meta = json.load(f)
    with open(output_dir / "feature_names.json") as f:
        feature_names = json.load(f)

    Xtr, ytr = train["X"], train["y"]
    Xva, yva = val["X"], val["y"]
    Xte, yte = test["X"], test["y"]
    pm_coarse_te = test["pm_coarse"]
    hpm_true_te = test["hpm_true"]
    test_t = split_meta["test_t"]
    H, W = split_meta["H"], split_meta["W"]
    target_mode = cfg["target"]["mode"]
    clip_negative = cfg["target"]["clip_negative"]

    print(f"Train: X={Xtr.shape}, y={ytr.shape}")
    print(f"Val:   X={Xva.shape}, y={yva.shape}")
    print(f"Test:  X={Xte.shape}, y={yte.shape}")
    print(f"Features ({len(feature_names)}): {feature_names}")

    # 训练
    from flaml import AutoML

    automl = AutoML()
    flaml_cfg = cfg["flaml"]
    print(f"\nTraining FLAML AutoML (time_budget={flaml_cfg['time_budget']}s)...")
    t0 = time.time()

    automl.fit(
        Xtr,
        ytr,
        X_val=Xva,
        y_val=yva,
        task="regression",
        metric=flaml_cfg["metric"],
        estimator_list=flaml_cfg["estimator_list"],
        eval_method=flaml_cfg["eval_method"],
        time_budget=flaml_cfg["time_budget"],
        n_jobs=flaml_cfg["n_jobs"],
        seed=flaml_cfg["seed"],
        verbose=1,
    )

    elapsed = time.time() - t0
    print(f"Training completed in {elapsed:.1f}s")
    print(f"Best estimator: {automl.best_estimator}")
    print(f"Best config: {automl.best_config}")
    print(f"Best val loss ({flaml_cfg['metric']}): {automl.best_loss}")

    # 保存模型
    joblib.dump(automl, output_dir / "model_flaml.joblib")

    # 保存最佳配置
    best_config = {
        "estimator": str(automl.best_estimator),
        "config": str(automl.best_config),
        "best_loss": float(automl.best_loss),
        "train_time_s": elapsed,
        "metric": flaml_cfg["metric"],
        "time_budget": flaml_cfg["time_budget"],
        "feature_names": feature_names,
    }
    with open(output_dir / "best_config.json", "w") as f:
        json.dump(best_config, f, indent=2)

    # 特征重要性
    try:
        importance = automl.feature_importances_
        feat_imp = {
            name: float(imp) for name, imp in zip(feature_names, importance)
        }
        # 按重要性排序
        feat_imp_sorted = dict(
            sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)
        )
        with open(output_dir / "feature_importance.json", "w") as f:
            json.dump(feat_imp_sorted, f, indent=2)
        print(f"Top 5 features: {list(feat_imp_sorted.items())[:5]}")
    except Exception as e:
        print(f"Feature importance not available: {e}")

    # 预测
    print("\nPredicting on test set...")
    pred_resid = automl.predict(Xte)

    # 反变换到物理 hPM25
    hpm_pred = inverse_transform(pred_resid, pm_coarse_te, target_mode, clip_negative)

    # 重建空间场 (n_test, H, W)
    n_test = len(test_t)
    pred_maps = hpm_pred.reshape(n_test, H, W)
    true_maps = hpm_true_te.reshape(n_test, H, W)
    bilinear_maps = pm_coarse_te.reshape(n_test, H, W)

    # 自检
    assert not np.isnan(pred_maps).any(), "pred_maps contains NaN!"
    assert not np.isinf(pred_maps).any(), "pred_maps contains Inf!"
    assert pred_maps.shape == (n_test, H, W), f"shape {pred_maps.shape} != ({n_test},{H},{W})"

    # 保存预测
    np.savez_compressed(
        output_dir / "pred_flaml.npz",
        pred_maps=pred_maps,
        true_maps=true_maps,
        bilinear_maps=bilinear_maps,
        test_t=np.array(test_t),
    )
    print(f"Saved pred_flaml.npz: pred_maps={pred_maps.shape}")

    # 训练日志
    log_path = output_dir / "train_log.txt"
    with open(log_path, "w") as f:
        f.write(f"Train time: {elapsed:.1f}s\n")
        f.write(f"Best estimator: {best_config['estimator']}\n")
        f.write(f"Best val loss: {best_config['best_loss']}\n")
        f.write(f"Features: {feature_names}\n")
        f.write(f"Train X shape: {Xtr.shape}\n")
        f.write(f"Val X shape: {Xva.shape}\n")
        f.write(f"Test X shape: {Xte.shape}\n")
        f.write(f"Test samples: {n_test}\n")
        f.write(f"Target mode: {target_mode}\n")
        f.write(f"Config: {best_config}\n")

    print("Done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    main(args.config, smoke=args.smoke)
