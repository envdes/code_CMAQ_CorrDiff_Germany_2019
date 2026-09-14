"""反归一化 & 对数残差逆变换 round-trip 单元测试"""

import numpy as np

from datasets.ctm import standard_norm


def denorm(x, lo, hi, eps=1e-6):
    """与 export_tabular.py 完全一致的 denorm"""
    return (x + 1) * 0.5 * (hi - lo + eps) + lo


# ---- denorm / standard_norm round-trip ---------------------------------------


def test_denorm_roundtrip_single():
    """单一值：standard_norm → denorm 回到原始值"""
    x_orig = 5.0
    lo, hi = 0.0, 10.0
    x_norm = standard_norm(torch_like(x_orig), lo, hi)
    x_back = denorm(x_norm.numpy().item(), lo, hi)
    assert abs(x_back - x_orig) < 1e-4, f"x_orig={x_orig}, x_back={x_back}"


def test_denorm_roundtrip_array():
    """数组：standard_norm → denorm 回到原始值，全量 1e-4 容差"""
    rng = np.random.RandomState(42)
    x_orig = rng.uniform(0.01, 100.0, size=(1000,)).astype(np.float32)
    lo, hi = 0.002, 1814.0

    x_norm = standard_norm(torch_like(x_orig), lo, hi)
    x_back = denorm(x_norm.numpy(), lo, hi)

    assert np.allclose(x_back, x_orig, atol=1e-4), f"max diff: {np.max(np.abs(x_back - x_orig))}"


def test_denorm_roundtrip_at_boundaries():
    """边界值 round-trip：lo 和 hi"""
    for val, lo, hi in [(0.0, 0.0, 10.0), (10.0, 0.0, 10.0), (5.0, 5.0, 5.0)]:
        x_norm = standard_norm(torch_like(val), lo, hi)
        x_back = denorm(x_norm.numpy().item(), lo, hi)
        assert abs(x_back - val) < 1e-4, f"val={val}, lo={lo}, hi={hi}, back={x_back}"


# ---- log_residual round-trip -------------------------------------------------


def test_log_residual_roundtrip():
    """正向对数残差 + 逆变换，断言重建 hPM25 在 1e-4 容差内"""
    rng = np.random.RandomState(42)
    hpm_true = rng.uniform(0.0, 200.0, size=(5000,)).astype(np.float64)
    pm_coarse = rng.uniform(0.0, 150.0, size=(5000,)).astype(np.float64)

    # 正向
    y = np.log1p(np.clip(hpm_true, 0, None)) - np.log1p(np.clip(pm_coarse, 0, None))

    # 逆变换（与 flaml_downscale.py 的 log_residual 逆变换一致）
    hpm_pred = np.expm1(y + np.log1p(np.clip(pm_coarse, 0, None)))
    hpm_pred = np.clip(hpm_pred, 0, None)

    assert np.allclose(hpm_pred, hpm_true, atol=1e-4), (
        f"max diff: {np.max(np.abs(hpm_pred - hpm_true))}"
    )


def test_log_residual_negative_clipping():
    """负 PM2.5 被 clip 到 0 后 round-trip 仍正确"""
    hpm_true = np.array([-5.0, 0.0, 10.0, 50.0], dtype=np.float64)
    pm_coarse = np.array([-2.0, 5.0, 5.0, 30.0], dtype=np.float64)

    hpm_true_clipped = np.clip(hpm_true, 0, None)
    pm_coarse_clipped = np.clip(pm_coarse, 0, None)

    y = np.log1p(hpm_true_clipped) - np.log1p(pm_coarse_clipped)
    hpm_pred = np.expm1(y + np.log1p(pm_coarse_clipped))
    hpm_pred = np.clip(hpm_pred, 0, None)

    assert np.allclose(hpm_pred, hpm_true_clipped, atol=1e-4)


# ---- helpers -----------------------------------------------------------------


def torch_like(x):
    """将 numpy 数组/标量转为 torch tensor（模拟 standard_norm 的输入）"""
    import torch

    if isinstance(x, np.ndarray):
        return torch.from_numpy(x)
    return torch.tensor(x, dtype=torch.float32)
