"""共享变换函数 —— 不依赖 torch、不依赖 datasets.ctm。

export_tabular.py 和 predict_external.py 共用此模块，保证逐字节一致。
"""

import numpy as np


def denorm(x, lo, hi, eps=1e-6):
    """反归一化：[-1,1] → 物理单位，与 standard_norm 互逆"""
    return (x + 1) * 0.5 * (hi - lo + eps) + lo


def standard_norm_np(data, lo, hi, eps=1e-6):
    """纯 numpy 版 standard_norm：物理量 → [-1,1]

    与 datasets/ctm.py::standard_norm 逐字节一致：
      y = (x - lo) / (hi - lo + eps)
      y = y * 2 - 1
    """
    data = np.asarray(data, dtype=np.float32)
    y = (data - lo) / (hi - lo + eps)
    y = y * 2.0 - 1.0
    return y


def build_coord_grid(H, W):
    """构建归一化坐标网格 (H*W, 2)，lat/lon 索引归一化到 [0,1]"""
    lat = np.arange(H, dtype=np.float32) / (H - 1)
    lon = np.arange(W, dtype=np.float32) / (W - 1)
    lat_grid, lon_grid = np.meshgrid(lat, lon, indexing="ij")
    return np.stack([lat_grid.ravel(), lon_grid.ravel()], axis=1)


def bilinear_upsample_np(data, target_shape):
    """双线性上采样 (C, H_src, W_src) → (C, H_dst, W_dst)

    与 PyTorch F.interpolate(..., mode='bilinear', align_corners=False) 逐像素一致。
    """
    data = np.asarray(data, dtype=np.float32)
    if data.ndim == 2:
        data = data[np.newaxis, :, :]
    C, H_src, W_src = data.shape
    H_dst, W_dst = target_shape

    scale_h = H_dst / H_src
    scale_w = W_dst / W_src

    # PyTorch align_corners=False: src = (dst + 0.5) / scale - 0.5
    ys = (np.arange(H_dst, dtype=np.float32) + 0.5) / scale_h - 0.5
    xs = (np.arange(W_dst, dtype=np.float32) + 0.5) / scale_w - 0.5

    # Clamp to valid range
    ys = np.clip(ys, 0.0, H_src - 1.0)
    xs = np.clip(xs, 0.0, W_src - 1.0)

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = np.minimum(y0 + 1, H_src - 1).astype(np.int32)
    x1 = np.minimum(x0 + 1, W_src - 1).astype(np.int32)

    wy = (ys - y0.astype(np.float32)).reshape(-1, 1)  # (H_dst, 1)
    wx = (xs - x0.astype(np.float32)).reshape(1, -1)  # (1, W_dst)

    result = np.zeros((C, H_dst, W_dst), dtype=np.float32)
    for c in range(C):
        src = data[c]
        # bilinear weights
        top = src[y0][:, x0] * (1 - wx) + src[y0][:, x1] * wx       # (H_dst, W_dst)
        bot = src[y1][:, x0] * (1 - wx) + src[y1][:, x1] * wx
        result[c] = (1 - wy) * top + wy * bot

    return result


def parse_minmax_txt(path):
    """解析 minmax.txt 返回 {varname: (lo, hi)} 字典

    格式示例：PM25 mins0.0195   maxs328.56
    """
    result = {}
    with open(path) as f:
        for line in f:
            tokens = line.strip().split()
            if len(tokens) < 2:
                continue
            varname = tokens[0]
            lo_str = line[line.find("mins") + 4 : line.find("mine")]
            hi_str = line[line.find("maxs") + 4 : line.find("maxe")]
            result[varname] = (float(lo_str), float(hi_str))
    return result
