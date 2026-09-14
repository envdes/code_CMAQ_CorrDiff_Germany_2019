
from __future__ import annotations

import numpy as np
from matplotlib.ticker import FuncFormatter, MaxNLocator

from figure_style_config import (
    DATA_PATH,
    TICK_PT,
    AXIS_LABEL_PT,
    TICK_WIDTH,
    TICK_LENGTH,
    MAP_BORDER_LINEWIDTH,
    GRID_LINEWIDTH,
    GRID_ALPHA,
)

EXPECTED_MASK_CELLS = 19_665

BINS = [
    ("1–5", lambda x: (x >= 1) & (x <= 5), lambda x: (x <= -1) & (x >= -5)),
    ("6–10", lambda x: (x >= 6) & (x <= 10), lambda x: (x <= -6) & (x >= -10)),
    ("11–18", lambda x: (x >= 11) & (x <= 18), lambda x: (x <= -11) & (x >= -18)),
    ("19–30", lambda x: (x >= 19) & (x <= 30), lambda x: (x <= -19) & (x >= -30)),
    (">30", lambda x: x > 30, lambda x: x < -30),
]


def load_data() -> dict[str, np.ndarray]:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing source data: {DATA_PATH}")
    with np.load(DATA_PATH) as source:
        raw_data = {key: source[key] for key in source.files}

    mask = raw_data["mask"].astype(bool)
    if int(mask.sum()) != EXPECTED_MASK_CELLS:
        raise ValueError("Unexpected German-domain mask; source data may have changed")

    dot_lr = raw_data["dot_lr"]
    dot_hr = raw_data["dot_hr"]
    dot_cd = raw_data["dot_cd"]

    data = {
        "mask": raw_data["mask"],
        "pop": raw_data["pop"],
        "lats": raw_data["lats"],
        "lons": raw_data["lons"],
        "delta_hr_lr": dot_hr - dot_lr,
        "delta_cd_mean": dot_cd.mean(axis=0) - dot_lr if dot_cd.ndim == 3 else dot_cd - dot_lr,
        "delta_cd_member": dot_cd - dot_lr[np.newaxis, :, :] if dot_cd.ndim == 3 else (dot_cd - dot_lr)[np.newaxis, :, :],
    }

    return data


def masked_population(pop: np.ndarray, mask: np.ndarray) -> np.ndarray:
    return np.where(mask, np.nan_to_num(pop, nan=0.0), 0.0)


def population_million(pop: np.ndarray, selection: np.ndarray) -> float:
    return float(pop[selection].sum(dtype=np.float64) / 1e6)


def bin_population(delta: np.ndarray, pop: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    positive = np.array([
        population_million(pop, positive(delta)) for _, positive, _ in BINS
    ])
    negative = np.array([
        population_million(pop, negative(delta)) for _, _, negative in BINS
    ])
    return positive, negative


def mask_extent(lons, lats, mask, padding: float = 0.15):
    lon2d, lat2d = np.meshgrid(lons, lats)
    return (
        float(lon2d[mask].min() - padding),
        float(lon2d[mask].max() + padding),
        float(lat2d[mask].min() - padding),
        float(lat2d[mask].max() + padding),
    )


def configure_map_axis(ax, lons, lats, mask, *, show_ylabel: bool) -> None:
    west, east, south, north = mask_extent(lons, lats, mask)
    ax.set_xlim(west, east)
    ax.set_ylim(south, north)
    ax.set_aspect("equal")

    for spine in ax.spines.values():
        spine.set_color("#3a3a3a")
        spine.set_linewidth(MAP_BORDER_LINEWIDTH)

    ax.contour(
        lons, lats, mask.astype(float), levels=[0.5],
        colors="#202020", linewidths=MAP_BORDER_LINEWIDTH, zorder=3,
    )

    ax.xaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.yaxis.set_major_locator(MaxNLocator(nbins=3))
    ax.xaxis.set_major_formatter(
        FuncFormatter(lambda x, _: f"{abs(x):g}°{'E' if x >= 0 else 'W'}")
    )
    ax.yaxis.set_major_formatter(
        FuncFormatter(lambda y, _: f"{abs(y):g}°{'N' if y >= 0 else 'S'}")
    )
    ax.tick_params(axis="both", labelsize=TICK_PT, width=TICK_WIDTH,
                   length=TICK_LENGTH, pad=5)
    ax.grid(True, linewidth=GRID_LINEWIDTH, alpha=GRID_ALPHA, zorder=2)
    ax.set_xlabel("Longitude", fontsize=AXIS_LABEL_PT, labelpad=7)
    if show_ylabel:
        ax.set_ylabel("Latitude", fontsize=AXIS_LABEL_PT, labelpad=3)
    else:
        ax.set_ylabel("")
        ax.tick_params(axis="y", which="both", left=False, labelleft=False)
