import netCDF4 as nc
from pyproj import Proj
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import FormatStrFormatter

# ============================================================
# 出版图设置
# ============================================================
FIG_WIDTH_IN = 10.0
FIG_HEIGHT_IN = 4.61
FONT_SIZE = 22
TITLE_SIZE = 16
AXIS_LABEL_SIZE = 16
TICK_LABEL_SIZE = 16
CBAR_LABEL_SIZE = 16
CBAR_TICK_SIZE = 16

OUT = os.environ.get("OUT", os.path.dirname(os.path.abspath(__file__)))
MIN_LON, MAX_LON = 6.018, 14.956
MIN_LAT, MAX_LAT = 47.031, 53.965
xticks = [7.0, 10.5, 14.0]
yticks = [47.5, 50.5, 53.5]

matplotlib.rcParams.update({
    "font.family": "DejaVu Serif",
    "font.size": FONT_SIZE,
    "mathtext.fontset": "dejavusans",
})

Lon, Lat = np.meshgrid(
    np.linspace(MIN_LON, MAX_LON, 192),
    np.linspace(MAX_LAT, MIN_LAT, 144),
)


def setup_ax(ax, title, row, col, nrows, ncols):
    ax.set_title(title, fontsize=TITLE_SIZE, pad=3)
    ax.set_xlim(MIN_LON, MAX_LON)
    ax.set_ylim(MIN_LAT, MAX_LAT)
    ax.set_xticks(xticks)
    ax.set_yticks(yticks)
    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
    if col != 0:
        ax.set_yticklabels([])
    if row != nrows - 1:
        ax.set_xticklabels([])
    else:
        ax.set_xlabel("Longitude", fontsize=AXIS_LABEL_SIZE)
    if col == 0:
        ax.set_ylabel("Latitude", fontsize=AXIS_LABEL_SIZE)
    ax.tick_params(labelsize=TICK_LABEL_SIZE)


# ============================================================
# 加载数据
# ============================================================
print("Loading data...")

# --- CMAQ 36km 原始 ---
f = nc.Dataset("<PATH_TO_DATA>", "r")
lcc = Proj(proj="lcc", lat_1=40.0, lat_2=53.0, lat_0=46.5, lon_0=12.0,
           x_0=0, y_0=0, a=6370000.0, b=6370000.0)
x_c = -2106000.0 + 18000 + np.arange(117) * 36000
y_c = -1580293.5 + 18000 + np.arange(97) * 36000
Xc, Yc = np.meshgrid(x_c, y_c)
cmaq_lon, cmaq_lat = lcc(Xc, Yc, inverse=True)
cmaq_lon = np.where(cmaq_lon > 180, cmaq_lon - 360, cmaq_lon)

# --- HR / Bilinear / FLAML ---
fd = np.load("<PATH_TO_DATA>")

# --- LR-interp (CMAQ 36km raw → 4km) ---
lr_interp = np.load("<PATH_TO_DATA>")["pred_maps"]

# --- CorrDiff ---
CORRDIFF_DIR = os.environ.get(
    "CORRDIFF_DIR", "<PATH_TO_DATA>"
)
cfiles_npz = sorted(glob.glob(os.path.join(CORRDIFF_DIR, "day_*.npz")))
cfiles_npy = sorted(
    glob.glob(os.path.join(CORRDIFF_DIR, "*_final.npy")),
    key=lambda x: int(os.path.basename(x).split("_")[0]),
)
if cfiles_npz:
    cfiles = cfiles_npz
    corrdiff_format = "npz"
elif cfiles_npy:
    cfiles = cfiles_npy
    corrdiff_format = "npy"
else:
    raise FileNotFoundError(f"No CorrDiff daily files found in {CORRDIFF_DIR}")

# --- Diffusion (Dmean) ---
SRC_D = "<PATH_TO_DATA>"
dfiles = sorted(
    [x for x in os.listdir(SRC_D) if "final" in x],
    key=lambda x: int(x.split("_")[0]),
)

# --- Reg ---
SRC_R = "<PATH_TO_DATA>"
rfiles_reg = sorted(
    [x for x in os.listdir(SRC_R) if "_reg.npy" in x],
    key=lambda x: int(x.split("_")[0]),
)
reg_all = np.stack([
    np.load(os.path.join(SRC_R, x)).astype(np.float32) for x in rfiles_reg
])

# --- RF ---
rfh = np.load("<PATH_TO_DATA>")
rf_raw = rfh["pred_maps"].astype(np.float32)
nh_rf = (rf_raw.shape[0] // 24) * 24
rf_daily = rf_raw[:nh_rf].reshape(-1, 24, 144, 192).mean(axis=1).astype(np.float32)


def load_corrdiff_daily(path):
    """Load one CorrDiff day and return its ensemble-mean daily field."""
    if corrdiff_format == "npz":
        return np.load(path)["final"][0].astype(np.float32).mean(axis=0)
    # Replacement arrays use the opposite latitude row order.
    return np.flipud(np.load(path).astype(np.float32).mean(axis=(0, 1)))


def load_month(day_slice, cmaq_h0, cmaq_h1):
    """加载一个月的数据，返回 dict"""
    data = {}

    # CMAQ
    data["cmaq"] = f.variables["PM25"][cmaq_h0:cmaq_h1, 0, :, :].mean(axis=0)

    # HR / LR-interp (CMAQ 36km→4km) / FLAML
    data["hr"] = fd["true_maps"][day_slice].mean(0).astype(np.float32)
    data["lr"] = lr_interp[day_slice].mean(0).astype(np.float32)
    # FLAML needs flipud
    data["flaml"] = np.flipud(fd["pred_maps"][day_slice].mean(0).astype(np.float32))

    # CorrDiff: daily ensemble mean
    data["corrdiff"] = np.stack([
        load_corrdiff_daily(x)
        for x in cfiles[day_slice]
    ]).mean(axis=0)

    # Diffusion (flipud + ensemble mean)
    data["diffusion"] = np.stack([
        np.flipud(
            np.load(os.path.join(SRC_D, x)).astype(np.float32).mean(axis=0)
        )
        for x in dfiles[day_slice]
    ]).mean(axis=0)

    # Reg
    data["reg"] = reg_all[day_slice].mean(axis=0)

    # RF
    data["rf"] = rf_daily[day_slice].mean(axis=0)

    return data


# 时间切片: Jan 2-31 (30天), Jul 1-31 (31天)
jan = load_month(slice(1, 31), 0, 30 * 24)
jul = load_month(slice(181, 212), 180 * 24, 211 * 24)
f.close()

# ============================================================
# 出图
# ============================================================

# 4×2 全景布局
LAYOUT_4X2 = [
    [("CMAQ (36 km)", "cmaq", True),  ("HR (4 km)", "hr", False),
     ("CorrDiff", "corrdiff", False), ("Diffusion", "diffusion", False)],
    [("Reg", "reg", False),          ("FLAML", "flaml", False),
     ("RF", "rf", False),            ("LR-interp", "lr", False)],
]

# 1×2 分面组
GROUPS_1X2 = [
    ("CMAQ_HR",     [("CMAQ (36 km)", "cmaq", True),  ("HR (4 km)", "hr", False)]),
    ("Reg_FLAML",   [("Reg", "reg", False),           ("FLAML", "flaml", False)]),
    ("RF_LR",       [("RF", "rf", False),             ("LR-interp", "lr", False)]),
    ("CorrDiff_Diffusion", [("CorrDiff", "corrdiff", False),
                             ("Diffusion", "diffusion", False)]),
]

for mo_name, mo_data in [("Jan", jan), ("Jul", jul)]:
    if mo_name == "Jan":
        vmax = 30
        cbar_ticks = [0, 10, 20, 30]
    else:
        vmax = 15
        cbar_ticks = [0, 5, 10, 15]
    cmap = plt.get_cmap("turbo")
    norm = Normalize(vmin=0, vmax=vmax)
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    # --- 4×2 全景 ---
    fig, axes = plt.subplots(
        2, 4,
        figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN),
        constrained_layout=True,
    )
    for r, row_data in enumerate(LAYOUT_4X2):
        for c, (title, key, is_cmaq) in enumerate(row_data):
            ax = axes[r, c]
            data = mo_data[key]
            if is_cmaq:
                ax.pcolormesh(cmaq_lon, cmaq_lat, np.maximum(data, 0),
                              cmap=cmap, norm=norm, shading="auto", rasterized=True)
            else:
                ax.pcolormesh(Lon, Lat, np.maximum(data, 0),
                              cmap=cmap, norm=norm, shading="auto", rasterized=True)
            setup_ax(ax, title, r, c, 2, 4)
    cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.012, pad=0.01)
    cbar.set_label(r"PM$_{2.5}$ ($\mu g/m^3$)", fontsize=CBAR_LABEL_SIZE)
    cbar.set_ticks(cbar_ticks)
    cbar.ax.tick_params(labelsize=CBAR_TICK_SIZE)
    fig.savefig(f"{OUT}/figure_spatial_{mo_name}_4x2.png", dpi=300, bbox_inches="tight")
    fig.savefig(f"{OUT}/figure_spatial_{mo_name}_4x2.pdf", bbox_inches="tight")
    plt.close(fig)

    # --- 1×2 分面 ---
    for gname, items in GROUPS_1X2:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5.8), constrained_layout=True)
        for ax, (title, key, is_cmaq) in zip(axes, items):
            data = mo_data[key]
            if is_cmaq:
                ax.pcolormesh(cmaq_lon, cmaq_lat, np.maximum(data, 0),
                              cmap=cmap, norm=norm, shading="auto", rasterized=True)
            else:
                ax.pcolormesh(Lon, Lat, np.maximum(data, 0),
                              cmap=cmap, norm=norm, shading="auto", rasterized=True)
            setup_ax(ax, title, 0, 0 if ax == axes[0] else 1, 1, 2)
        cbar = fig.colorbar(sm, ax=axes, orientation="vertical", fraction=0.03, pad=0.02)
        cbar.set_label(r"PM$_{2.5}$ ($\mu g/m^3$)", fontsize=18)
        cbar.ax.tick_params(labelsize=14)
        fig.savefig(f"{OUT}/{gname}_{mo_name}.png", dpi=300, bbox_inches="tight")
        plt.close(fig)

    print(f"{mo_name}: 5 figures saved")

print("Done.")
