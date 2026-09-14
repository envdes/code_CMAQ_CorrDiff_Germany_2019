import os
import sys
import numpy as np
import netCDF4 as nc
from pyproj import Proj
from pathlib import Path

SCRIPT_DIR = Path(__file__).parent.absolute()
GIT_DRAW_DIR = SCRIPT_DIR.parent.parent
STAT_PERF_DIR = GIT_DRAW_DIR / "Statistical_Performance"
OUTPUT_DIR = GIT_DRAW_DIR / "data"
OUTPUT_FILE = OUTPUT_DIR / "figure3_figure4_spatial_maps.npz"

CMAQ_NC_PATH = "../36km.CMAQ.2019.conc"

MIN_LON, MAX_LON = 6.018, 14.956
MIN_LAT, MAX_LAT = 47.031, 53.965

MODEL_FILES = {
    'truth': '',
    'lr_interp': '',
    'flaml': '',
    'corrdiff': '',
    'diffusion': '',
    'reg': '',
    'rf': '',
}


def load_model_data(name, npz_file):
    path = STAT_PERF_DIR / npz_file
    if not path.exists():
        raise FileNotFoundError(f"Model data not found: {path}")

    with np.load(path) as f:
        data = f['data'].astype(np.float32)

    print(f"  Loaded {name}: {data.shape}, range [{data.min():.2f}, {data.max():.2f}]")
    return data


def compute_monthly_mean(data, day_slice):
    return data[day_slice].mean(axis=0).astype(np.float32)


def cmaq_coordinates():
    projection = Proj(
        proj="lcc",
        lat_1=40.0,
        lat_2=53.0,
        lat_0=46.5,
        lon_0=12.0,
        x_0=0,
        y_0=0,
        a=6370000.0,
        b=6370000.0,
    )
    x = -2106000.0 + 18000 + np.arange(117) * 36000
    y = -1580293.5 + 18000 + np.arange(97) * 36000
    grid_x, grid_y = np.meshgrid(x, y)
    lon, lat = projection(grid_x, grid_y, inverse=True)
    lon = np.where(lon > 180, lon - 360, lon)
    return lon, lat


def load_cmaq_monthly(nc_file, hour_start, hour_end):
    with nc.Dataset(nc_file, "r") as f:
        data = f.variables["PM25"][hour_start:hour_end, 0, :, :].mean(axis=0)
    return np.asarray(data, dtype=np.float32)


def main():
    print("=" * 70)
    print("Figure 3 & 4 Spatial Maps Data Export")
    print("=" * 70)
    print()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading CMAQ 36km data...")
    if not Path(CMAQ_NC_PATH).exists():
        print(f"  WARNING: CMAQ NetCDF not found: {CMAQ_NC_PATH}")
        print(f"  CMAQ panels will be skipped")
        cmaq_lon, cmaq_lat = None, None
        jan_cmaq36, jul_cmaq36 = None, None
    else:
        cmaq_lon, cmaq_lat = cmaq_coordinates()
        jan_cmaq36 = load_cmaq_monthly(CMAQ_NC_PATH, 0, 30 * 24)
        jul_cmaq36 = load_cmaq_monthly(CMAQ_NC_PATH, 180 * 24, 211 * 24)
        print(f"  jan_cmaq36: {jan_cmaq36.shape}, [{jan_cmaq36.min():.2f}, {jan_cmaq36.max():.2f}]")
        print(f"  jul_cmaq36: {jul_cmaq36.shape}, [{jul_cmaq36.min():.2f}, {jul_cmaq36.max():.2f}]")
    print()

    print("Loading model data...")
    models = {}
    for name, npz_file in MODEL_FILES.items():
        models[name] = load_model_data(name, npz_file)

    print()

    jan_slice = slice(1, 31)
    jul_slice = slice(181, 212)

    print(f"Computing January monthly means (days 1-30)...")
    jan_hr = compute_monthly_mean(models['truth'], jan_slice)
    jan_lr_interp = compute_monthly_mean(models['lr_interp'], jan_slice)
    jan_flaml = np.flipud(compute_monthly_mean(models['flaml'], jan_slice))
    jan_corrdiff = compute_monthly_mean(models['corrdiff'], jan_slice)
    jan_diffusion = np.flipud(compute_monthly_mean(models['diffusion'], jan_slice))
    jan_reg = compute_monthly_mean(models['reg'], jan_slice)
    jan_rf = compute_monthly_mean(models['rf'], jan_slice)

    print(f"  jan_hr: {jan_hr.shape}, [{jan_hr.min():.2f}, {jan_hr.max():.2f}]")
    print(f"  jan_lr_interp: {jan_lr_interp.shape}, [{jan_lr_interp.min():.2f}, {jan_lr_interp.max():.2f}]")
    print(f"  jan_flaml: {jan_flaml.shape}, [{jan_flaml.min():.2f}, {jan_flaml.max():.2f}]")
    print(f"  jan_corrdiff: {jan_corrdiff.shape}, [{jan_corrdiff.min():.2f}, {jan_corrdiff.max():.2f}]")
    print(f"  jan_diffusion: {jan_diffusion.shape}, [{jan_diffusion.min():.2f}, {jan_diffusion.max():.2f}]")
    print(f"  jan_reg: {jan_reg.shape}, [{jan_reg.min():.2f}, {jan_reg.max():.2f}]")
    print(f"  jan_rf: {jan_rf.shape}, [{jan_rf.min():.2f}, {jan_rf.max():.2f}]")
    print()

    print(f"Computing July monthly means (days 181-211)...")
    jul_hr = compute_monthly_mean(models['truth'], jul_slice)
    jul_lr_interp = compute_monthly_mean(models['lr_interp'], jul_slice)
    jul_flaml = np.flipud(compute_monthly_mean(models['flaml'], jul_slice))
    jul_corrdiff = compute_monthly_mean(models['corrdiff'], jul_slice)
    jul_diffusion = np.flipud(compute_monthly_mean(models['diffusion'], jul_slice))
    jul_reg = compute_monthly_mean(models['reg'], jul_slice)
    jul_rf = compute_monthly_mean(models['rf'], jul_slice)

    print(f"  jul_hr: {jul_hr.shape}, [{jul_hr.min():.2f}, {jul_hr.max():.2f}]")
    print(f"  jul_lr_interp: {jul_lr_interp.shape}, [{jul_lr_interp.min():.2f}, {jul_lr_interp.max():.2f}]")
    print(f"  jul_flaml: {jul_flaml.shape}, [{jul_flaml.min():.2f}, {jul_flaml.max():.2f}]")
    print(f"  jul_corrdiff: {jul_corrdiff.shape}, [{jul_corrdiff.min():.2f}, {jul_corrdiff.max():.2f}]")
    print(f"  jul_diffusion: {jul_diffusion.shape}, [{jul_diffusion.min():.2f}, {jul_diffusion.max():.2f}]")
    print(f"  jul_reg: {jul_reg.shape}, [{jul_reg.min():.2f}, {jul_reg.max():.2f}]")
    print(f"  jul_rf: {jul_rf.shape}, [{jul_rf.min():.2f}, {jul_rf.max():.2f}]")
    print()

    hr_lon = np.linspace(MIN_LON, MAX_LON, 192)
    hr_lat = np.linspace(MAX_LAT, MIN_LAT, 144)

    print("Saving to compressed NPZ...")
    save_dict = {
        'jan_hr': jan_hr,
        'jan_lr_interp': jan_lr_interp,
        'jan_flaml': jan_flaml,
        'jan_corrdiff': jan_corrdiff,
        'jan_diffusion': jan_diffusion,
        'jan_reg': jan_reg,
        'jan_rf': jan_rf,
        'jul_hr': jul_hr,
        'jul_lr_interp': jul_lr_interp,
        'jul_flaml': jul_flaml,
        'jul_corrdiff': jul_corrdiff,
        'jul_diffusion': jul_diffusion,
        'jul_reg': jul_reg,
        'jul_rf': jul_rf,
        'hr_lon': hr_lon,
        'hr_lat': hr_lat,
    }

    if jan_cmaq36 is not None:
        save_dict['jan_cmaq36'] = jan_cmaq36
        save_dict['jul_cmaq36'] = jul_cmaq36
        save_dict['cmaq36_lon'] = cmaq_lon
        save_dict['cmaq36_lat'] = cmaq_lat

    np.savez_compressed(OUTPUT_FILE, **save_dict)

    file_size_mb = OUTPUT_FILE.stat().st_size / (1024 * 1024)
    print(f"✓ Saved: {OUTPUT_FILE}")
    print(f"  File size: {file_size_mb:.2f} MB")
    print()

    print("NPZ keys:")
    with np.load(OUTPUT_FILE) as f:
        for key in sorted(f.keys()):
            arr = f[key]
            print(f"  {key:20s}: shape {str(arr.shape):15s}, dtype {arr.dtype}")
    print()

    print("=" * 70)
    print("✓ Export completed successfully!")
    print("=" * 70)
    print()

    if jan_cmaq36 is None:
        print("Note: CMAQ 36km data not included (NetCDF not found)")
        print("      CMAQ panels will be skipped in plotting script")
    else:
        print("Note: CMAQ 36km data successfully extracted from 40GB NetCDF")
        print("      All 8 model panels ready for plotting")
    print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
