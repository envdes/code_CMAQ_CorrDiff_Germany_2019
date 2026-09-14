import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import geopandas as gpd
from matplotlib.ticker import FixedLocator
from matplotlib.patches import ConnectionPatch

geo1 = "geo_em.d01.nc/geo_em.d01.nc"
geo2 = "geo_em.d02.nc/geo_em.d02.nc"
geo3 = "geo_em.d03.nc/geo_em.d03.nc"

shp_germany = "germany.shp"
shp_states = "germany_States_level.shp"

out_png = "study_area_lambert_style_annotated.png"
out_pdf = "study_area_lambert_style_annotated.pdf"
LATEX_TEXTWIDTH_PT = 390.0
TEX_POINTS_PER_INCH = 72.27
FIG_WIDTH_IN = LATEX_TEXTWIDTH_PT / TEX_POINTS_PER_INCH
FIG_HEIGHT_IN = FIG_WIDTH_IN * 7 / 16
FIG_DPI = 300

FONT_FAMILY = "Liberation Serif"
BASE_FONT_PT = 10
PANEL_TITLE_PT = 12
MAP_LABEL_PT = 10
TICK_PT = 10
DOMAIN_LABEL_PT = 8
TITLE_PAD = 6
AXIS_LINEWIDTH = 0.8
DOMAIN_LINEWIDTH = 1.1
CONNECTION_LINEWIDTH = 1.0

def get_domain_info(path):
    ds = xr.open_dataset(path)

    lat = ds["XLAT_M"].isel(Time=0).values
    lon = ds["XLONG_M"].isel(Time=0).values

    top_lon = lon[0, :]
    top_lat = lat[0, :]

    right_lon = lon[1:, -1]
    right_lat = lat[1:, -1]

    bottom_lon = lon[-1, -2::-1]
    bottom_lat = lat[-1, -2::-1]

    left_lon = lon[-2:0:-1, 0]
    left_lat = lat[-2:0:-1, 0]

    boundary_lon = np.concatenate([top_lon, right_lon, bottom_lon, left_lon, [top_lon[0]]])
    boundary_lat = np.concatenate([top_lat, right_lat, bottom_lat, left_lat, [top_lat[0]]])

    attrs = ds.attrs
    info = {
        "lon2d": lon,
        "lat2d": lat,
        "boundary_lon": boundary_lon,
        "boundary_lat": boundary_lat,
        "lon_min": float(np.min(lon)),
        "lon_max": float(np.max(lon)),
        "lat_min": float(np.min(lat)),
        "lat_max": float(np.max(lat)),
        "lon_c": float(np.mean(lon)),
        "lat_c": float(np.mean(lat)),
        "attrs": attrs,
    }
    ds.close()
    return info

d01 = get_domain_info(geo1)
d02 = get_domain_info(geo2)
d03 = get_domain_info(geo3)

attrs = d01["attrs"]

cen_lat = float(attrs.get("CEN_LAT", d01["lat_c"]))
cen_lon = float(attrs.get("CEN_LON", d01["lon_c"]))
truelat1 = float(attrs.get("TRUELAT1", 30.0))
truelat2 = float(attrs.get("TRUELAT2", 60.0))
stand_lon = float(attrs.get("STAND_LON", cen_lon))

proj_lcc = ccrs.LambertConformal(
    central_longitude=stand_lon,
    central_latitude=cen_lat,
    standard_parallels=(truelat1, truelat2)
)
proj_pc = ccrs.PlateCarree()

gdf = gpd.read_file(shp_germany)
if gdf.crs is not None and str(gdf.crs) != "EPSG:4326":
    gdf = gdf.to_crs("EPSG:4326")

gdf_states = gpd.read_file(shp_states)
if gdf_states.crs is not None and str(gdf_states.crs) != "EPSG:4326":
    gdf_states = gdf_states.to_crs("EPSG:4326")

def draw_domain(ax, domain, color="white", lw=2.0, z=5):
    ax.plot(
        domain["boundary_lon"],
        domain["boundary_lat"],
        transform=proj_pc,
        color=color,
        linewidth=lw,
        zorder=z
    )

def add_label_box(ax, text, x, y, fontsize=DOMAIN_LABEL_PT):
    ax.text(
        x, y, text,
        transform=ax.transAxes,
        fontsize=fontsize,
        ha="left",
        va="center",
        zorder=20,
        bbox=dict(
            facecolor="white",
            edgecolor="0.3",
            boxstyle="round,pad=0.4",
            alpha=0.65
        )
    )

plt.rcParams["font.family"] = FONT_FAMILY
plt.rcParams["font.size"] = BASE_FONT_PT
plt.rcParams["axes.titlesize"] = PANEL_TITLE_PT
plt.rcParams["axes.labelsize"] = MAP_LABEL_PT
plt.rcParams["xtick.labelsize"] = TICK_PT
plt.rcParams["ytick.labelsize"] = TICK_PT

fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN), dpi=FIG_DPI)
gs = fig.add_gridspec(1, 2, width_ratios=[2.35, 1.15], wspace=0.05)

ax1 = fig.add_subplot(gs[0, 0], projection=proj_lcc)
ax1.set_title("(a) CMAQ nested modeling domains", fontsize=PANEL_TITLE_PT,
              pad=TITLE_PAD)

xy_d01 = proj_lcc.transform_points(
    proj_pc,
    d01["boundary_lon"],
    d01["boundary_lat"],
)
x_d01 = xy_d01[:, 0]
y_d01 = xy_d01[:, 1]

pad_x = (x_d01.max() - x_d01.min()) * 0.02
pad_y = (y_d01.max() - y_d01.min()) * 0.02

ax1.set_xlim(x_d01.min() - pad_x, x_d01.max() + pad_x)
ax1.set_ylim(y_d01.min() - pad_y, y_d01.max() + pad_y)

ax1.add_feature(cfeature.OCEAN.with_scale("50m"), facecolor="#5dade2", edgecolor="none", zorder=0)
ax1.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#5a9e0b", edgecolor="none", zorder=1)
ax1.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.6, edgecolor="black", zorder=2)
ax1.add_feature(cfeature.BORDERS.with_scale("50m"), linewidth=0.5, edgecolor="black", zorder=2)

xticks = [-20, 0, 20, 40]
yticks = [30, 35, 40, 45, 50, 55]

gl = ax1.gridlines(
    crs=proj_pc,
    draw_labels=True,
    linewidth=0.5,
    color="k",
    alpha=0.15,
    linestyle="-"
)
gl.top_labels = False
gl.right_labels = False
gl.xlocator = FixedLocator(xticks)
gl.ylocator = FixedLocator(yticks)
gl.xlabel_style = {"size": 12}
gl.ylabel_style = {"size": 12}

draw_domain(ax1, d01, color="white", lw=2.4, z=5)
draw_domain(ax1, d02, color="white", lw=2.2, z=6)
draw_domain(ax1, d03, color="red", lw=2.2, z=7)

for spine in ax1.spines.values():
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

add_label_box(ax1, "domain1", 0.05, 0.94, fontsize=12)
add_label_box(ax1, "domain2", 0.24, 0.79, fontsize=12)
add_label_box(ax1, "domain3", 0.42, 0.64, fontsize=12)

ax2 = fig.add_subplot(gs[0, 1])

gdf.plot(
    ax=ax2,
    facecolor="#f2f2f2",
    edgecolor="none",
    zorder=1
)

gdf_states.boundary.plot(
    ax=ax2,
    color="black",
    linewidth=0.8,
    zorder=2
)

gdf.boundary.plot(
    ax=ax2,
    color="black",
    linewidth=0.8,
    zorder=3
)

ax2.plot(
    d03["boundary_lon"],
    d03["boundary_lat"],
    color="red",
    linewidth=1.8,
    zorder=4
)

minx, miny, maxx, maxy = gdf.total_bounds
padx = (maxx - minx) * 0.10
pady = (maxy - miny) * 0.10

ax2.set_xlim(minx - padx, maxx + padx)
ax2.set_ylim(miny - pady, maxy + pady)

ax2.set_aspect("equal")
ax2.set_xticks([])
ax2.set_yticks([])
ax2.set_xlabel("")
ax2.set_ylabel("")

for spine in ax2.spines.values():
    spine.set_linewidth(1.2)
    spine.set_edgecolor("black")

xy1 = (d03["lon_max"], d03["lat_max"])
xy2 = (d03["lon_max"], d03["lat_min"])

x0, x1 = ax2.get_xlim()
y0, y1 = ax2.get_ylim()

right_target_top = (x0, y0 + 0.80 * (y1 - y0))
right_target_bottom = (x0, y0 + 0.10 * (y1 - y0))

con1 = ConnectionPatch(
    xyA=xy1, coordsA=proj_pc._as_mpl_transform(ax1),
    xyB=right_target_top, coordsB=ax2.transData,
    color="black", linewidth=2.0, zorder=10
)
con2 = ConnectionPatch(
    xyA=xy2, coordsA=proj_pc._as_mpl_transform(ax1),
    xyB=right_target_bottom, coordsB=ax2.transData,
    color="black", linewidth=2.0, zorder=10
)

fig.add_artist(con1)
fig.add_artist(con2)

plt.subplots_adjust(left=0.035, right=0.99, top=0.985, bottom=0.04, wspace=0.05)

plt.savefig(out_png, dpi=500, bbox_inches="tight")
plt.savefig(out_pdf, bbox_inches="tight")
plt.show()