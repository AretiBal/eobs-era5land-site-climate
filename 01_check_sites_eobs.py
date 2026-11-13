"""
01_check_sites_eobs.py  --  CHECK ONLY, NO EXTRACTION
- Reads the site list and runs sanity checks.
- Opens E-OBS rr and tg and checks period, units and grid.
- Finds, for each site, the nearest E-OBS grid cell that HAS data.
- Saves diagnostic plots to figures/.
- Writes output/sites_eobs_cells.csv (site -> grid cell table). No extraction.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib
matplotlib.use("Agg")  # save plots as PNG files, no pop-up windows
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.spatial import cKDTree

# ----------------------------------------------------------------- settings
BASE_DIR = Path(__file__).resolve().parent
SITES_XLSX = BASE_DIR / "data" / "high_status_wbs.xlsx"
EOBS_FILES = {
    "rr": BASE_DIR / "data" / "rr_ens_mean_0.1deg_reg_v34.0e.nc",
    "tg": BASE_DIR / "data" / "tg_ens_mean_0.1deg_reg_v34.0e.nc",
}
FIG_DIR = BASE_DIR / "figures"
OUT_DIR = BASE_DIR / "output"

START_DATE, END_DATE = "2017-01-01", "2026-06-30"
REGION_ORDER = ["East", "North", "South", "West"]
REGION_COLORS = {"East": "tab:red", "North": "tab:blue",
                 "South": "tab:orange", "West": "tab:green"}
MAX_DIST_KM = 30           # warn if the cell used is further away than this
MIN_VALID_FRACTION = 0.95  # cell "has data" if valid on >=95% of the test days
EARTH_RADIUS_KM = 6371.0


def header(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


def to_xyz(lat, lon):
    """lat/lon in degrees -> points on a unit sphere (correct distances)."""
    lat, lon = np.radians(lat), np.radians(lon)
    return np.column_stack([np.cos(lat) * np.cos(lon),
                            np.cos(lat) * np.sin(lon),
                            np.sin(lat)])


def chord_to_km(chord):
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.clip(chord / 2, 0, 1))


# ----------------------------------------------------------------- 1. sites
header("1. SITES (Excel)")
sites = pd.read_excel(SITES_XLSX)
print("File:", SITES_XLSX)
print("Rows:", len(sites))
print("Columns:", list(sites.columns))

required = ["wb_id", "lat", "lon", "country", "countryName", "region"]
missing_cols = [c for c in required if c not in sites.columns]
assert not missing_cols, f"Missing columns: {missing_cols}"

n_missing_xy = int(sites[["lat", "lon"]].isna().any(axis=1).sum())
n_dup_id = int(sites["wb_id"].duplicated().sum())
unknown_regions = set(sites["region"].dropna()) - set(REGION_ORDER)

print("\nSites per region (extraction order):")
print(sites["region"].value_counts().reindex(REGION_ORDER).to_string())
print("Unexpected region labels:", unknown_regions or "none")
print("Sites with missing lat/lon:", n_missing_xy)
print("Duplicated wb_id:", n_dup_id)
print("Duplicated lat/lon pairs:", int(sites.duplicated(["lat", "lon"]).sum()))
print(f"Lat range: {sites.lat.min():.3f} to {sites.lat.max():.3f}")
print(f"Lon range: {sites.lon.min():.3f} to {sites.lon.max():.3f}")
print("\nSites per region and country:")
print(sites.groupby(["region", "countryName"]).size().to_string())

assert n_missing_xy == 0, "Some sites have no coordinates"
assert n_dup_id == 0, "Some wb_id values are duplicated"
assert not unknown_regions, f"Unknown region labels: {unknown_regions}"

# ----------------------------------------------------------------- 2. E-OBS
header("2. E-OBS FILES")
ds = {}
expected_days = len(pd.date_range(START_DATE, END_DATE, freq="D"))
for var, path in EOBS_FILES.items():
    d = xr.open_dataset(path)  # lazy: no data loaded yet
    ds[var] = d
    t0, t1 = pd.Timestamp(d.time.values[0]), pd.Timestamp(d.time.values[-1])
    n_days = d.sel(time=slice(START_DATE, END_DATE)).sizes["time"]
    print(f"\n[{var}] {path.name}")
    print(f"  long_name: {d[var].attrs.get('long_name')}, "
          f"units: {d[var].attrs.get('units')}")
    print(f"  full period: {t0.date()} to {t1.date()}")
    print(f"  days in {START_DATE}..{END_DATE}: {n_days} (expected {expected_days})")
    print(f"  grid: {d.sizes['latitude']} lat x {d.sizes['longitude']} lon, "
          f"lat {float(d.latitude.min()):.2f}..{float(d.latitude.max()):.2f}, "
          f"lon {float(d.longitude.min()):.2f}..{float(d.longitude.max()):.2f}")
    assert t0 <= pd.Timestamp(START_DATE) and t1 >= pd.Timestamp(END_DATE), \
        f"{var}: requested period not fully covered"
    assert n_days == expected_days, f"{var}: missing days in the period"

GRID_TOL_DEG = 1e-4  # ~10 m; anything below this is rounding noise
dlat = float(np.abs(ds["rr"].latitude.values - ds["tg"].latitude.values).max())
dlon = float(np.abs(ds["rr"].longitude.values - ds["tg"].longitude.values).max())
print(f"\nMax grid difference rr vs tg: lat {dlat:.2e} deg, lon {dlon:.2e} deg")
assert dlat < GRID_TOL_DEG and dlon < GRID_TOL_DEG, "rr and tg grids really differ"
print("rr and tg on the same grid (within rounding): True")
# give tg exactly the rr coordinates so xarray never misaligns them
ds["tg"] = ds["tg"].assign_coords(latitude=ds["rr"].latitude,
                                  longitude=ds["rr"].longitude)

# ----------------------------------------------------------------- 3. cells with data
header("3. GRID CELLS WITH DATA")
test_days = pd.date_range(START_DATE, END_DATE, freq="MS")  # 1st of each month
print(f"Reading {len(test_days)} test days (1st of each month) ...")
valid = None
for var in ["rr", "tg"]:
    sample = ds[var][var].sel(time=test_days).load()
    v = sample.notnull().mean("time") >= MIN_VALID_FRACTION
    print(f"  {var}: cells with data = {int(v.sum())} of {v.size}")
    valid = v if valid is None else (valid & v)
print(f"  cells with data in BOTH rr and tg = {int(valid.sum())}")

# ----------------------------------------------------------------- 4. nearest cell with data
header("4. NEAREST GRID CELL WITH DATA FOR EACH SITE")
lat2d, lon2d = np.meshgrid(valid.latitude.values, valid.longitude.values,
                           indexing="ij")
vmask = valid.values
site_xyz = to_xyz(sites["lat"].values, sites["lon"].values)

_, idx_any = cKDTree(to_xyz(lat2d.ravel(), lon2d.ravel())).query(site_xyz)
chord, idx_valid = cKDTree(to_xyz(lat2d[vmask], lon2d[vmask])).query(site_xyz)
ilat2d, ilon2d = np.meshgrid(np.arange(lat2d.shape[0]),
                             np.arange(lat2d.shape[1]), indexing="ij")
sites["eobs_ilat"] = ilat2d[vmask][idx_valid]   # row index in the E-OBS grid
sites["eobs_ilon"] = ilon2d[vmask][idx_valid]   # column index in the E-OBS grid

sites["eobs_lat"] = lat2d[vmask][idx_valid].round(2)
sites["eobs_lon"] = lon2d[vmask][idx_valid].round(2)
sites["dist_km"] = chord_to_km(chord).round(2)
sites["moved"] = ~vmask.ravel()[idx_any]  # nearest cell had no data

summary = sites.groupby("region").agg(
    n_sites=("wb_id", "size"),
    n_unique_cells=("eobs_lat", lambda s: len(
        sites.loc[s.index, ["eobs_lat", "eobs_lon"]].drop_duplicates())),
    n_moved=("moved", "sum"),
    mean_dist_km=("dist_km", "mean"),
    max_dist_km=("dist_km", "max"),
).reindex(REGION_ORDER)
print(summary.round(2).to_string())

print(f"\nSites moved to another cell (nearest one had no data): "
      f"{int(sites['moved'].sum())}")
if sites["moved"].any():
    print(sites.loc[sites["moved"], ["wb_id", "region", "countryName", "lat",
                                     "lon", "eobs_lat", "eobs_lon",
                                     "dist_km"]].to_string(index=False))

too_far = sites[sites["dist_km"] > MAX_DIST_KM]
print(f"\nWARNING - sites further than {MAX_DIST_KM} km from their cell: "
      f"{len(too_far)}")
if len(too_far):
    print(too_far[["wb_id", "region", "countryName", "lat", "lon",
                   "dist_km"]].to_string(index=False))

# ----------------------------------------------------------------- 4b. save table
header("4b. SAVE SITE -> E-OBS CELL TABLE")
OUT_DIR.mkdir(exist_ok=True)
cells_file = OUT_DIR / "sites_eobs_cells.csv"
cols = ["wb_id", "wb_name", "region", "country", "countryName", "lat", "lon",
        "eobs_lat", "eobs_lon", "eobs_ilat", "eobs_ilon", "dist_km", "moved"]
table = sites[cols].copy()
table["region"] = pd.Categorical(table["region"], REGION_ORDER, ordered=True)
table = table.sort_values(["region", "wb_id"])
table.to_csv(cells_file, index=False)
print(f"Saved: {cells_file} ({len(table)} rows)")

# ----------------------------------------------------------------- 5. plots
header("5. PLOTS")
FIG_DIR.mkdir(exist_ok=True)

# 5a. map of sites on the E-OBS grid
fig, ax = plt.subplots(figsize=(10, 10))
pad = 2
valid_crop = valid.sel(latitude=slice(sites.lat.min() - pad, sites.lat.max() + pad),
                       longitude=slice(sites.lon.min() - pad, sites.lon.max() + pad))
valid_crop.astype(int).plot(ax=ax, cmap=ListedColormap(["white", "lightgrey"]),
                            add_colorbar=False)
for region in REGION_ORDER:
    s = sites[sites["region"] == region]
    ax.scatter(s["lon"], s["lat"], s=8, color=REGION_COLORS[region],
               label=f"{region} (n={len(s)})")
m = sites[sites["moved"]]
ax.scatter(m["lon"], m["lat"], s=40, facecolors="none", edgecolors="k",
           label=f"moved to nearest cell with data (n={len(m)})")
ax.set_title("Sites on the E-OBS grid (grey = cells with rr and tg data)")
ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1))  # outside the map
f1 = FIG_DIR / "01_sites_on_eobs_grid.png"
fig.savefig(f1, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved:", f1)

# 5b. distance from site to the cell used
fig, ax = plt.subplots(figsize=(8, 4))
for region in REGION_ORDER:
    ax.hist(sites.loc[sites["region"] == region, "dist_km"], bins=30,
            alpha=0.5, color=REGION_COLORS[region], label=region)
ax.axvline(MAX_DIST_KM, color="k", ls="--", label=f"{MAX_DIST_KM} km limit")
ax.set_xlabel("distance from site to E-OBS cell centre (km)")
ax.set_ylabel("number of sites")
ax.legend()
f2 = FIG_DIR / "01_distance_to_eobs_cell.png"
fig.savefig(f2, dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved:", f2)

header("DONE - check finished, nothing extracted")
