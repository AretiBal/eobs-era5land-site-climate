"""
02_extract_eobs.py -- EXTRACT E-OBS rr and tg for every site
- Reads output/sites_eobs_cells.csv (made by 01_check_sites_eobs.py).
- Loops region (East, North, South, West) -> year (2017..2026).
- One chunk per region-year: output/eobs_chunks/eobs_{region}_{year}.parquet
- Existing chunks are skipped, so the script can be stopped and restarted.
"""
from pathlib import Path
import time
import numpy as np
import pandas as pd
import xarray as xr

# ----------------------------------------------------------------- settings
BASE_DIR = Path(__file__).resolve().parent
CELLS_CSV = BASE_DIR / "output" / "sites_eobs_cells.csv"
EOBS_FILES = {
    "rr": BASE_DIR / "data" / "rr_ens_mean_0.1deg_reg_v34.0e.nc",
    "tg": BASE_DIR / "data" / "tg_ens_mean_0.1deg_reg_v34.0e.nc",
}
CHUNK_DIR = BASE_DIR / "output" / "eobs_chunks"

START_DATE, END_DATE = "2017-01-01", "2026-06-30"
REGION_ORDER = ["East", "North", "South", "West"]
YEARS = range(2017, 2027)

# TEST MODE: list of (region, year) to run only these chunks; None = run all
ONLY_CHUNKS = None

GRID_TOL_DEG = 1e-4
RR_RANGE = (0, 500)   # plausible daily rainfall, mm
TG_RANGE = (-60, 50)  # plausible daily mean temperature, °C
META_COLS = ["wb_id", "region", "country", "countryName", "lat", "lon",
             "eobs_lat", "eobs_lon", "dist_km", "moved"]


def header(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# ----------------------------------------------------------------- 1. inputs
header("1. INPUTS")
cells = pd.read_csv(CELLS_CSV)
print(f"Site table: {CELLS_CSV} ({len(cells)} sites)")
print(cells["region"].value_counts().reindex(REGION_ORDER).to_string())
assert cells["wb_id"].is_unique, "duplicated wb_id in site table"

ds_rr = xr.open_dataset(EOBS_FILES["rr"])
ds_tg = xr.open_dataset(EOBS_FILES["tg"])
dlat = float(np.abs(ds_rr.latitude.values - ds_tg.latitude.values).max())
dlon = float(np.abs(ds_rr.longitude.values - ds_tg.longitude.values).max())
assert dlat < GRID_TOL_DEG and dlon < GRID_TOL_DEG, "rr and tg grids differ"
rr = ds_rr["rr"]
tg = ds_tg["tg"].assign_coords(latitude=ds_rr.latitude,
                               longitude=ds_rr.longitude)

# the saved grid indices must point to the saved cell coordinates
ok = (np.allclose(rr.latitude.values[cells["eobs_ilat"]], cells["eobs_lat"], atol=0.006)
      and np.allclose(rr.longitude.values[cells["eobs_ilon"]], cells["eobs_lon"], atol=0.006))
print("Grid indices match cell coordinates:", ok)
assert ok

CHUNK_DIR.mkdir(parents=True, exist_ok=True)


# ----------------------------------------------------------------- functions
def extract_chunk(region_cells, year):
    """Return a long table (one row per site and day) for one region-year."""
    t0 = max(pd.Timestamp(f"{year}-01-01"), pd.Timestamp(START_DATE))
    t1 = min(pd.Timestamp(f"{year}-12-31"), pd.Timestamp(END_DATE))
    dates = pd.date_range(t0, t1, freq="D")

    uc = region_cells[["eobs_ilat", "eobs_ilon"]].drop_duplicates().reset_index(drop=True)
    i0, i1 = uc["eobs_ilat"].min(), uc["eobs_ilat"].max() + 1
    j0, j1 = uc["eobs_ilon"].min(), uc["eobs_ilon"].max() + 1

    values = {}
    for name, da in [("rr", rr), ("tg", tg)]:
        sub = da.sel(time=slice(t0, t1)).isel(latitude=slice(i0, i1),
                                               longitude=slice(j0, j1))
        assert np.array_equal(sub.time.values, dates.values), f"{name}: dates differ"
        box = sub.values                                   # (time, lat, lon) in memory
        values[name] = box[:, uc["eobs_ilat"] - i0, uc["eobs_ilon"] - j0]  # (time, cell)

    n_t, n_c = len(dates), len(uc)
    cell_df = pd.DataFrame({
        "eobs_ilat": np.tile(uc["eobs_ilat"].values, n_t),
        "eobs_ilon": np.tile(uc["eobs_ilon"].values, n_t),
        "date": np.repeat(dates.values, n_c),
        "rr": values["rr"].ravel(),
        "tg": values["tg"].ravel(),
    })
    df = (region_cells[META_COLS + ["eobs_ilat", "eobs_ilon"]]
          .merge(cell_df, on=["eobs_ilat", "eobs_ilon"], how="left")
          .drop(columns=["eobs_ilat", "eobs_ilon"])
          .sort_values(["wb_id", "date"])
          .reset_index(drop=True))
    return df, dates


def check_chunk(df, n_sites, dates):
    expected = n_sites * len(dates)
    print(f"   rows: {len(df):,} (expected {n_sites} sites x {len(dates)} days = {expected:,})")
    assert len(df) == expected, "wrong number of rows"
    assert not df.duplicated(["wb_id", "date"]).any(), "duplicated site-days"
    for v, (lo, hi) in [("rr", RR_RANGE), ("tg", TG_RANGE)]:
        n_nan = int(df[v].isna().sum())
        print(f"   {v}: NaN {n_nan} ({100 * n_nan / len(df):.2f}%), "
              f"min {df[v].min():.1f}, mean {df[v].mean():.1f}, max {df[v].max():.1f}")
        n_out = int(((df[v] < lo) | (df[v] > hi)).sum())
        if n_out:
            print(f"   WARNING: {n_out} {v} values outside {lo}..{hi}")
        if n_nan:
            bad = df.loc[df[v].isna()].groupby("wb_id").size().sort_values(ascending=False)
            print(f"   WARNING: {len(bad)} sites have NaN {v} days, e.g.:")
            print(bad.head(10).to_string())


# ----------------------------------------------------------------- 2. extraction
header("2. EXTRACTION")
todo = [(r, y) for r in REGION_ORDER for y in YEARS
        if ONLY_CHUNKS is None or (r, y) in ONLY_CHUNKS]
print(f"Chunks to process: {len(todo)}" + ("  (TEST MODE)" if ONLY_CHUNKS else ""))

for k, (region, year) in enumerate(todo, 1):
    f = CHUNK_DIR / f"eobs_{region}_{year}.parquet"
    if f.exists():
        print(f"\n[{k}/{len(todo)}] {region} {year}: already exists, skipped")
        continue
    start = time.time()
    region_cells = cells[cells["region"] == region]
    n_cells = len(region_cells[["eobs_ilat", "eobs_ilon"]].drop_duplicates())
    print(f"\n[{k}/{len(todo)}] {region} {year}: {len(region_cells)} sites, {n_cells} cells")

    df, dates = extract_chunk(region_cells, year)
    check_chunk(df, len(region_cells), dates)

    tmp = f.with_name(f.name + ".tmp")
    df.to_parquet(tmp, index=False)
    tmp.replace(f)  # rename only when fully written
    print(f"   saved {f.name} ({f.stat().st_size / 1e6:.1f} MB) in {time.time() - start:.0f} s")

# ----------------------------------------------------------------- summary
header("DONE")
n_found = sum((CHUNK_DIR / f"eobs_{r}_{y}.parquet").exists()
              for r in REGION_ORDER for y in YEARS)
print(f"Chunks present in {CHUNK_DIR}: {n_found} of {len(REGION_ORDER) * len(YEARS)}")
