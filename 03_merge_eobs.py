"""
03_merge_eobs.py -- MERGE the E-OBS chunks, CHECK, SAVE and PLOT
- Reads the 40 chunks in output/eobs_chunks/.
- Checks completeness (all sites x all days, no duplicates, NaN report).
- Saves output/eobs_daily_2015_2024.parquet and .csv.
- Saves plots to figures/.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- settings
BASE_DIR = Path(__file__).resolve().parent
CHUNK_DIR = BASE_DIR / "output" / "eobs_chunks"
CELLS_CSV = BASE_DIR / "output" / "sites_eobs_cells.csv"
OUT_PARQUET = BASE_DIR / "output" / "eobs_daily_2015_2024.parquet"
OUT_CSV = BASE_DIR / "output" / "eobs_daily_2015_2024.csv"
FIG_DIR = BASE_DIR / "figures"

START_DATE, END_DATE = "2015-01-01", "2024-12-30"
LAST_FULL_YEAR = 2024          # for annual rainfall totals (2026 is half a year)
REGION_ORDER = ["East", "North", "South", "West"]
YEARS = range(2015, 2025)
REGION_COLORS = {"East": "tab:red", "North": "tab:blue",
                 "South": "tab:orange", "West": "tab:green"}


def header(text):
    print("\n" + "=" * 70 + f"\n{text}\n" + "=" * 70)


# ----------------------------------------------------------------- 1. read chunks
header("1. READ CHUNKS")
files = [CHUNK_DIR / f"eobs_{r}_{y}.parquet" for r in REGION_ORDER for y in YEARS]
missing = [f.name for f in files if not f.exists()]
print(f"Chunks expected: {len(files)}, found: {len(files) - len(missing)}")
assert not missing, f"Missing chunks: {missing}"
df = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"Merged rows: {len(df):,}")

# ----------------------------------------------------------------- 2. checks
header("2. CHECKS")
cells = pd.read_csv(CELLS_CSV)
n_days = len(pd.date_range(START_DATE, END_DATE, freq="D"))
expected = len(cells) * n_days
print(f"Rows: {len(df):,} (expected {len(cells)} sites x {n_days} days = {expected:,})")
assert len(df) == expected

n_dup = int(df.duplicated(["wb_id", "date"]).sum())
print("Duplicated site-days:", n_dup)
assert n_dup == 0

days_per_site = df.groupby("wb_id").size()
print("Sites in data:", len(days_per_site), "| all sites from table present:",
      set(days_per_site.index) == set(cells["wb_id"]))
print("Days per site: min", days_per_site.min(), "max", days_per_site.max())
assert set(days_per_site.index) == set(cells["wb_id"])
assert (days_per_site == n_days).all()
print("Date range:", df["date"].min().date(), "to", df["date"].max().date())

for v in ["rr", "tg"]:
    nan_rows = df[df[v].isna()]
    print(f"\nMissing {v}: {len(nan_rows)} values")
    if len(nan_rows):
        print(nan_rows[["wb_id", "region", "countryName", "date"]].to_string(index=False))

# E-OBS values have at most 2 decimals; round so printouts, parquet and CSV agree
df[["rr", "tg"]] = df[["rr", "tg"]].astype("float64").round(2)

print("\nSummary per region:")
summary = df.groupby("region").agg(
    n_sites=("wb_id", "nunique"),
    tg_mean=("tg", "mean"), tg_min=("tg", "min"), tg_max=("tg", "max"),
    rr_mean_daily=("rr", "mean"), rr_max=("rr", "max"),
).reindex(REGION_ORDER)
print(summary.round(2).to_string())

# ----------------------------------------------------------------- 3. save
header("3. SAVE")
df["region"] = pd.Categorical(df["region"], REGION_ORDER, ordered=True)
df = df.sort_values(["region", "wb_id", "date"]).reset_index(drop=True)
df["region"] = df["region"].astype(str)

tmp = OUT_PARQUET.with_name(OUT_PARQUET.name + ".tmp")
df.to_parquet(tmp, index=False)
tmp.replace(OUT_PARQUET)
print(f"Saved: {OUT_PARQUET} ({OUT_PARQUET.stat().st_size / 1e6:.1f} MB)")

csv_df = df.copy()
csv_df["date"] = csv_df["date"].dt.strftime("%Y-%m-%d")
tmp = OUT_CSV.with_name(OUT_CSV.name + ".tmp")
csv_df.to_csv(tmp, index=False, na_rep="NA")   # NA = missing, read natively by R
tmp.replace(OUT_CSV)
print(f"Saved: {OUT_CSV} ({OUT_CSV.stat().st_size / 1e6:.1f} MB)")
del csv_df

# ----------------------------------------------------------------- 4. plots
header("4. PLOTS")
FIG_DIR.mkdir(exist_ok=True)
df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month

# 4a. example daily time series: the site nearest the centre of each region
examples = []
for region in REGION_ORDER:
    c = cells[cells["region"] == region]
    d2 = (c["lat"] - c["lat"].median()) ** 2 + (c["lon"] - c["lon"].median()) ** 2
    examples.append(c.loc[d2.idxmin()])

fig, axes = plt.subplots(4, 2, figsize=(15, 12), sharex=True)
for i, s in enumerate(examples):
    ts = df[df["wb_id"] == s["wb_id"]]
    col = REGION_COLORS[s["region"]]
    axes[i, 0].plot(ts["date"], ts["tg"], lw=0.5, color=col)
    axes[i, 0].set_ylabel("tg (°C)")
    axes[i, 0].set_title(f"{s['region']}: {s['wb_id']} ({s['countryName']})",
                         loc="left", fontsize=9)
    axes[i, 1].bar(ts["date"], ts["rr"], width=1, color=col)
    axes[i, 1].set_ylabel("rr (mm/day)")
fig.suptitle("E-OBS daily mean temperature (left) and rainfall (right), "
             "one example site per region")
fig.tight_layout()
f = FIG_DIR / "03_example_timeseries.png"
fig.savefig(f, dpi=150)
plt.close(fig)
print("Saved:", f)

# 4b. seasonal cycle per region
tg_month = df.groupby(["region", "month"])["tg"].mean().unstack(0)
rr_month = (df.groupby(["region", "wb_id", "year", "month"])["rr"].sum(min_count=1)
              .groupby(["region", "month"]).mean().unstack(0))
fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
for region in REGION_ORDER:
    axes[0].plot(tg_month.index, tg_month[region], marker="o",
                 color=REGION_COLORS[region], label=region)
    axes[1].plot(rr_month.index, rr_month[region], marker="o",
                 color=REGION_COLORS[region], label=region)
axes[0].set_ylabel("mean tg (°C)")
axes[1].set_ylabel("mean monthly total rr (mm)")
for ax in axes:
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("month")
    ax.legend()
fig.suptitle("Seasonal cycle per region, average over all sites, 2015-2024")
fig.tight_layout()
f = FIG_DIR / "03_seasonal_cycle_by_region.png"
fig.savefig(f, dpi=150)
plt.close(fig)
print("Saved:", f)

# 4c. map: mean temperature and mean annual rainfall per site
site_tg = df.groupby("wb_id")["tg"].mean().rename("tg_mean")
site_rr = (df[df["year"] <= LAST_FULL_YEAR].groupby(["wb_id", "year"])["rr"].sum()
             .groupby("wb_id").mean().rename("rr_annual"))
m = cells.set_index("wb_id").join(site_tg).join(site_rr)
print(f"Site mean tg: {m['tg_mean'].min():.1f} to {m['tg_mean'].max():.1f} °C")
print(f"Site mean annual rr (2015-{LAST_FULL_YEAR}): "
      f"{m['rr_annual'].min():.0f} to {m['rr_annual'].max():.0f} mm")

fig, axes = plt.subplots(1, 2, figsize=(16, 8))
for ax, col, cmap, label in [
        (axes[0], "tg_mean", "RdYlBu_r", "mean tg 2015-2024 (°C)"),
        (axes[1], "rr_annual", "YlGnBu", f"mean annual rr 2015-{LAST_FULL_YEAR} (mm)")]:
    sc = ax.scatter(m["lon"], m["lat"], c=m[col], cmap=cmap, s=10)
    fig.colorbar(sc, ax=ax, shrink=0.6, label=label)
    ax.set_aspect(1 / np.cos(np.radians(50)))
    ax.set_xlabel("longitude")
    ax.set_ylabel("latitude")
fig.suptitle("E-OBS climate per site")
fig.tight_layout()
f = FIG_DIR / "03_map_mean_tg_annual_rr.png"
fig.savefig(f, dpi=150)
plt.close(fig)
print("Saved:", f)

header("DONE")
