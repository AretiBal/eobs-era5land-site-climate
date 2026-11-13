"""
04_plot_random_sites.py -- PLOT daily E-OBS series for random example sites
- Reads output/eobs_daily_2017_2026.parquet.
- Picks N_PER_REGION random sites per region (fixed seed = repeatable).
- Saves one figure per region: figures/04_random_sites_{region}.png
"""
from pathlib import Path
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------- settings
BASE_DIR = Path(__file__).resolve().parent
DATA = BASE_DIR / "output" / "eobs_daily_2017_2026.parquet"
FIG_DIR = BASE_DIR / "figures"

REGION_ORDER = ["East", "North", "South", "West"]
REGION_COLORS = {"East": "tab:red", "North": "tab:blue",
                 "South": "tab:orange", "West": "tab:green"}
N_PER_REGION = 2
SEED = 42  # change this number to get other random sites

# ----------------------------------------------------------------- read
df = pd.read_parquet(DATA)
print(f"Read {DATA.name}: {len(df):,} rows, {df['wb_id'].nunique()} sites")
sites = df.drop_duplicates("wb_id")[["wb_id", "region", "countryName",
                                     "dist_km", "moved"]]
FIG_DIR.mkdir(exist_ok=True)

# ----------------------------------------------------------------- plot
for region in REGION_ORDER:
    pick = sites[sites["region"] == region].sample(N_PER_REGION, random_state=SEED)
    col = REGION_COLORS[region]
    fig, axes = plt.subplots(N_PER_REGION, 2, figsize=(15, 3.2 * N_PER_REGION),
                             sharex=True, squeeze=False)
    print(f"\n{region}:")
    for i, (_, s) in enumerate(pick.iterrows()):
        ts = df[df["wb_id"] == s["wb_id"]]
        print(f"  {s['wb_id']} ({s['countryName']}), {s['dist_km']:.1f} km from cell"
              f"{' (moved)' if s['moved'] else ''}: "
              f"tg mean {ts['tg'].mean():.1f} °C, rr mean {ts['rr'].mean():.2f} mm/day, "
              f"missing tg/rr: {ts['tg'].isna().sum()}/{ts['rr'].isna().sum()}")
        title = f"{s['wb_id']} ({s['countryName']}), {s['dist_km']:.1f} km from E-OBS cell"
        axes[i, 0].plot(ts["date"], ts["tg"], lw=0.5, color=col)
        axes[i, 0].set_ylabel("tg (°C)")
        axes[i, 0].set_title(title, loc="left", fontsize=9)
        axes[i, 1].bar(ts["date"], ts["rr"], width=1, color=col)
        axes[i, 1].set_ylabel("rr (mm/day)")
    fig.suptitle(f"{region}: E-OBS daily mean temperature (left) and rainfall (right), "
                 f"{N_PER_REGION} random sites (seed {SEED})")
    fig.tight_layout()
    f = FIG_DIR / f"04_random_sites_{region}.png"
    fig.savefig(f, dpi=150)
    plt.close(fig)
    print("  Saved:", f)
