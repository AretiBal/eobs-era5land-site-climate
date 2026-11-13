# eobs-era5land-site-climate

Extraction of daily precipitation and temperature time series for a set of
water-body sites across Europe, from E-OBS (and, in a later step, ERA5-Land),
so that the two datasets can be compared site by site.

Period: 2017-01-01 to 2026-06-30.
Sites are processed by region in the order East, North, South, West.

## Data (not included in this repository)

1. **E-OBS v34.0e**, ensemble mean, 0.1° regular grid, daily:
   - `rr_ens_mean_0.1deg_reg_v34.0e.nc` (precipitation, mm)
   - `tg_ens_mean_0.1deg_reg_v34.0e.nc` (mean temperature, °C)

   Download from the Copernicus Climate Data Store
   (https://cds.climate.copernicus.eu, dataset "E-OBS daily gridded meteorological
   data for Europe") and place both files in `data/`.

2. **Site list**: `data/high_status_wbs.xlsx` with at least the columns
   `wb_id, lat, lon, country, countryName, region`
   (region one of East, North, South, West). Available on request.

## Workflow

| Script | What it does | Writes |
|---|---|---|
| `01_check_sites_eobs.py` | Checks the site list and E-OBS files; assigns each site to the nearest E-OBS grid cell that has data | `output/sites_eobs_cells.csv`, figures |
| `02_extract_eobs.py` | Extracts daily rr and tg per site, in chunks of one region and one year; existing chunks are skipped, so it can be restarted | `output/eobs_chunks/*.parquet` |
| `03_merge_eobs.py` | Merges the chunks, checks completeness, saves the final dataset | `output/eobs_daily_2017_2026.parquet` and `.csv`, figures |
| `04_plot_random_sites.py` | Plots daily series for random example sites per region | figures |

Run them in order from the project folder:

    pip install -r requirements.txt
    python 01_check_sites_eobs.py
    python 02_extract_eobs.py
    python 03_merge_eobs.py
    python 04_plot_random_sites.py

## Output

`output/eobs_daily_2017_2026.csv` (and `.parquet`): one row per site and day.

| Column | Description |
|---|---|
| `wb_id` | water-body identifier |
| `region`, `country`, `countryName` | from the site list |
| `lat`, `lon` | site coordinates |
| `eobs_lat`, `eobs_lon` | centre of the E-OBS grid cell used |
| `dist_km` | distance from site to that grid cell centre |
| `moved` | `True` if the nearest cell had no data and the nearest cell *with* data was used |
| `date` | YYYY-MM-DD |
| `rr` | daily precipitation (mm) |
| `tg` | daily mean temperature (°C) |

Missing values are written as `NA`. In R: `arrow::read_parquet()` or `read.csv()`.

## Notes

- Grid cells are about 10 km wide, so sites close to each other can share a
  cell and therefore have identical series.
- Sites on coasts or islands whose nearest cell is sea are moved to the
  nearest cell with data (column `moved`); a warning is printed if this is
  further than 30 km.
- E-OBS daily precipitation in many countries refers to 06–06 UTC rather
  than calendar days.

## Acknowledgement

We acknowledge the E-OBS dataset and the data providers in the ECA&D project
(https://www.ecad.eu).
Cornes, R., G. van der Schrier, E.J.M. van den Besselaar, and P.D. Jones. 2018:
An Ensemble Version of the E-OBS Temperature and Precipitation Datasets,
J. Geophys. Res. Atmos., 123. doi:10.1029/2017JD028200
