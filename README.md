# ESA-LSTM: drought prediction for Luxembourg cropland

A lag model that predicts how crop greenness (VCI) changes over the next 16 days in the
Luxembourg region, from three drought indices: TRI (morning land surface warming),
VCI (vegetation condition) and SPI-1 (rainfall over the last 30 days).

## Start here

**[notebooks/luxembourg_drought_prediction.ipynb](notebooks/luxembourg_drought_prediction.ipynb)**
walks through the whole method, from raw data to results:

1. Load the four datasets and cut them to the study box
2. Compute TRI, VCI and SPI-1, each on its own grid
3. Put all three onto one grid: the 0.05° MSG grid, cropland cells only
4. Split each season into 16-day windows (April–September, 2018–2025)
5. Build model rows: inputs at t-2, t-1 and t, and the target VCI(t+1) − VCI(t)
6. Compare the models

## Inputs

| Dataset | Used for | Resolution |
|---|---|---|
| MSG LST (LSA SAF) | TRI | 0.05°, every 15 min |
| MODIS NDVI | VCI | 250 m, 16-day |
| ERA5-Land precipitation | SPI-1 | 0.1°, daily |
| ESA WorldCover | cropland mask | 10 m |

## Result

RMSE on held-out seasons, in VCI points (lower is better):

| | no change | linear | GBM |
|---|---|---|---|
| all seasons | 13.7 | 11.9 | 13.0 |

Linear regression beats "no change" in 6 of the 8 seasons. Per-season numbers and maps are in the notebook.

## Quick start

```bash
uv sync                      # install the environment from uv.lock
# create .env with your credentials (see docs/setup.md)
uv run --env-file .env python -m esa_lstm.download.msg_lst
uv run --env-file .env python -m esa_lstm.download.modis_ndvi
uv run --env-file .env python -m esa_lstm.download.era5land_daily
uv run python -m esa_lstm.download.worldcover
uv run jupyter lab           # open the notebook and run all cells
```

The downloads need free accounts (NASA Earthdata, LSA SAF, Copernicus CDS) and take a while.
**[docs/setup.md](docs/setup.md)** covers accounts, credentials, the other datasets and development tools.
