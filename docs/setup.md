# Setup on a new machine

## 1. Install uv and the project

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh   # install uv (once per machine)
git clone <repo-url> esa-lstm && cd esa-lstm
uv sync                                           # creates .venv with exact versions from uv.lock
```

## 2. Credentials

Create `.env` in the project root (it is gitignored):

```bash
EARTHDATA_USER=...   # https://urs.earthdata.nasa.gov  (modis_ndvi, ecostress)
EARTHDATA_PASS=...
LSASAF_USER=...      # LSA SAF registration            (msg_lst)
LSASAF_PASS=...
CDSAPI_URL=https://cds.climate.copernicus.eu/api
CDSAPI_KEY=...       # token from your CDS profile page (era5land)
```

For ERA5-Land, also accept the licence once on the dataset's "Download" tab at
https://cds.climate.copernicus.eu, or requests fail.

EDO CDI and WorldCover need no account.

## 3. Download data

Each command writes to `data/raw/<dataset>/`, using the box and years in `src/esa_lstm/config.py`.
Add `--help` to any of them for options.

```bash
uv run --env-file .env python -m esa_lstm.download.msg_lst --test     # check LSA SAF login first
uv run --env-file .env python -m esa_lstm.download.msg_lst
uv run --env-file .env python -m esa_lstm.download.modis_ndvi
uv run --env-file .env python -m esa_lstm.download.ecostress        # ~18 GB for the default box
uv run --env-file .env python -m esa_lstm.download.era5land          # monthly precipitation (SPI-1/3)
uv run --env-file .env python -m esa_lstm.download.era5land_daily    # daily precipitation (SPI-30d/90d); slow, one CDS request per year
uv run python -m esa_lstm.download.edo_cdi
uv run python -m esa_lstm.download.worldcover
```

## 4. Harmonise and explore

```bash
uv run python -m esa_lstm.preprocess.harmonise   # builds data/processed/cube_lux_dekadal.nc
uv run jupyter lab                               # then open notebooks/00_inventory.ipynb ... 04_harmonisation.ipynb
```

The first run reads all MSG and MODIS files and caches them in `data/interim/` (a few minutes;
later runs take about 15 s and up to ~7 GB of memory). Pass `--rebuild` after re-downloading.
SPI uses whichever ERA5-Land files exist (monthly, daily or both).

Note: the monthly ERA5-Land product is wrong for Sep 2022 - Feb 2024 (see `notebooks/01_quality`).

## 5. Development

```bash
uv add <package>          # add a dependency (updates pyproject.toml and uv.lock; commit both)
uv add --dev <package>    # dev-only tool
uv run pytest             # run tests
uv run ruff check .       # lint
```
