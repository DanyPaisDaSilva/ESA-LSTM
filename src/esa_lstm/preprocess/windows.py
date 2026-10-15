"""
Build the lag model's input table (docs/PLAN.md section 3): one row per cropland cell and
16-day MODIS window starting in the growing season, with VCI, TRI and SPI-1.

The indices and the grid are the same as in the dekadal cube (harmonise.py); what differs:
    time step  the 16-day MODIS windows themselves (they start on day of year 1, 17, 33, ...)
    VCI        per image, without gap filling or interpolation in time: both would use the
               next window, which is what the lag model predicts
    TRI        mean of the clear mornings in the window (one is enough), raw TRI in K
    SPI-1      over the 30 days ending on the window's last day, from daily ERA5-Land
Only cropland cells (crop share >= CROP_FRACTION_MIN) are kept.

Usage
-----
    uv run python -m esa_lstm.preprocess.windows            # uses the MSG/MODIS caches
    uv run python -m esa_lstm.preprocess.windows --rebuild  # re-read the raw MSG/MODIS files
"""
import argparse

import numpy as np
import pandas as pd
import xarray as xr

from esa_lstm.config import (
    CROP_FRACTION_MIN,
    GROWING_SEASON_END_MONTH,
    GROWING_SEASON_START_MONTH,
    PROCESSED_DIR,
    STUDY_END_YEAR,
    STUDY_START_YEAR,
)
from esa_lstm.preprocess import grid, harmonise, indices, load

TABLE_PATH = PROCESSED_DIR / "indices.parquet"
WINDOW_DAYS = 16
SPI_DAYS = 30  # SPI-1


def window_starts(years):
    """First days of all MODIS 16-day windows in the given years (day of year 1, 17, ..., 353)."""
    return pd.DatetimeIndex([pd.Timestamp(year, 1, 1) + pd.Timedelta(days=d)
                             for year in years for d in range(0, 365, WINDOW_DAYS)])


def season_windows(years=range(STUDY_START_YEAR, STUDY_END_YEAR + 1)):
    """The windows that start in the growing season (12 per year for April-September)."""
    starts = window_starts(years)
    return starts[(starts.month >= GROWING_SEASON_START_MONTH) & (starts.month <= GROWING_SEASON_END_MONTH)]


def mean_per_window(daily, starts):
    """Mean of the daily values inside each window; NaN where there is none. Days outside every window are ignored."""
    days = pd.DatetimeIndex(daily.time.values)
    i = starts.searchsorted(days, side="right") - 1  # the last window starting on or before each day
    window = starts[np.clip(i, 0, None)]
    inside = (i >= 0) & (days < window + pd.Timedelta(days=WINDOW_DAYS))
    labels = xr.DataArray(window[inside], dims="time", coords={"time": daily.time[inside]}, name="window")
    return daily.isel(time=inside).groupby(labels).mean().rename(window="time").reindex(time=starts)


def spi_per_window(precip, starts):
    """SPI-1 over the 30 days ending on each window's last day, fitted per window of the year."""
    years = range(pd.Timestamp(precip.time.values[0]).year, pd.Timestamp(precip.time.values[-1]).year + 1)
    every = window_starts(years)  # the distribution is fitted on all years, not only the study years
    spi = indices.spi_daily(precip, SPI_DAYS, starts=every, ends=every + pd.Timedelta(days=WINDOW_DAYS - 1))
    return spi.reindex(time=starts)


def to_table(vci, tri, spi1, crop_mask):
    """The (time, lat, lon) fields as one row per cropland cell and window, in the columns of PLAN.md section 3."""
    ds = xr.Dataset({"vci": vci, "tri": tri, "spi1": spi1, "crop": crop_mask})
    df = ds.to_dataframe().reset_index()
    df = df[df["crop"].astype(bool)].rename(columns={"time": "window_start"})
    df.insert(0, "cell_id", [f"{lat:.2f}_{lon:.2f}" for lat, lon in zip(df["lat"], df["lon"])])
    df = df[["cell_id", "lat", "lon", "window_start", "vci", "tri", "spi1"]].astype({"cell_id": str})
    return df.sort_values(["cell_id", "window_start"], ignore_index=True)


def build_table(rebuild=False):
    target = grid.target_grid()
    starts = season_windows()
    crop_mask = grid.cropland_share(load.load_worldcover(), target) >= CROP_FRACTION_MIN
    vci = harmonise.vci_per_image(target, rebuild).reindex(time=starts)  # a missing image stays NaN
    tri = mean_per_window(indices.daily_lst_rise(load.load_msg_lst(rebuild)["LST"]), starts)
    spi1 = grid.interpolate(spi_per_window(load.load_era5_precip("daily"), starts), target)
    return to_table(vci, tri, spi1, crop_mask)


def save_table(table, path=TABLE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    table.to_parquet(path, index=False)
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rebuild", action="store_true", help="re-read the raw MSG/MODIS files instead of the caches")
    a = p.parse_args()
    table = build_table(a.rebuild)
    print(table.describe(include="all").T)
    print(f"Saved {save_table(table)}")


if __name__ == "__main__":
    main()
