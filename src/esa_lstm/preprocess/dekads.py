"""
The common time step: dekads.

A dekad is a third of a month: days 1-10, 11-20 and 21 to the end of the month
(8 to 11 days). There are 36 per year; each is labelled by its first day.
Dekads are the time step of EDO's drought indicator, and sit between the 16-day MODIS
images and the daily TRI values.
"""
import pandas as pd
import xarray as xr

from esa_lstm.config import (
    GROWING_SEASON_END_MONTH,
    GROWING_SEASON_START_MONTH,
    STUDY_END_YEAR,
    STUDY_START_YEAR,
)


def dekad_start(times):
    """First day of the dekad each time falls in, e.g. 2022-07-15 13:00 -> 2022-07-11."""
    t = pd.DatetimeIndex(times).normalize()
    first_day = t.day.map(lambda d: 1 if d <= 10 else 11 if d <= 20 else 21)
    return t - pd.to_timedelta(t.day - first_day, unit="D")


def dekad_end(starts):
    """Last day of each dekad: the 10th, the 20th, or the last day of the month."""
    s = pd.DatetimeIndex(starts)
    return pd.DatetimeIndex([d + pd.Timedelta(days=9) if d.day < 21 else d + pd.offsets.MonthEnd(0)
                             for d in s])


def dekad_middle(starts):
    s = pd.DatetimeIndex(starts)
    return s + (dekad_end(s) + pd.Timedelta(days=1) - s) / 2


def dekad_of_year(starts):
    """1 (1-10 January) ... 36 (21-31 December)."""
    s = pd.DatetimeIndex(starts)
    return (s.month - 1) * 3 + s.day // 10 + 1


def season_dekads(start_year=STUDY_START_YEAR, end_year=STUDY_END_YEAR,
                  first_month=GROWING_SEASON_START_MONTH, last_month=GROWING_SEASON_END_MONTH):
    """Start dates of all growing-season dekads in the study period (18 per year)."""
    months = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-01", freq="MS")
    months = months[(months.month >= first_month) & (months.month <= last_month)]
    return pd.DatetimeIndex([m + pd.Timedelta(days=d) for m in months for d in (0, 10, 20)])


def mean_per_dekad(da, min_count=1):
    """Average a sub-dekadal series (e.g. daily) per dekad; NaN where fewer than min_count values."""
    dekad = xr.DataArray(dekad_start(da.time.values), dims="time", coords={"time": da.time}, name="dekad")
    mean = da.groupby(dekad).mean()
    count = da.notnull().groupby(dekad).sum()
    return mean.where(count >= min_count).rename(dekad="time")
