"""
The three drought indices. Each is computed on its input's own grid and time step;
bringing them onto the common grid happens afterwards (harmonise.py).

TRI  Temperature Rise Index (MSG LST, 0.05 deg, every 15 min):
     LST at sunrise + 3.5 h minus LST at sunrise + 1.5 h, per pixel and clear day (K).
     Dry crops cool less by evaporation, so their surface warms faster in the morning.
VCI  Vegetation Condition Index (MODIS NDVI, 250 m, every 16 days):
     100 * (NDVI - lowest) / (highest - lowest), where lowest/highest are that pixel's
     extremes for the same 16-day period over the reference years. 0 = worst on record.
SPI  Standardized Precipitation Index (ERA5-Land, 0.1 deg, monthly or daily):
     rain summed over a window, turned into "how many standard deviations from normal"
     by fitting a gamma distribution per pixel and calendar period over the reference years.
"""
import numpy as np
import pandas as pd
import xarray as xr
from scipy import stats

from esa_lstm.config import SPI_REF_YEARS, TRI_T1_HOURS, TRI_T2_HOURS, VCI_REF_YEARS
from esa_lstm.preprocess.dekads import dekad_end, dekad_of_year, dekad_start


def years(time, first_last):
    """Boolean mask: is each time step inside the (first, last) years, inclusive?"""
    return time.dt.year.isin(range(first_last[0], first_last[1] + 1))


# ---------------------------------------------------------------- TRI
def sunrise_utc(day, lat, lon):
    """Sunrise in minutes after 00 UTC, for arrays of lat/lon (NOAA formula, ~1 min accurate)."""
    g = 2 * np.pi / 365 * (pd.Timestamp(day).dayofyear - 1)  # fraction of the year, in radians
    equation_of_time = 229.18 * (0.000075 + 0.001868 * np.cos(g) - 0.032077 * np.sin(g)
                                 - 0.014615 * np.cos(2 * g) - 0.040849 * np.sin(2 * g))  # minutes
    declination = (0.006918 - 0.399912 * np.cos(g) + 0.070257 * np.sin(g) - 0.006758 * np.cos(2 * g)
                   + 0.000907 * np.sin(2 * g) - 0.002697 * np.cos(3 * g) + 0.00148 * np.sin(3 * g))
    phi = np.radians(lat)
    cos_hour_angle = (np.cos(np.radians(90.833)) / (np.cos(phi) * np.cos(declination))
                      - np.tan(phi) * np.tan(declination))  # 90.833: sun's disc + refraction
    hour_angle = np.degrees(np.arccos(np.clip(cos_hour_angle, -1, 1)))
    return 720 - 4 * (lon + hour_angle) - equation_of_time


def lst_at(day_lst, minutes):
    """LST of one day at a per-pixel time (minutes after 00 UTC).

    Linear interpolation between the two 15-minute slots around that time. The result
    is NaN unless both slots are clear; a slot missing from the archive counts as not clear.
    """
    day = pd.Timestamp(day_lst.time.values[0]).normalize()
    all_slots = pd.date_range(day_lst.time.values[0], day_lst.time.values[-1], freq="15min")
    values = day_lst.reindex(time=all_slots).values           # (slot, lat, lon), gaps -> NaN
    slot_minutes = (all_slots - day).total_seconds().to_numpy() / 60

    position = (minutes - slot_minutes[0]) / 15  # e.g. 3.4 = 40 % of the way from slot 3 to slot 4
    before = np.floor(position).astype(int)
    inside = (before >= 0) & (before + 1 < len(all_slots))
    before = np.clip(before, 0, len(all_slots) - 2)
    v0 = np.take_along_axis(values, before[None], axis=0)[0]
    v1 = np.take_along_axis(values, before[None] + 1, axis=0)[0]
    return np.where(inside, v0 + (position - before) * (v1 - v0), np.nan)


def daily_lst_rise(lst, t1_hours=TRI_T1_HOURS, t2_hours=TRI_T2_HOURS):
    """TRI per pixel and day: LST(sunrise + t2) - LST(sunrise + t1), in K. NaN if not clear."""
    lat, lon = xr.broadcast(lst.lat, lst.lon)  # sunrise is computed for every pixel
    days, rises = [], []
    for day, day_lst in lst.groupby(lst.time.dt.floor("D")):
        sunrise = sunrise_utc(day, lat.values, lon.values)
        rises.append(lst_at(day_lst, sunrise + 60 * t2_hours) - lst_at(day_lst, sunrise + 60 * t1_hours))
        days.append(day)
    return xr.DataArray(np.array(rises, dtype="float32"), dims=("time", "lat", "lon"),
                        coords={"time": days, "lat": lst.lat, "lon": lst.lon},
                        attrs={"long_name": f"LST rise from sunrise + {t1_hours} h to + {t2_hours} h",
                               "units": "K"})


def anomaly_per_dekad(da):
    """z-score per pixel and dekad of the year: (value - mean) / std over all years present."""
    period = xr.DataArray(dekad_of_year(da.time.values), dims="time", coords={"time": da.time},
                          name="dekad_of_year")
    mean = da.groupby(period).mean().sel(dekad_of_year=period)
    std = da.groupby(period).std().sel(dekad_of_year=period)
    return ((da - mean) / std).drop_vars("dekad_of_year")


# ---------------------------------------------------------------- VCI
def vci(ndvi, ref_years=VCI_REF_YEARS):
    """VCI (0-100) per pixel and 16-day image; NaN where NDVI is missing (e.g. cloud)."""
    period = ndvi.time.dt.dayofyear  # MODIS 16-day periods start on the same days every year
    ref = ndvi.sel(time=years(ndvi.time, ref_years))
    lowest = ref.groupby("time.dayofyear").min().sel(dayofyear=period)
    highest = ref.groupby("time.dayofyear").max().sel(dayofyear=period)
    out = (100 * (ndvi - lowest) / (highest - lowest)).drop_vars("dayofyear")
    return out.assign_attrs(long_name="Vegetation Condition Index", units="%",
                            reference=f"{ref_years[0]}-{ref_years[1]}")


# ---------------------------------------------------------------- SPI
def spi_one_period(totals, ref_years):
    """SPI for one calendar period (e.g. all Julys), with the distribution fitted per pixel."""
    ref = totals.sel(time=years(totals.time, ref_years))
    # "Dry" = below 0.01 mm, not == 0: rolling sums leave rounding residues like 1e-14.
    p_dry = (ref < 0.01).mean("time")  # share of reference years with no rain at all
    wet = ref.where(ref >= 0.01)
    # Gamma distribution of the wet totals, maximum-likelihood estimate of Thom (1958)
    mean = wet.mean("time")
    a = np.log(mean) - np.log(wet).mean("time")
    shape = (1 + np.sqrt(1 + 4 * a / 3)) / (4 * a)
    scale = mean / shape
    # probability of a total this low or lower, then the matching standard-normal value
    prob = p_dry + (1 - p_dry) * xr.apply_ufunc(stats.gamma.cdf, totals, shape, 0, scale)
    return xr.apply_ufunc(stats.norm.ppf, prob.clip(1e-6, 1 - 1e-6))


def spi(totals, period, ref_years=SPI_REF_YEARS):
    """SPI of precipitation totals; `period` labels each time step (month, or dekad of year).

    Each calendar period gets its own distribution, since e.g. July and January rain differ.
    """
    out = totals.groupby(period).map(spi_one_period, ref_years=ref_years)
    return out.drop_vars(period.name, errors="ignore").assign_attrs(
        units="1", reference=f"{ref_years[0]}-{ref_years[1]}")


def spi_monthly(precip, months, ref_years=SPI_REF_YEARS):
    """SPI-<months> from monthly totals: rain of this month and the (months - 1) before it."""
    totals = precip.rolling(time=months).sum()
    return spi(totals, totals.time.dt.month, ref_years).assign_attrs(long_name=f"SPI-{months} (monthly ERA5-Land)")


def spi_daily(precip, days, ref_years=SPI_REF_YEARS, starts=None, ends=None):
    """SPI over the `days` days ending on the last day of each period, labelled by the period's first day.

    Periods are dekads, or any other periods given by their first and last days (`starts`,
    `ends`; e.g. the 16-day MODIS windows, which start on the same day of the year every
    year). Each calendar period gets its own distribution: the dekad of the year, or the
    day of the year it starts on.
    """
    totals = precip.rolling(time=days).sum()
    dekads = starts is None
    if dekads:
        starts = pd.DatetimeIndex(dekad_start(totals.time.values).unique())
        ends = dekad_end(starts)
    starts, ends = pd.DatetimeIndex(starts), pd.DatetimeIndex(ends)
    in_record = ends.isin(totals.time.values)  # only periods fully in the record
    starts, ends = starts[in_record], ends[in_record]
    totals = totals.sel(time=ends).assign_coords(time=starts)
    period = xr.DataArray(dekad_of_year(starts) if dekads else starts.dayofyear,
                          dims="time", coords={"time": starts}, name="period")
    return spi(totals, period, ref_years).assign_attrs(long_name=f"SPI-{days}d (daily ERA5-Land)")


def monthly_to_dekads(da, dekad_starts):
    """Give each dekad the value of its calendar month.

    Note: SPI of month M includes rain until the end of M, so the month's first dekads
    see some rain that falls after them.
    """
    months = pd.DatetimeIndex(dekad_starts).to_period("M").to_timestamp()
    return da.sel(time=months).assign_coords(time=pd.DatetimeIndex(dekad_starts))
