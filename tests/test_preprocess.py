"""Tests for the preprocessing steps, on small hand-made inputs whose answer is known."""
from datetime import UTC, date

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from astral import Observer
from astral.sun import sunrise

from esa_lstm.preprocess import dekads, grid, harmonise, indices, windows


def field(values, lat, lon, time=None):
    dims = ("lat", "lon") if time is None else ("time", "lat", "lon")
    coords = {"lat": lat, "lon": lon} | ({} if time is None else {"time": time})
    return xr.DataArray(np.asarray(values, dtype="float64"), dims=dims, coords=coords)


# ---------------------------------------------------------------- grid
def test_target_grid_is_the_msg_grid_in_the_box():
    t = grid.target_grid((5.3, 49.2, 6.8, 50.3), 0.05)
    assert t.sizes == {"lat": 23, "lon": 31}
    assert (t.lat.values[0], t.lat.values[-1]) == (50.3, 49.2)  # north to south
    assert (t.lon.values[0], t.lon.values[-1]) == (5.3, 6.8)


def test_block_mean_averages_the_pixels_inside_each_cell():
    target = xr.Dataset(coords={"lat": [1.0, 0.0], "lon": [0.0, 1.0]})  # cells 1 deg wide
    values = np.arange(16.0).reshape(4, 4)
    values[0, 0] = np.nan  # a cloudy pixel is skipped
    fine = field(values, lat=[1.25, 0.75, 0.25, -0.25], lon=[-0.25, 0.25, 0.75, 1.25])
    mean = grid.block_mean(fine, target)
    assert mean.sel(lat=1.0, lon=0.0) == pytest.approx(np.mean([1, 4, 5]))
    assert mean.sel(lat=0.0, lon=1.0) == pytest.approx(np.mean([10, 11, 14, 15]))

    weights = fine * 0
    weights[2:, 2:] = 1  # only the south-east cell has weight
    weighted = grid.block_mean(fine, target, weights=weights)
    assert np.isnan(weighted.sel(lat=1.0, lon=0.0))
    assert weighted.sel(lat=0.0, lon=1.0) == pytest.approx(12.5)


def test_cropland_share_ignores_no_data():
    target = xr.Dataset(coords={"lat": [0.0, -1.0], "lon": [0.0, 1.0]})
    classes = field([[40, 40], [30, 0]], lat=[0.2, -0.2], lon=[-0.2, 0.2])  # 0 = no data
    assert grid.cropland_share(classes, target).sel(lat=0.0, lon=0.0) == pytest.approx(2 / 3)


# ---------------------------------------------------------------- dekads
def test_dekad_helpers():
    t = pd.to_datetime(["2022-02-05 13:00", "2022-02-15 00:00", "2022-02-28 00:00", "2024-02-21 00:00"])
    starts = dekads.dekad_start(t)
    assert list(starts.day) == [1, 11, 21, 21]
    assert list(dekads.dekad_end(starts).day) == [10, 20, 28, 29]  # 2024 is a leap year
    assert list(dekads.dekad_of_year(starts)) == [4, 5, 6, 6]
    assert len(dekads.season_dekads(2018, 2018, 4, 9)) == 18


def test_mean_per_dekad_needs_enough_values():
    days = pd.date_range("2022-04-01", "2022-04-20")
    values = np.where(days.day <= 10, 1.0, np.nan)
    values[12] = 5.0  # a single value in the second dekad
    da = xr.DataArray(values, dims="time", coords={"time": days})
    out = dekads.mean_per_dekad(da, min_count=2)
    assert out.values[0] == 1.0 and np.isnan(out.values[1])


# ---------------------------------------------------------------- TRI
def test_sunrise_matches_astral():
    lat, lon = 49.75, 6.05
    for d in [date(2022, 4, 1), date(2022, 6, 21), date(2022, 9, 30)]:
        ref = sunrise(Observer(latitude=lat, longitude=lon), date=d, tzinfo=UTC)
        assert indices.sunrise_utc(d, lat, lon) == pytest.approx(ref.hour * 60 + ref.minute + ref.second / 60, abs=2)


def test_lst_rise_of_steady_warming():
    day = pd.Timestamp("2022-06-21")
    slots = pd.date_range(day + pd.Timedelta(hours=3), day + pd.Timedelta(hours=9), freq="15min")
    hours = ((slots - day).total_seconds() / 3600).to_numpy()
    lst = field(np.repeat(2.0 * hours[:, None, None], 2, axis=2), lat=[49.75], lon=[6.0, 6.05], time=slots)
    assert np.allclose(indices.daily_lst_rise(lst, 1.5, 3.5), 4.0, atol=1e-4)  # 2 K/h for 2 h

    lst[5:, 0, 0] = np.nan  # cloud after 04:15 at one pixel: no value there
    rise = indices.daily_lst_rise(lst, 1.5, 3.5)
    assert np.isnan(rise[0, 0, 0]) and rise[0, 0, 1] == pytest.approx(4.0, abs=1e-4)


# ---------------------------------------------------------------- VCI
def test_vci_is_0_in_the_worst_and_100_in_the_best_year():
    t = pd.to_datetime([f"{y}-06-10" for y in range(2000, 2010)])
    ndvi = field(np.linspace(0.2, 0.8, t.size)[:, None, None], lat=[0.0], lon=[0.0], time=t)
    v = indices.vci(ndvi, (2000, 2009))
    assert float(v.min()) == 0 and float(v.max()) == 100


# ---------------------------------------------------------------- SPI
def test_spi_is_standard_normal_over_the_reference_years():
    rng = np.random.default_rng(0)
    t = pd.date_range("1981-01-01", "2020-12-01", freq="MS")
    rain = rng.gamma(2.0, 30.0, size=(t.size, 2, 2))
    rain[rng.random(rain.shape) < 0.05] = 0  # some completely dry months
    s = indices.spi_monthly(field(rain, [0.0, 1.0], [0.0, 1.0], t), 1, (1981, 2020))
    assert list(s.time.values) == list(t.values)  # order kept after grouping by month
    assert abs(float(s.mean())) < 0.1 and abs(float(s.std()) - 1) < 0.1


def test_spi_daily_gives_one_value_per_dekad():
    t = pd.date_range("1990-01-01", "2000-12-31", freq="D")
    rain = np.random.default_rng(1).gamma(0.5, 4.0, size=(t.size, 1, 1))
    s = indices.spi_daily(field(rain, [0.0], [0.0], t), 30, (1990, 2000))
    assert set(pd.DatetimeIndex(s.time.values).day) == {1, 11, 21}
    assert abs(float(s.mean())) < 0.15


# ---------------------------------------------------------------- EDO CDI
def test_cdi_on_grid_takes_the_pixel_at_each_cell_centre():
    target = xr.Dataset(coords={"lat": [1.0, 0.0], "lon": [0.0, 1.0]})
    starts = pd.to_datetime(["2022-07-01", "2022-07-11"])
    classes = np.array([[0, 1, 2, 3], [4, 5, 6, 0], [1, 1, 2, 2], [3, 3, 0, 0]], dtype="float64")
    # pixels finer than the cells and not aligned with them
    cdi = field(np.stack([classes, classes]), lat=[1.2, 0.6, 0.1, -0.4], lon=[-0.2, 0.4, 0.9, 1.5], time=starts)
    on_grid = harmonise.cdi_on_grid(cdi, target, starts)
    assert on_grid.sel(lat=1.0, lon=0.0).values.tolist() == [0, 0]  # pixel at (1.2, -0.2)
    assert on_grid.sel(lat=0.0, lon=1.0).values.tolist() == [2, 2]  # pixel at (0.1, 0.9)


def test_cdi_shares_and_drought_dekads():
    starts = pd.to_datetime(["2022-07-01", "2022-07-11"])
    cdi = field([[[2, 3], [0, 1]], [[0, 0], [np.nan, 2]]], lat=[1.0, 0.0], lon=[0.0, 1.0], time=starts)
    shares = harmonise.cdi_shares(cdi)
    assert shares["warning"].tolist() == pytest.approx([0.25, 1 / 3])  # NaN cells are left out
    assert shares["alert"].tolist() == pytest.approx([0.25, 0])
    assert harmonise.drought_dekads(cdi).tolist() == [True, False]  # 0.5 vs 1/3 in Warning or Alert
    weights = field([[0, 0], [1, 1]], lat=[1.0, 0.0], lon=[0.0, 1.0])  # only the southern row counts
    assert harmonise.drought_dekads(cdi, weights).tolist() == [False, True]


# ---------------------------------------------------------------- lag-model table (16-day windows)
def test_season_windows_are_the_modis_windows():
    starts = windows.season_windows([2022, 2024])
    assert len(starts) == 24  # 12 per season
    assert (starts[0].date(), starts[11].date()) == (date(2022, 4, 7), date(2022, 9, 30))
    assert starts[12].date() == date(2024, 4, 6)  # leap year: same day of year, one day earlier


def test_windows_tri_is_the_mean_of_the_clear_mornings():
    days = pd.date_range("2022-04-01", "2022-05-10")
    values = np.arange(days.size, dtype="float64")
    values[days == "2022-04-10"] = np.nan  # a cloudy morning
    values[(days >= "2022-04-23") & (days <= "2022-05-08")] = np.nan  # a fully cloudy window
    daily = field(values[:, None, None], lat=[0.0], lon=[0.0], time=days)
    tri = windows.mean_per_window(daily, windows.season_windows([2022]))[:, 0, 0]
    first = values[(days >= "2022-04-07") & (days <= "2022-04-22")]  # 1-6 April: before the first window
    assert tri.sel(time="2022-04-07") == pytest.approx(np.nanmean(first))
    assert np.isnan(tri.sel(time="2022-04-23"))
    assert tri.sel(time="2022-05-09") == pytest.approx(values[-2:].mean())  # one or two mornings are enough


def test_windows_spi_ends_on_the_windows_last_day():
    t = pd.date_range("1990-01-01", "2000-12-31", freq="D")
    rain = np.random.default_rng(2).gamma(0.5, 4.0, size=(t.size, 1, 1))
    starts = windows.window_starts(range(1990, 2001))
    window = pd.Timestamp("1995-07-12")  # runs to 27 July

    def spi_with_extra_rain(day):
        r = rain.copy()
        r[t == day] += 500
        s = indices.spi_daily(field(r, [0.0], [0.0], t), 30, (1990, 2000), starts, starts + pd.Timedelta(days=15))
        return float(s.sel(time=window)[0, 0])

    no_extra = spi_with_extra_rain(None)
    assert spi_with_extra_rain("1995-07-28") == pytest.approx(no_extra)  # the day after: not counted
    assert spi_with_extra_rain("1995-07-27") > no_extra + 1  # the last day: counted


def test_windows_table_keeps_vci_gaps_and_only_cropland():
    starts = pd.to_datetime(["2022-04-07", "2022-04-23"])
    lat, lon = [49.3, 49.25], [5.65, 5.7]
    vci = field(np.full((2, 2, 2), 50.0), lat, lon, starts)
    vci[1, 0, 0] = np.nan  # a cloudy image
    crop = field([[True, False], [False, True]], lat, lon).astype(bool)
    table = windows.to_table(vci, vci * 0 + 8, vci * 0, crop)
    assert list(table.columns) == ["cell_id", "lat", "lon", "window_start", "vci", "tri", "spi1"]
    assert sorted(table["cell_id"].unique()) == ["49.25_5.70", "49.30_5.65"]
    assert len(table) == 4
    gap = table[(table["cell_id"] == "49.30_5.65") & (table["window_start"] == "2022-04-23")]
    assert gap["vci"].isna().all()
