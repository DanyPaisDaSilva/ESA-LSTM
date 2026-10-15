"""
Build the harmonised cube: every index on the common 0.05 deg grid (grid.py), for every
growing-season dekad of the study period (dekads.py), saved to data/processed/.

Steps per index (the index itself is computed on its own grid first, see indices.py):
    TRI  15-min MSG LST -> daily LST rise -> dekad mean (needs TRI_MIN_DAYS clear days)
         -> also as an anomaly (z-score per dekad of year)
    VCI  250 m MODIS -> VCI per pixel -> cropland-weighted mean per cell (skipped when
         less than VCI_MIN_VISIBLE of the cell's cropland is cloud-free)
         -> fill gaps of up to two images -> from 16-day images to dekads by linear
         interpolation in time
    SPI  0.1 deg ERA5-Land -> SPI per pixel -> dekads -> bilinear onto the cells
         monthly input: SPI-1 and SPI-3, each dekad gets its month's value
         daily input:   SPI over the 30 and 90 days ending on each dekad's last day
    CDI  EDO's 1/24 deg class maps, already per dekad -> each cell takes the class of the
         CDI pixel at its centre (classes are categories and cannot be averaged)
Plus the cropland share of each cell and the cropland mask (share >= CROP_FRACTION_MIN).

Usage
-----
    uv run python -m esa_lstm.preprocess.harmonise            # uses whichever ERA5 files exist
    uv run python -m esa_lstm.preprocess.harmonise --rebuild  # re-read the raw MSG/MODIS files
"""
import argparse

import pandas as pd
import xarray as xr

from esa_lstm.config import (
    CDI_CLASSES,
    CDI_DROUGHT_SHARE,
    CROP_FRACTION_MIN,
    PROCESSED_DIR,
    TRI_MIN_DAYS,
    VCI_MIN_VISIBLE,
)
from esa_lstm.preprocess import dekads, grid, indices, load

CUBE_PATH = PROCESSED_DIR / "cube_lux_dekadal.nc"


def tri_on_grid(dekad_starts, rebuild=False):
    """TRI is already on the 0.05 deg MSG grid: only the time step changes."""
    daily = indices.daily_lst_rise(load.load_msg_lst(rebuild)["LST"])
    tri = dekads.mean_per_dekad(daily, min_count=TRI_MIN_DAYS).reindex(time=dekad_starts)
    tri = tri.assign_attrs(long_name="TRI: mean daily morning LST rise", units="K")
    tri_z = indices.anomaly_per_dekad(tri).assign_attrs(long_name="TRI anomaly (z-score per dekad of year)")
    return tri, tri_z


def vci_per_image(target, rebuild=False):
    """VCI per cell and 16-day image: the 250 m VCI averaged with each pixel weighted by its cropland share.

    NaN where less than VCI_MIN_VISIBLE of the cell's cropland is cloud-free: a few clear
    pixels can give a noisy mean.
    """
    load.load_modis(rebuild)  # builds the cache if needed
    ndvi = load.load_modis_ndvi()
    crop_250m = grid.cropland_share(load.load_worldcover(), ndvi)
    vci = grid.block_mean(indices.vci(ndvi), target, weights=crop_250m)
    visible = grid.block_mean(ndvi.notnull().astype("float32"), target, weights=crop_250m)
    return vci.where(visible >= VCI_MIN_VISIBLE)


def vci_on_grid(target, dekad_starts, rebuild=False):
    """VCI per cell (vci_per_image), with short gaps filled, interpolated to dekads."""
    # Fill gaps when at most two images in a row are missing (valid neighbours at most 48 days apart)
    vci = vci_per_image(target, rebuild).interpolate_na("time", max_gap=pd.Timedelta(days=48))
    # A 16-day image describes the middle of its period: place it there, then interpolate
    vci = vci.assign_coords(time=vci.time + pd.Timedelta(days=8))
    return vci.interp(time=dekads.dekad_middle(dekad_starts)).assign_coords(time=dekad_starts)


def spi_on_grid(target, dekad_starts):
    """All SPI variants the downloaded ERA5-Land files allow, on the common grid."""
    out = {}
    if load.era5_available("monthly"):
        precip = load.load_era5_precip("monthly")
        for months in (1, 3):
            spi = indices.monthly_to_dekads(indices.spi_monthly(precip, months), dekad_starts)
            out[f"spi{months}m"] = grid.interpolate(spi, target)
    if load.era5_available("daily"):
        precip = load.load_era5_precip("daily")
        for days in (30, 90):
            spi = indices.spi_daily(precip, days).reindex(time=dekad_starts)
            out[f"spi{days}d"] = grid.interpolate(spi, target)
    if not out:
        raise FileNotFoundError("No ERA5-Land precipitation found (monthly or daily).")
    return out


def cdi_on_grid(cdi, target, dekad_starts):
    """EDO CDI classes on the common cells: each cell takes the class of the CDI pixel at its centre."""
    on_grid = cdi.sel(lat=target.lat, lon=target.lon, method="nearest")
    on_grid = on_grid.assign_coords(lat=target.lat, lon=target.lon).reindex(time=dekad_starts)
    return on_grid.assign_attrs(cdi.attrs)


def build_cube(rebuild=False):
    target = grid.target_grid()
    dekad_starts = dekads.season_dekads()
    crop_share = grid.cropland_share(load.load_worldcover(), target)
    tri, tri_z = tri_on_grid(dekad_starts, rebuild)
    cube = xr.Dataset({
        "tri": tri,
        "tri_z": tri_z,
        "vci": vci_on_grid(target, dekad_starts, rebuild),
        **spi_on_grid(target, dekad_starts),
        "cdi": cdi_on_grid(load.load_edo_cdi(), target, dekad_starts),
        "crop_share": crop_share,
        "crop_mask": (crop_share >= CROP_FRACTION_MIN).assign_attrs(
            long_name=f"cropland share >= {CROP_FRACTION_MIN}"),
    })
    cube.attrs = {"title": "Harmonised drought indices, Luxembourg box",
                  "grid": "0.05 deg MSG LST cell centres",
                  "time": "first day of each growing-season dekad"}
    return cube


def cropland_weights(cube):
    """Weight of each cell in the cropland series: its cropland share, 0 outside the cropland mask."""
    return cube["crop_share"].where(cube["crop_mask"], 0)


def cropland_series(cube):
    """One series per index: mean over the cropland cells, each weighted by its cropland share.

    CDI classes cannot be averaged, so for the CDI the series are the share of the cropland
    in each drought stage (columns cdi_watch, cdi_warning, cdi_alert).
    """
    weights = cropland_weights(cube)
    indices_only = cube[[v for v in cube.data_vars if "time" in cube[v].dims and v != "cdi"]]
    series = indices_only.weighted(weights).mean(("lat", "lon")).to_dataframe()
    return series.join(cdi_shares(cube["cdi"], weights).add_prefix("cdi_"))


def cdi_shares(cdi, weights=None):
    """Per dekad, the share of the area in each CDI drought stage (columns watch, warning, alert).

    weights: (lat, lon), e.g. cropland_weights(cube); default: every cell counts the same.
    """
    def share(in_class):
        in_class = in_class.where(cdi.notnull())
        mean = in_class.mean(("lat", "lon")) if weights is None else in_class.weighted(weights).mean(("lat", "lon"))
        return mean.to_series()

    return pd.DataFrame({CDI_CLASSES[code].lower(): share(cdi == code) for code in (1, 2, 3)})


def drought_dekads(cdi, weights=None):
    """True for the dekads when at least CDI_DROUGHT_SHARE of the area is in CDI Warning or Alert."""
    shares = cdi_shares(cdi, weights)
    return (shares["warning"] + shares["alert"]) >= CDI_DROUGHT_SHARE


def save_cube(cube, path=CUBE_PATH):
    path.parent.mkdir(parents=True, exist_ok=True)
    cube.to_netcdf(path, encoding={v: {"zlib": True} for v in cube.data_vars if cube[v].dtype != bool})
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--rebuild", action="store_true", help="re-read the raw MSG/MODIS files instead of the caches")
    a = p.parse_args()
    cube = build_cube(a.rebuild)
    print(cube)
    print(f"Saved {save_cube(cube)}")


if __name__ == "__main__":
    main()
