"""
Download ERA5-Land monthly precipitation for the study box (input for monthly SPI),
then combine everything into one file of monthly totals in millimetres.

What you get
------------
ERA5-Land is a reanalysis (past weather re-simulated by ECMWF's model, held close to
observations): a gap-free grid of 0.1 degrees (~7 x 11 km here). This script uses the
"monthly averaged reanalysis" dataset, which is small and enough for monthly SPI.
For daily data (SPI over any window) see era5land_daily.py.

Known problem
-------------
For Sep 2022 - Feb 2024 this monthly product is 8-75 mm/month lower than the daily
ERA5-Land data summed per month; all other months agree. See notebooks/01_quality.

Setup (one time)
----------------
1. Free account at https://cds.climate.copernicus.eu
2. On the ERA5-Land monthly dataset page, "Download" tab: accept the licence once.
3. In .env:  CDSAPI_URL=https://cds.climate.copernicus.eu/api  and  CDSAPI_KEY=<token from your profile>

Usage
-----
    uv run --env-file .env python -m esa_lstm.download.era5land   # box and years from config.py

Units
-----
The monthly dataset stores precipitation as the *average daily* total in metres. The
script converts it to monthly totals in mm (x 1000 x days in month).
"""
import argparse
import zipfile
from pathlib import Path

import xarray as xr

from esa_lstm.config import DEFAULT_BBOX, ERA5_START_YEAR, RAW_DIR, STUDY_END_YEAR

DATASET = "reanalysis-era5-land-monthly-means"


# ---------------------------------------------------------------- shared with era5land_daily.py
def year_chunks(start, end, size):
    """(first, last) year pairs, e.g. (1981, 1990), (1991, 2000), ..."""
    return [(y, min(y + size - 1, end)) for y in range(start, end + 1, size)]


def cds_area(bbox):
    lon_min, lat_min, lon_max, lat_max = bbox
    return [lat_max, lon_min, lat_min, lon_max]  # CDS order: North, West, South, East


def retrieve(dataset, request, target):
    """Ask the CDS for a file (requests wait in a queue on their side) and save it to target."""
    import cdsapi
    part = target.with_name(target.name + ".part")
    cdsapi.Client().retrieve(dataset, request, str(part))
    if zipfile.is_zipfile(part):  # the CDS sometimes zips the NetCDF even when asked not to
        with zipfile.ZipFile(part) as z:
            data = z.read(next(n for n in z.namelist() if n.endswith(".nc")))
        part.write_bytes(data)
    part.replace(target)


def open_parts(files):
    """Open the downloaded pieces as one Dataset with a `time` axis (duplicates removed)."""
    parts = []
    for f in files:
        ds = xr.open_dataset(f).rename({"valid_time": "time"})
        parts.append(ds.drop_vars(["number", "expver"], errors="ignore"))
    ds = xr.concat(parts, dim="time").sortby("time")
    return ds.sel(time=~ds.get_index("time").duplicated())


# ---------------------------------------------------------------- monthly
def build_request(y0, y1, bbox):
    return {
        "product_type": ["monthly_averaged_reanalysis"],
        "variable": ["total_precipitation"],
        "year": [str(y) for y in range(y0, y1 + 1)],
        "month": [f"{m:02d}" for m in range(1, 13)],
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": cds_area(bbox),
    }


def combine(files, out_file):
    ds = open_parts(files)
    tp_mm = (ds["tp"] * 1000 * ds["time"].dt.days_in_month).astype("float32")  # m/day -> mm/month
    tp_mm.attrs = {"units": "mm/month", "long_name": "Monthly total precipitation (ERA5-Land)"}
    tp_mm.to_dataset(name="precip_mm").to_netcdf(out_file, encoding={"precip_mm": {"zlib": True}})
    print(f"Combined {ds.sizes['time']} months into {out_file}")
    print(f"Sanity check: mean monthly total over the area = {float(tp_mm.mean()):.1f} mm "
          "(Western Europe: roughly 50-90 mm)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, default=ERA5_START_YEAR)
    p.add_argument("--end-year", type=int, default=STUDY_END_YEAR)
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    p.add_argument("--out", type=Path, default=RAW_DIR / "era5land")
    p.add_argument("--years-per-request", type=int, default=10,
                   help="the CDS refuses very large requests; 10 years of monthly data is fine")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    files = []
    for y0, y1 in year_chunks(a.start_year, a.end_year, a.years_per_request):
        target = a.out / f"era5land_tp_monthlymean_{y0}_{y1}.nc"
        files.append(target)
        if target.exists():
            print(f"{target.name}: already downloaded")
            continue
        print(f"Requesting {y0}-{y1} (queued on the CDS side; can take a few minutes)")
        retrieve(DATASET, build_request(y0, y1, a.bbox), target)
    combine(files, a.out / f"era5land_precip_monthly_mm_{a.start_year}_{a.end_year}.nc")


if __name__ == "__main__":
    main()
