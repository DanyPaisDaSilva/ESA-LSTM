"""
Download ERA5-Land daily precipitation for the study box (input for SPI over any
window, e.g. the 30 or 90 days ending each dekad), then combine everything into one
file of daily totals in millimetres.

How daily totals are built
--------------------------
The script uses the hourly "reanalysis-era5-land" dataset but asks only for the 00:00
UTC step of each day. ERA5-Land precipitation accumulates from 00 UTC, and the 00 UTC
value holds the total of the *previous* day. So the value at 00 UTC on day D+1 is the
total of day D: the dates are shifted back by one day. That is also why 1 January of
the year after --end-year is requested.

Setup
-----
As era5land.py (CDS account, CDSAPI_URL / CDSAPI_KEY in .env), but accept the licence
of the hourly ERA5-Land dataset.

Usage
-----
    uv run --env-file .env python -m esa_lstm.download.era5land_daily   # box and years from config.py

One request per year (~45 in total); requests queue on the CDS side, so a full run
takes a couple of hours. Re-running skips the years already downloaded.
"""
import argparse
from pathlib import Path

import pandas as pd

from esa_lstm.config import DEFAULT_BBOX, ERA5_START_YEAR, RAW_DIR, STUDY_END_YEAR
from esa_lstm.download.era5land import cds_area, open_parts, retrieve

DATASET = "reanalysis-era5-land"


def build_request(year, bbox, months=range(1, 13), days=range(1, 32)):
    return {
        "variable": ["total_precipitation"],
        "year": [str(year)],
        "month": [f"{m:02d}" for m in months],
        "day": [f"{d:02d}" for d in days],  # the CDS skips dates that do not exist
        "time": ["00:00"],
        "data_format": "netcdf",
        "download_format": "unarchived",
        "area": cds_area(bbox),
    }


def combine(files, out_file, start_year, end_year):
    ds = open_parts(files)
    days = ds.get_index("time").normalize() - pd.Timedelta(days=1)  # 00 UTC of D+1 = total of D
    tp = ds["tp"].assign_coords(time=days).sel(time=slice(f"{start_year}", f"{end_year}"))
    tp_mm = (tp * 1000).clip(min=0).astype("float32")  # m -> mm; tiny negative values occur
    tp_mm.attrs = {"units": "mm/day", "long_name": "Daily total precipitation (ERA5-Land)"}
    tp_mm.to_dataset(name="precip_mm").to_netcdf(out_file, encoding={"precip_mm": {"zlib": True}})

    expected = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    missing = expected.difference(tp_mm.get_index("time"))
    print(f"Combined {tp_mm.sizes['time']} days into {out_file}")
    if len(missing):
        print(f"  ! {len(missing)} days missing, e.g. {list(missing[:3].date)}")
    annual = float(tp_mm.resample(time="YS").sum().mean())
    print(f"Sanity check: mean annual total over the area = {annual:.0f} mm (Western Europe: ~700-1200 mm)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, default=ERA5_START_YEAR)
    p.add_argument("--end-year", type=int, default=STUDY_END_YEAR)
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    p.add_argument("--out", type=Path, default=RAW_DIR / "era5land_daily")
    a = p.parse_args()

    a.out.mkdir(parents=True, exist_ok=True)
    todo = {f"era5land_tp_00utc_{y}_{y}.nc": build_request(y, a.bbox)
                for y in range(a.start_year, a.end_year + 1)}
    nxt = a.end_year + 1  # its 1 January holds the total of 31 December
    todo[f"era5land_tp_00utc_{nxt}0101.nc"] = build_request(nxt, a.bbox, months=[1], days=[1])

    files = []
    for name, request in todo.items():
        target = a.out / name
        if not target.exists():
            print(f"Requesting {name} (queued on the CDS side; can take a few minutes)")
            try:
                retrieve(DATASET, request, target)
            except Exception as e:  # e.g. next year's 1 January not published yet
                print(f"  ! {name} failed: {e}")
                continue
        files.append(target)
    combine(files, a.out / f"era5land_precip_daily_mm_{a.start_year}_{a.end_year}.nc",
            a.start_year, a.end_year)


if __name__ == "__main__":
    main()
