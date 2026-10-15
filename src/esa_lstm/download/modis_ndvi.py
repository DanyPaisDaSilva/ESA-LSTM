"""
Download MODIS MOD13Q1 NDVI for the study box through NASA AppEEARS (input for VCI).

What you get
------------
MOD13Q1 is NASA's 16-day vegetation product at 250 m. NDVI (Normalized Difference
Vegetation Index) is the greenness measure VCI is built from. Also fetched:
  * pixel_reliability (0 good, 1 marginal, 2 snow/ice, 3 cloudy) to drop bad pixels.
AppEEARS adds the VI_Quality layer (detailed quality bits; not used so far) on its own.
AppEEARS returns one GeoTIFF per layer per 16-day period, in plain lon/lat, with the
period's first day in the file name (e.g. doy2022177 = 2022, day 177).

Setup
-----
    EARTHDATA_USER=... and EARTHDATA_PASS=... in .env  (free: https://urs.earthdata.nasa.gov)

Usage
-----
    uv run --env-file .env python -m esa_lstm.download.modis_ndvi        # box and years from config.py
    uv run --env-file .env python -m esa_lstm.download.modis_ndvi --list-layers

Re-running resumes the saved AppEEARS task and skips finished files. If the box, years
or layers changed, the script stops; add --new-task to submit a new request.
AppEEARS processes the request on its side first; that can take minutes to hours.
"""
import argparse
from pathlib import Path

from esa_lstm.config import DEFAULT_BBOX, MODIS_START_YEAR, RAW_DIR, STUDY_END_YEAR
from esa_lstm.download._appeears import AppEEARS, area_task, date_range

PRODUCT = "MOD13Q1.061"
DEFAULT_LAYERS = ["_250m_16_days_NDVI", "_250m_16_days_pixel_reliability"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, default=MODIS_START_YEAR)
    p.add_argument("--end-year", type=int, default=STUDY_END_YEAR)
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS)
    p.add_argument("--out", type=Path, default=RAW_DIR / "modis_ndvi")
    p.add_argument("--new-task", action="store_true", help="submit a fresh task instead of resuming")
    p.add_argument("--list-layers", action="store_true", help="print the product's layer names and exit")
    a = p.parse_args()

    api = AppEEARS()
    if a.list_layers:
        print(sorted(api.layers(PRODUCT)))
        return
    api.login()
    layers = [(PRODUCT, layer) for layer in a.layers]
    api.check_layers(layers)
    task = area_task(f"MOD13Q1 NDVI {a.start_year}-{a.end_year}", layers,
                     date_range(a.start_year, a.end_year), a.bbox)
    task_id = api.submit_or_resume(task, a.out, a.new_task)
    api.wait(task_id)
    api.download_bundle(task_id, a.out)
    print(f"Done. Files are in {a.out.resolve()}")


if __name__ == "__main__":
    main()
