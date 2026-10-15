"""
Download ECOSTRESS scenes for the study box through NASA AppEEARS (validation data).

What you get
------------
ECOSTRESS is a NASA thermal sensor on the International Space Station: 70 m pixels,
but it passes over at a different time of day each time. For the growing season of
each study year (config.py) this fetches:
  * ECO_L2T_LSTE.002  LST (land surface temperature, K), cloud (cloud mask), QC
  * ECO_L4T_ESI.002   ESI (Evaporative Stress Index: actual vs. possible plant water
                      use; low = stressed)
File names hold the acquisition date and time (UTC) and the UTM tile (31N or 32N); a
scene crossing both tiles comes as two files. Remove clouds with the `cloud` layer:
the QC flags of version 2 do not include clouds.

Setup
-----
    EARTHDATA_USER=... and EARTHDATA_PASS=... in .env  (free: https://urs.earthdata.nasa.gov)

Usage
-----
    uv run --env-file .env python -m esa_lstm.download.ecostress                  # box and years from config.py
    uv run --env-file .env python -m esa_lstm.download.ecostress --list-layers    # layer names per product

Notes
-----
* ECOSTRESS covers 52S-52N only, and its record starts in July 2018.
* 70 m scenes add up quickly: the default box over 2018-2025 is about 18 GB.
* AppEEARS caps the size of one request, so there is one task per year, saved in its
  own subfolder (<out>/2018/, <out>/2019/, ...). Re-running resumes the saved tasks;
  if the box, season or layers changed, add --new-task to submit new ones.
"""
import argparse
import calendar
from pathlib import Path

from esa_lstm.config import (
    DEFAULT_BBOX,
    GROWING_SEASON_END_MONTH,
    GROWING_SEASON_START_MONTH,
    RAW_DIR,
    STUDY_END_YEAR,
    STUDY_START_YEAR,
)
from esa_lstm.download._appeears import AppEEARS, area_task, date_range

DEFAULT_LAYERS = [
    "ECO_L2T_LSTE.002:LST",
    "ECO_L2T_LSTE.002:cloud",
    "ECO_L2T_LSTE.002:QC",
    "ECO_L4T_ESI.002:ESI",
]

# Growing season as MM-DD: first day of the first month to last day of the last month
_last_day = calendar.monthrange(2001, GROWING_SEASON_END_MONTH)[1]  # 2001: any non-leap year
DEFAULT_SEASON = [f"{GROWING_SEASON_START_MONTH:02d}-01", f"{GROWING_SEASON_END_MONTH:02d}-{_last_day:02d}"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, default=STUDY_START_YEAR)
    p.add_argument("--end-year", type=int, default=STUDY_END_YEAR)
    p.add_argument("--season", nargs=2, default=DEFAULT_SEASON, metavar=("MM-DD", "MM-DD"))
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    p.add_argument("--layers", nargs="+", default=DEFAULT_LAYERS, help="PRODUCT:LAYER pairs")
    p.add_argument("--out", type=Path, default=RAW_DIR / "ecostress")
    p.add_argument("--new-task", action="store_true", help="submit fresh tasks instead of resuming")
    p.add_argument("--list-layers", action="store_true", help="print layer names per product and exit")
    a = p.parse_args()

    layers = [tuple(x.split(":", 1)) for x in a.layers]
    api = AppEEARS()
    if a.list_layers:
        for product in sorted({prod for prod, _ in layers}):
            print(product, sorted(api.layers(product)))
        return
    if a.bbox[3] > 52:
        print("! Box extends north of 52N, where ECOSTRESS has no data.")

    api.login()
    api.check_layers(layers)
    # Submit all years first so NASA processes them in parallel, then collect each one.
    task_ids = {}
    for year in range(a.start_year, a.end_year + 1):
        task = area_task(f"ECOSTRESS {year}", layers, date_range(year, year, *a.season), a.bbox)
        task_ids[year] = api.submit_or_resume(task, a.out / str(year), a.new_task)
    for year, task_id in task_ids.items():
        print(f"{year}:")
        api.wait(task_id)
        api.download_bundle(task_id, a.out / str(year))
    print(f"Done. Files are in {a.out.resolve()}/<year>/")


if __name__ == "__main__":
    main()
