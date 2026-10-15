"""
Download MSG SEVIRI Land Surface Temperature (LSA SAF product "MLST") for the morning
hours needed by the Temperature Rise Index (TRI). Each full-disk file (~4 MB) is cut
to the study box right away, so only the small cut-outs are kept.

Setup
-----
    LSASAF_USER=... and LSASAF_PASS=... in .env  (from your LSA SAF registration)

Usage
-----
    uv run --env-file .env python -m esa_lstm.download.msg_lst --test      # check login, show file contents
    uv run --env-file .env python -m esa_lstm.download.msg_lst --dry-run   # count files, download nothing
    uv run --env-file .env python -m esa_lstm.download.msg_lst             # real run; resumable

Which time slots are downloaded
-------------------------------
TRI uses the LST rise between 1.5 h and 3.5 h after sunrise. MSG gives one image every
15 minutes. For every day of the growing season this script computes sunrise (UTC) at
the centre of the box and downloads the slots from sunrise + 1.5 h - buffer to
sunrise + 3.5 h + buffer. The buffer (default 30 min) covers the sunrise differences
across the box.

Outputs
-------
<out>/<year>/MLST_crop_YYYYMMDDHHMM.nc   one cut-out per slot (LST, quality_flag, standard_error)
<out>/missing_slots.txt                  slots the server does not have
<out>/failed_slots.txt                   slots that failed; re-run to retry
"""
import argparse
import os
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pandas as pd
import requests
import xarray as xr
from astral import Observer
from astral.sun import sunrise

from esa_lstm.config import (
    DEFAULT_BBOX,
    GROWING_SEASON_END_MONTH,
    GROWING_SEASON_START_MONTH,
    RAW_DIR,
    STUDY_END_YEAR,
    STUDY_START_YEAR,
)
from esa_lstm.download._http import LoginError, download_file

BASE_URL = "https://datalsasaf.lsasvcs.ipma.pt/PRODUCTS/MSG/MLST/NETCDF"
MB_PER_FILE = 4.3  # approximate size of one full-disk file

# The netCDF library is not thread-safe: downloads run in parallel, cutting one at a time.
NETCDF_LOCK = threading.Lock()


# ---------------------------------------------------------------- which slots
def slots_for_day(day, bbox, buffer_min=30):
    """15-minute slot start times (UTC) covering sunrise + 1.5 h ... sunrise + 3.5 h."""
    lon_min, lat_min, lon_max, lat_max = bbox
    centre = Observer(latitude=(lat_min + lat_max) / 2, longitude=(lon_min + lon_max) / 2)
    sr = sunrise(centre, date=day, tzinfo=UTC)
    start = sr + timedelta(minutes=90 - buffer_min)
    end = sr + timedelta(minutes=210 + buffer_min)
    first = start.replace(minute=start.minute // 15 * 15, second=0, microsecond=0)
    return list(pd.date_range(first, end, freq="15min").to_pydatetime())


def all_slots(start_year, end_year, first_month, last_month, bbox, buffer_min):
    days = pd.date_range(f"{start_year}-01-01", f"{end_year}-12-31", freq="D")
    days = days[(days.month >= first_month) & (days.month <= last_month)]
    return [t for day in days.date for t in slots_for_day(day, bbox, buffer_min)]


def url_for(t):
    return f"{BASE_URL}/{t:%Y/%m/%d}/NETCDF4_LSASAF_MSG_LST_MSG-Disk_{t:%Y%m%d%H%M}.nc"


# ---------------------------------------------------------------- cut and save
def cut_to_box(ds, bbox):
    """The full disk has 1-D lat (north to south) and lon axes, so a plain selection works."""
    lon_min, lat_min, lon_max, lat_max = bbox
    out = ds.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))
    if out.sizes["lat"] == 0 or out.sizes["lon"] == 0:
        raise ValueError("The box does not overlap the file's grid.")
    return out


def save_cut(src, dest, bbox):
    with NETCDF_LOCK, xr.open_dataset(src) as ds:
        cut = cut_to_box(ds, bbox).load()
        dest.parent.mkdir(parents=True, exist_ok=True)
        part = dest.with_name(dest.name + ".part")
        cut.to_netcdf(part, encoding={v: {"zlib": True, "complevel": 4} for v in cut.data_vars})
        part.replace(dest)
    return cut


def process_slot(session, t, out_dir, tmp_dir, bbox):
    """Download one full-disk file, cut it, delete the full file. Returns (t, status)."""
    dest = out_dir / f"{t:%Y}" / f"MLST_crop_{t:%Y%m%d%H%M}.nc"
    if dest.exists():
        return t, "cached"
    tmp = tmp_dir / f"{t:%Y%m%d%H%M}.nc"
    try:
        if download_file(url_for(t), tmp, session=session, timeout=(15, 120)) == "missing":
            return t, "missing"
        save_cut(tmp, dest, bbox)
        return t, "ok"
    except LoginError:
        raise  # every later slot would fail too: stop the run
    except Exception as e:  # network drop after retries, corrupt file, ...: note it, carry on
        return t, f"failed: {type(e).__name__}: {e}"
    finally:
        tmp.unlink(missing_ok=True)


def login_session():
    user, pw = os.environ.get("LSASAF_USER"), os.environ.get("LSASAF_PASS")
    if not user or not pw:
        sys.exit("Set the LSASAF_USER and LSASAF_PASS environment variables first.")
    s = requests.Session()
    s.auth = (user, pw)  # HTTP Basic auth
    return s


# ---------------------------------------------------------------- main
def run_test(out_dir, bbox):
    """Fetch one file, print its contents and the cut-out: a quick login and format check."""
    t = datetime(2022, 7, 1, 6, 0, tzinfo=UTC)
    print("Requesting", url_for(t))
    dest = out_dir / "test_full_disk.nc"
    try:
        status = download_file(url_for(t), dest, session=login_session())
    except LoginError as e:
        sys.exit(f"LOGIN PROBLEM: {e}")
    if status == "missing":
        sys.exit("File not found (404). Check the URL pattern in the LSA SAF data service.")
    with xr.open_dataset(dest) as ds:
        print("\n--- full file ---\n", ds)
    print("\n--- cut to box ---\n", save_cut(dest, out_dir / "test_cropped.nc", bbox))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, default=STUDY_START_YEAR)
    p.add_argument("--end-year", type=int, default=STUDY_END_YEAR)
    p.add_argument("--first-month", type=int, default=GROWING_SEASON_START_MONTH)
    p.add_argument("--last-month", type=int, default=GROWING_SEASON_END_MONTH)
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    p.add_argument("--buffer-min", type=int, default=30)
    p.add_argument("--out", type=Path, default=RAW_DIR / "msg_lst")
    p.add_argument("--workers", type=int, default=3, help="parallel downloads")
    p.add_argument("--dry-run", action="store_true", help="only count files and estimate volume")
    p.add_argument("--test", action="store_true", help="download one file and print its structure")
    a = p.parse_args()
    bbox = tuple(a.bbox)
    a.out.mkdir(parents=True, exist_ok=True)

    if a.test:
        run_test(a.out, bbox)
        return

    slots = all_slots(a.start_year, a.end_year, a.first_month, a.last_month, bbox, a.buffer_min)
    print(f"{len(slots)} slots, approx. {len(slots) * MB_PER_FILE / 1000:.1f} GB to transfer "
          f"(only the small cut-outs are kept).")
    if a.dry_run:
        return

    session = login_session()
    missing, failed, done = [], [], 0
    t0 = time.time()
    with tempfile.TemporaryDirectory(dir=a.out) as tmp, ThreadPoolExecutor(a.workers) as pool:
        futures = [pool.submit(process_slot, session, t, a.out, Path(tmp), bbox) for t in slots]
        try:
            for i, fut in enumerate(as_completed(futures), 1):
                t, status = fut.result()
                if status == "missing":
                    missing.append(t)
                elif status.startswith("failed"):
                    failed.append(f"{t:%Y-%m-%d %H:%M}  {status}")
                    print(f"  ! {t:%Y-%m-%d %H:%M} {status}", flush=True)
                else:
                    done += 1
                if i % 25 == 0 or i == len(futures):
                    minutes_left = (len(futures) - i) / (i / (time.time() - t0)) / 60
                    print(f"[{i}/{len(futures)}] ok/cached: {done}  missing: {len(missing)}  "
                          f"failed: {len(failed)}  ~{minutes_left:.0f} min left", flush=True)
        except (LoginError, KeyboardInterrupt) as e:
            pool.shutdown(wait=False, cancel_futures=True)  # drop queued slots instead of running them
            sys.exit(f"LOGIN PROBLEM: {e}" if isinstance(e, LoginError)
                     else "Interrupted. Re-run the same command to resume; finished files are skipped.")

    if missing:
        (a.out / "missing_slots.txt").write_text("\n".join(f"{t:%Y-%m-%d %H:%M}" for t in sorted(missing)))
        print(f"{len(missing)} slots are not on the server; listed in missing_slots.txt")
    if failed:
        (a.out / "failed_slots.txt").write_text("\n".join(sorted(failed)))
        print(f"{len(failed)} slots failed; listed in failed_slots.txt. Re-run to retry them.")


if __name__ == "__main__":
    main()
