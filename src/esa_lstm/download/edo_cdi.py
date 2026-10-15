"""
Download the EDO Combined Drought Indicator (CDI) from the Copernicus EDO data archive.

What you get
------------
The European Drought Observatory (EDO, run by the EU's Joint Research Centre)
publishes the CDI: one Europe-wide map every 10 days (~5 km, from 2012) that classes
each cell as Watch, Warning, Alert or a recovery class. It is used to check whether
our own indices flag the same drought periods. No account needed.

How it works
------------
The archive is a set of plain web folders whose exact names were not known in
advance, so the script looks them up: it lists Drought_Observatories_datasets, takes
the EDO "Combined Drought" folder, picks its newest version folder (ver...), and keeps
the files whose names contain a year in the requested range. Files are Europe-wide;
they are cut to the study box during processing.

Usage
-----
    uv run python -m esa_lstm.download.edo_cdi --list         # show what it found, download nothing
    uv run python -m esa_lstm.download.edo_cdi                # study years from config.py
    uv run python -m esa_lstm.download.edo_cdi --dataset-url <folder URL>   # if it picks the wrong folder

If nothing suitable is found, the CDI is also on the EDO portal
(https://drought.emergency.copernicus.eu) and its WCS map service.
"""
import argparse
import re
import sys
from pathlib import Path
from urllib.parse import urljoin

import requests

from esa_lstm.config import RAW_DIR, STUDY_END_YEAR, STUDY_START_YEAR
from esa_lstm.download._http import download_file

# The old JRC mirror (jeodpp.jrc.ec.europa.eu/ftp/jrc-opendata/DROUGHTOBS/...) is no
# longer updated and its CDI folder is empty; the data moved here.
ROOT = "https://drought.emergency.copernicus.eu/data/Drought_Observatories_datasets/"
DATA_EXT = (".tif", ".tiff", ".nc", ".zip", ".gz")


def name_of(url):
    return url.rstrip("/").rsplit("/", 1)[-1]


def list_dir(url):
    """Read a plain 'Index of' web folder: returns (subfolder URLs, file URLs)."""
    r = requests.get(url, timeout=60)
    r.raise_for_status()
    dirs, files = [], []
    for href in re.findall(r'href="([^"]+)"', r.text):
        if href.startswith(("?", "/", "#", "..", "http", "mailto")):
            continue  # sorting links, parent folder, external links
        (dirs if href.endswith("/") else files).append(urljoin(url, href))
    return dirs, files


def find_cdi_folder():
    dirs, _ = list_dir(ROOT)
    cdi = [d for d in dirs if "combined_drought" in name_of(d).lower()]
    cdi.sort(key=lambda d: not name_of(d).upper().startswith("EDO"))  # prefer the EDO one
    if not cdi:
        print("No Combined Drought Indicator folder found. Folders available:")
        for d in dirs:
            print("  ", name_of(d))
        sys.exit("Pass the right one with --dataset-url.")
    return cdi[0]


def newest_version(folder):
    """The ver... subfolder with the highest version number (or the folder itself if none)."""
    dirs, _ = list_dir(folder)
    versions = [d for d in dirs if name_of(d).lower().startswith("ver")]
    if not versions:
        return folder
    return max(versions, key=lambda d: [int(x) for x in re.findall(r"\d+", name_of(d))])


def data_files(folder, depth=0, max_depth=4):
    """All data files in a folder and its subfolders."""
    dirs, files = list_dir(folder)
    out = [f for f in files if f.lower().endswith(DATA_EXT)]
    if depth < max_depth:
        for d in dirs:
            out += data_files(d, depth + 1, max_depth)
    return out


def years_in(name):
    """Years in a file name, from a standalone YYYY, YYYYMM or YYYYMMDD."""
    return {int(y) for y in re.findall(r"(?<!\d)((?:19|20)\d{2})(?:\d{2}(?:\d{2})?)?(?!\d)", name)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--start-year", type=int, default=STUDY_START_YEAR)
    p.add_argument("--end-year", type=int, default=STUDY_END_YEAR)
    p.add_argument("--dataset-url", help="CDI folder URL, skips auto-discovery")
    p.add_argument("--version", help="version folder name, e.g. ver4-0-0 (default: newest)")
    p.add_argument("--include-undated", action="store_true", help="also fetch files with no year in the name")
    p.add_argument("--out", type=Path, default=RAW_DIR / "edo_cdi")
    p.add_argument("--list", action="store_true", help="only list matching files")
    a = p.parse_args()

    folder = a.dataset_url or find_cdi_folder()
    version = urljoin(folder, a.version.rstrip("/") + "/") if a.version else newest_version(folder)
    print("Dataset folder:", folder)
    print("Version folder:", version)

    def wanted(url):
        years = years_in(name_of(url))
        return any(a.start_year <= y <= a.end_year for y in years) if years else a.include_undated

    files = [f for f in data_files(version) if wanted(f)]
    print(f"{len(files)} files match {a.start_year}-{a.end_year}")
    if a.list or not files:
        for f in files:
            print("  ", f[len(version):])
        if not files:
            print("Nothing matched. Try --list with --include-undated to see what is there.")
        return

    for i, f in enumerate(files, 1):
        status = download_file(f, a.out / f[len(version):])  # keep the archive's subfolders
        if status != "cached" or i == len(files):
            print(f"  [{i}/{len(files)}] {name_of(f)}: {status}", flush=True)
    print(f"Done. Files are in {a.out.resolve()}")


if __name__ == "__main__":
    main()
