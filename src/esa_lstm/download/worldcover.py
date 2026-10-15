"""
Download ESA WorldCover 2021 (v200) land-cover tiles covering the study box.
Used to find cropland.

What you get
------------
WorldCover is split into 3 x 3 degree tiles, named after their south-west corner
(e.g. N48E003 covers 48-51N, 3-6E). Each tile is one GeoTIFF with ~10 m pixels whose
values are land-cover classes, e.g. 10 tree cover, 30 grassland, 40 cropland,
50 built-up (0 = no data). No account needed: files come from ESA's public storage.

Usage
-----
    uv run python -m esa_lstm.download.worldcover        # box from config.py (2 tiles)
"""
import argparse
import math
from pathlib import Path

from esa_lstm.config import DEFAULT_BBOX, RAW_DIR
from esa_lstm.download._http import download_file

BASE_URL = "https://esa-worldcover.s3.eu-central-1.amazonaws.com/v200/2021/map"
FILE_PATTERN = "ESA_WorldCover_10m_2021_v200_{tile}_Map.tif"


def tiles_for_bbox(bbox):
    """Names of the 3 x 3 degree tiles that overlap the box, e.g. ["N48E003", "N48E006"]."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lats = range(math.floor(lat_min / 3) * 3, math.ceil(lat_max / 3) * 3, 3)
    lons = range(math.floor(lon_min / 3) * 3, math.ceil(lon_max / 3) * 3, 3)
    return [f"{'N' if la >= 0 else 'S'}{abs(la):02d}{'E' if lo >= 0 else 'W'}{abs(lo):03d}"
            for la in lats for lo in lons]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bbox", type=float, nargs=4, default=DEFAULT_BBOX,
                   metavar=("LON_MIN", "LAT_MIN", "LON_MAX", "LAT_MAX"))
    p.add_argument("--out", type=Path, default=RAW_DIR / "worldcover")
    a = p.parse_args()

    tiles = tiles_for_bbox(a.bbox)
    print(f"{len(tiles)} tiles cover the box: {', '.join(tiles)}")
    for tile in tiles:
        name = FILE_PATTERN.format(tile=tile)
        status = download_file(f"{BASE_URL}/{name}", a.out / name)  # "missing" = all-ocean tile
        print(f"  {tile}: {status}")
    print(f"Done. Files are in {a.out.resolve()}")


if __name__ == "__main__":
    main()
