"""
Project-wide settings. Edit values here, not in the scripts or notebooks.

All paths are absolute (built from this file's location), so code works from any directory.
"""
from pathlib import Path

# ---------------------------------------------------------------- paths
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"              # downloads exactly as fetched, one folder per dataset
INTERIM_DIR = DATA_DIR / "interim"      # caches and intermediate results (safe to delete)
PROCESSED_DIR = DATA_DIR / "processed"  # the harmonised cube used by the model

# ---------------------------------------------------------------- study area and period
# Study box as (lon_min, lat_min, lon_max, lat_max) in degrees: Luxembourg and surroundings.
# The MSG LST files were downloaded for an older, larger box (3.0, 48.5, 8.5, 52.0);
# the readers cut everything to this one.
DEFAULT_BBOX = (5.3, 49.2, 6.8, 50.3)

STUDY_START_YEAR = 2018  # first full ECOSTRESS growing season
STUDY_END_YEAR = 2025

# Growing season, first and last month (inclusive): April-September.
GROWING_SEASON_START_MONTH = 4
GROWING_SEASON_END_MONTH = 9

# ---------------------------------------------------------------- longer histories
# VCI and SPI compare each year with a long record, so their inputs start earlier.
ERA5_START_YEAR = 1981   # first year of ERA5-Land precipitation downloaded (for SPI)
MODIS_START_YEAR = 2000  # first year of MODIS NDVI (Terra launched in late 1999)

SPI_REF_YEARS = (1991, 2020)  # SPI distribution is fitted on these years (WMO standard normal)
VCI_REF_YEARS = (2000, 2025)  # NDVI min/max per pixel come from these years (all of MODIS)

# ---------------------------------------------------------------- harmonisation
# Common grid: the 0.05 degree MSG LST cells inside DEFAULT_BBOX (23 x 31 cells).
TARGET_RES = 0.05

# Cropland: WorldCover class 40. A grid cell counts as cropland when at least
# CROP_FRACTION_MIN of its area is cropland.
CROPLAND_CLASS = 40
CROP_FRACTION_MIN = 0.5

# VCI of a cell is only used when at least this share of its cropland is cloud-free;
# otherwise it counts as a gap (short gaps are filled from the images before and after).
VCI_MIN_VISIBLE = 0.5

# TRI = LST rise between these two times after local sunrise (hours).
TRI_T1_HOURS = 1.5
TRI_T2_HOURS = 3.5
TRI_MIN_DAYS = 2  # clear days needed in a dekad for a TRI value

# ---------------------------------------------------------------- EDO CDI (validation)
# Classes of the EDO Combined Drought Indicator. 1-3 are the stages of a drought, 4-6 the
# ways out of it. The files also use 7 (no data) and 8 (outside the model domain).
CDI_CLASSES = {0: "No drought", 1: "Watch", 2: "Warning", 3: "Alert", 4: "Recovery",
               5: "Temporary soil moisture recovery", 6: "Temporary vegetation recovery"}
# A dekad counts as a drought dekad (shaded in plots) when at least this share of the
# area, cropland or box, is in Warning or Alert.
CDI_DROUGHT_SHARE = 0.5
