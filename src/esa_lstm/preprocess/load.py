"""
Readers for the raw downloads in data/raw/.

Every reader returns xarray data in the same layout, so later steps never need to
know where the data came from:
  * dimensions `time`, `lat`, `lon`, with lat running north to south;
  * plain longitude/latitude in degrees (all downloads are already in EPSG:4326);
  * invalid pixels set to NaN.

MSG and MODIS come as thousands of small files, which take a minute or two to read.
The stacked result is therefore cached in data/interim/; pass rebuild=True after
re-downloading.
"""
import re
from pathlib import Path

import numpy as np
import pandas as pd
import rioxarray
import xarray as xr

from esa_lstm.config import CDI_CLASSES, DEFAULT_BBOX, INTERIM_DIR, RAW_DIR
from esa_lstm.preprocess.grid import nearest_centre, target_grid


# ---------------------------------------------------------------- helpers
def parse_doy(name):
    """Acquisition time from AppEEARS file names: "doy2018209194136" = 2018, day 209, 19:41:36 UTC."""
    year, doy, hh, mm, ss = re.search(r"doy(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})", name).groups()
    return pd.Timestamp(f"{year}-01-01 {hh}:{mm}:{ss}") + pd.Timedelta(days=int(doy) - 1)


def snap(coord, res):
    """Round coordinates to their regular grid: MSG stores 3.0500030 for 3.05 (float32 noise)."""
    return np.round(np.round(np.asarray(coord, dtype="float64") / res) * res, 6)


def cut_to_box(da, bbox=DEFAULT_BBOX):
    lon_min, lat_min, lon_max, lat_max = bbox
    return da.sel(lat=slice(lat_max, lat_min), lon=slice(lon_min, lon_max))  # lat runs north to south


def read_geotiff(path, raw=False):
    """One single-band GeoTIFF as a (lat, lon) DataArray.

    raw=False applies the file's scale factor and turns its fill value into NaN;
    raw=True keeps the stored integers (needed for class maps and flag layers).
    """
    da = rioxarray.open_rasterio(path, mask_and_scale=not raw).squeeze("band", drop=True)
    return clean_raster(da)


def clean_raster(da):
    """Rename to lat/lon and drop the raster metadata. Its _FillValue (e.g. 0 for WorldCover)
    would otherwise travel with derived results and turn real zeros into NaN when saved."""
    da = da.rename(y="lat", x="lon").drop_vars("spatial_ref")
    da.attrs, da.encoding = {}, {}
    return da


def cached(path, build, rebuild=False):
    """Open the dataset saved at `path`; the first time, create it with build() and save it."""
    if rebuild or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        part = path.with_name(path.name + ".part")
        ds = build()
        ds.to_netcdf(part, encoding={v: {**ds[v].encoding, "zlib": True} for v in ds.data_vars})  # compressed
        part.replace(path)
    return xr.open_dataset(path)


# ---------------------------------------------------------------- MSG LST (for TRI)
def load_msg_lst(rebuild=False):
    """All downloaded morning MSG LST slots, cut to the study box.

    Returns a Dataset with LST (deg C) and standard_error (K), on the 0.05 deg MSG grid.
    LSA SAF only computes LST for clear-sky land pixels, so NaN LST means cloud.
    """
    def build():
        slots = []
        for f in sorted((RAW_DIR / "msg_lst").glob("[0-9]*/MLST_crop_*.nc")):
            with xr.open_dataset(f) as ds:
                ds = ds[["LST", "standard_error"]].astype("float32")
                ds = ds.assign_coords(lat=snap(ds.lat, 0.05), lon=snap(ds.lon, 0.05))
                slots.append(cut_to_box(ds).load())
        ds = xr.concat(slots, dim="time")
        # Each raw file stores its time as int16 "seconds since <that slot>", which would
        # overflow for the whole stack: store minutes since 1970 instead.
        ds["time"].encoding = {"units": "minutes since 1970-01-01", "dtype": "int64"}
        return ds

    return cached(INTERIM_DIR / "msg_lst.nc", build, rebuild)


# ---------------------------------------------------------------- MODIS NDVI (for VCI)
def load_modis(rebuild=False):
    """MOD13Q1 at 250 m, one image per 16 days: Dataset with `ndvi` and `reliability`.

    `time` is the first day of each 16-day period. reliability: 0 good, 1 marginal,
    2 snow/ice, 3 cloudy. No pixels are removed here; see load_modis_ndvi().
    """
    def build():
        folder = next((RAW_DIR / "modis_ndvi").glob("MOD13Q1.061_*"))
        ndvi_files = sorted(folder.glob("*_NDVI_doy*.tif"))
        images = []
        for f in ndvi_files:
            ndvi = read_geotiff(f)  # scaled to -0.2..1, fill value -> NaN
            reliability = read_geotiff(str(f).replace("_NDVI_", "_pixel_reliability_"), raw=True)
            images.append(xr.Dataset({"ndvi": ndvi, "reliability": reliability}))
        times = pd.DatetimeIndex([parse_doy(f.name) for f in ndvi_files], name="time")
        ds = xr.concat(images, dim=times)
        # Store like the original files (int16 x 0.0001) to keep the cache small. A float32
        # scale factor makes xarray read NDVI back as float32 (half the memory of float64).
        ds["ndvi"].encoding = {"dtype": "int16", "scale_factor": np.float32(1e-4), "_FillValue": -32768}
        ds["reliability"].encoding = {"dtype": "int8"}
        return ds

    return cached(INTERIM_DIR / "modis_ndvi.nc", build, rebuild)


def load_modis_ndvi(max_reliability=1):
    """MODIS NDVI with pixels above `max_reliability` set to NaN (default: keep good and marginal)."""
    ds = load_modis()
    return ds["ndvi"].where(ds["reliability"] <= max_reliability)


# ---------------------------------------------------------------- ERA5-Land precipitation (for SPI)
ERA5_FOLDER = {"monthly": "era5land", "daily": "era5land_daily"}


def era5_available(freq):
    """True if the combined "monthly" or "daily" ERA5-Land file has been downloaded."""
    return any((RAW_DIR / ERA5_FOLDER[freq]).glob(f"era5land_precip_{freq}_mm_*.nc"))


def load_era5_precip(freq="monthly"):
    """ERA5-Land precipitation over the box, in mm per month ("monthly") or per day ("daily")."""
    files = sorted((RAW_DIR / ERA5_FOLDER[freq]).glob(f"era5land_precip_{freq}_mm_*.nc"))
    if not files:
        raise FileNotFoundError(f"No {freq} ERA5-Land file in data/raw/{ERA5_FOLDER[freq]}: run its download first.")
    da = xr.open_dataset(files[-1])["precip_mm"].rename(latitude="lat", longitude="lon")
    return da.assign_coords(lat=snap(da.lat, 0.1), lon=snap(da.lon, 0.1))


# ---------------------------------------------------------------- WorldCover (cropland)
def load_worldcover(bbox=DEFAULT_BBOX, margin=0.05):
    """WorldCover land-cover classes (10 m, uint8; 0 = no data) for the box plus a margin.

    The margin makes sure the outer cells of the 0.05 deg grid, which reach 0.025 deg
    beyond the box, are fully covered.
    """
    from rioxarray.merge import merge_arrays
    lon_min, lat_min, lon_max, lat_max = bbox
    tiles = [rioxarray.open_rasterio(f, mask_and_scale=False).squeeze("band", drop=True)
             .rio.clip_box(lon_min - margin, lat_min - margin, lon_max + margin, lat_max + margin)
             for f in sorted((RAW_DIR / "worldcover").glob("ESA_WorldCover_10m_*_Map.tif"))]
    return clean_raster(merge_arrays(tiles))


# ---------------------------------------------------------------- EDO CDI (validation)
def load_edo_cdi(bbox=DEFAULT_BBOX, margin=0.05):
    """EDO Combined Drought Indicator for the box plus a margin: one class map per dekad.

    1/24 deg pixels; `time` is the first day of each dekad. Classes as in
    config.CDI_CLASSES; 7 (no data) and 8 (outside the model domain) become NaN. The
    margin keeps the pixels under the outer cells of the common grid.
    """
    files = sorted((RAW_DIR / "edo_cdi").glob("cdinx_m_edo_*.nc"))  # one Europe-wide file per year
    if not files:
        raise FileNotFoundError("No EDO CDI files in data/raw/edo_cdi: run its download first.")
    lon_min, lat_min, lon_max, lat_max = bbox
    years = []
    for f in files:
        with xr.open_dataset(f) as ds:
            cdi = ds["cdinx"].squeeze("band", drop=True)
            years.append(cdi.sel(lat=slice(lat_max + margin, lat_min - margin),
                                 lon=slice(lon_min - margin, lon_max + margin)).load())
    cdi = xr.concat(years, dim="time")
    cdi = cdi.where(cdi <= max(CDI_CLASSES)).astype("float32")
    cdi.attrs = {"long_name": "EDO Combined Drought Indicator class",
                 "classes": "; ".join(f"{k} {v}" for k, v in CDI_CLASSES.items())}
    return cdi.rename("cdi")


# ---------------------------------------------------------------- ECOSTRESS (validation)
def ecostress_files():
    """One row per ECOSTRESS GeoTIFF: product, layer, time (UTC), UTM tile, path."""
    rows = []
    for f in (RAW_DIR / "ecostress").glob("[0-9]*/ECO_*/*.tif"):
        product, layer = re.match(r"(ECO_L\w+?\.002)_(\w+?)_doy", f.name).groups()
        rows.append({"product": product, "layer": layer, "time": parse_doy(f.name),
                     "tile": f.stem.rsplit("_", 1)[-1], "path": f})
    return pd.DataFrame(rows).sort_values("time", ignore_index=True)


def ecostress_cloud_stats():
    """Clear and cloudy pixel counts per scene, from the AppEEARS cloud-layer statistics."""
    csvs = (RAW_DIR / "ecostress").glob("*/ECO-L2T-LSTE-002-cloud-Statistics-QA.csv")
    df = pd.concat(pd.read_csv(f) for f in csvs).rename(columns={"0": "clear", "1": "cloudy", "UTM Zone": "tile"})
    df["time"] = pd.to_datetime(df["Date"]).dt.tz_localize(None)
    df["clear_share"] = df["clear"] / (df["clear"] + df["cloudy"])
    return df[["time", "tile", "clear", "cloudy", "clear_share"]].sort_values("time", ignore_index=True)


def ecostress_cloud_per_cell(rebuild=False):
    """Clear and cloudy ECOSTRESS pixels per scene and cell of the common 0.05 deg grid.

    Returns a Dataset with dimensions (scene, lat, lon): `clear` and `cloudy` pixel counts,
    and `pixels`, the scene's pixels inside the cell whether observed or not (scenes often
    cover only part of the box). `time` and `tile` label each scene. A 70 m pixel belongs
    to the cell its centre falls in. Reading every cloud layer takes a minute or two, so
    the result is cached.
    """
    def build():
        target = target_grid()
        n_lat, n_lon = target.sizes["lat"], target.sizes["lon"]
        scenes = ecostress_files().query("layer == 'cloud'").reset_index(drop=True)
        counts = np.zeros((len(scenes), n_lat * n_lon, 3), dtype="int32")
        cell_of = {}  # many scenes share a pixel grid: the cell of every pixel, per grid
        for i, f in enumerate(scenes.itertuples()):
            cloud = read_geotiff(f.path, raw=True)  # 0 clear, 1 cloudy, 255 outside the scene
            key = (cloud.shape, float(cloud.lat[0]), float(cloud.lon[0]))
            if key not in cell_of:
                i_lat = pd.Index(target.lat.values).get_indexer(nearest_centre(cloud.lat.values, target.lat.values))
                i_lon = pd.Index(target.lon.values).get_indexer(nearest_centre(cloud.lon.values, target.lon.values))
                cell = i_lat[:, None] * n_lon + i_lon[None, :]
                cell_of[key] = np.where((i_lat[:, None] >= 0) & (i_lon[None, :] >= 0), cell, -1).ravel()
            cell = cell_of[key]
            state = np.minimum(cloud.values.ravel(), 2)  # 0 clear, 1 cloudy, 2 not observed
            inside = cell >= 0
            counts[i] = np.bincount(cell[inside] * 3 + state[inside], minlength=n_lat * n_lon * 3).reshape(-1, 3)
        counts = counts.reshape(len(scenes), n_lat, n_lon, 3)
        dims = ("scene", "lat", "lon")
        return xr.Dataset({"clear": (dims, counts[..., 0]), "cloudy": (dims, counts[..., 1]),
                           "pixels": (dims, counts.sum(axis=-1, dtype="int32"))},
                          coords={"lat": target.lat, "lon": target.lon,
                                  "time": ("scene", scenes["time"].values), "tile": ("scene", scenes["tile"].values)})

    return cached(INTERIM_DIR / "ecostress_cloud_per_cell.nc", build, rebuild)


def read_ecostress_lst(lst_path):
    """One ECOSTRESS LST scene in deg C, cloudy pixels set to NaN (from the scene's cloud layer)."""
    cloud = read_geotiff(str(lst_path).replace("_LST_doy", "_cloud_doy"), raw=True)
    return (read_geotiff(Path(lst_path)) - 273.15).where(cloud == 0)
