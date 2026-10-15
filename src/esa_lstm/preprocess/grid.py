"""
The common grid, and the two ways data are brought onto it.

The common grid is the 0.05 degree MSG LST grid cut to the study box: 23 x 31 cells,
identified by their centre coordinates (lat 50.30 ... 49.20, lon 5.30 ... 6.80).

* Finer data (MODIS 250 m, WorldCover 10 m) -> block_mean(): average of every fine
  pixel whose centre lies inside the cell, optionally weighted (e.g. by cropland share).
* Coarser data (ERA5-Land 0.1 deg) -> interpolate(): bilinear interpolation between
  the coarse cell centres. This adds no detail; it only puts values on the same cells.
"""
import numpy as np
import xarray as xr

from esa_lstm.config import CROPLAND_CLASS, DEFAULT_BBOX, TARGET_RES


def target_grid(bbox=DEFAULT_BBOX, res=TARGET_RES):
    """Empty Dataset holding the cell-centre coordinates of the common grid."""
    lon_min, lat_min, lon_max, lat_max = bbox
    lon = np.arange(round(lon_min / res), round(lon_max / res) + 1) * res
    lat = np.arange(round(lat_max / res), round(lat_min / res) - 1, -1) * res  # north to south
    return xr.Dataset(coords={"lat": np.round(lat, 6), "lon": np.round(lon, 6)})


def nearest_centre(coord, centres):
    """For each fine coordinate, the target cell centre it falls in (NaN if outside the grid)."""
    step = centres[1] - centres[0]  # negative for lat (north to south)
    i = np.round((coord - centres[0]) / step).astype(int)
    inside = (i >= 0) & (i < len(centres))
    return np.where(inside, centres[np.clip(i, 0, len(centres) - 1)], np.nan)


def block_mean(da, target, weights=None):
    """Weighted mean of all fine pixels inside each target cell.

    da:      (..., lat, lon) on a grid finer than `target`; NaN pixels are skipped.
    weights: (lat, lon) on da's grid, e.g. cropland share; default: equal weights.
    A cell with no valid weight becomes NaN.
    """
    w = da.notnull().astype("float32")  # missing pixels get no weight
    if weights is not None:
        w = w * weights

    def sum_per_cell(x):
        x = x.assign_coords(cell_lat=("lat", nearest_centre(x.lat.values, target.lat.values)),
                            cell_lon=("lon", nearest_centre(x.lon.values, target.lon.values)))
        total = x.groupby("cell_lat").sum("lat").groupby("cell_lon").sum("lon")
        return total.rename(cell_lat="lat", cell_lon="lon")

    mean = sum_per_cell(da.fillna(0) * w) / sum_per_cell(w)  # 0 / 0 -> NaN where nothing is valid
    return mean.reindex(lat=target.lat, lon=target.lon).assign_attrs(da.attrs)


def cropland_share(classes, target):
    """Share of each target cell that WorldCover marks as cropland (0-1)."""
    is_crop = (classes == CROPLAND_CLASS).astype("float32").where(classes != 0)  # 0 = no data
    share = block_mean(is_crop, target)
    return share.assign_attrs(long_name="cropland share (WorldCover)", units="1")


def interpolate(da, target):
    """Bilinear interpolation of a coarser field (e.g. ERA5 0.1 deg) onto the target cells."""
    return da.interp(lat=target.lat, lon=target.lon).assign_attrs(da.attrs)


def cell_edges(centres):
    """Cell edges from cell centres (for pcolormesh and grid-outline plots)."""
    c = np.asarray(centres, dtype="float64")
    mid = (c[1:] + c[:-1]) / 2
    return np.concatenate([[2 * c[0] - mid[0]], mid, [2 * c[-1] - mid[-1]]])
