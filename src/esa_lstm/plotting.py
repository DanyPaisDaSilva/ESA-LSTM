"""
Shared figure style for the notebooks, so every figure uses the same colours.

Colour rules:
* each dataset and each index has one fixed colour (DATASET_COLOR, INDEX_COLOR);
* magnitudes use one blue ramp (SEQUENTIAL);
* indices with a "normal" middle use red-grey-blue (DIVERGING): red = dry / stressed,
  blue = wet / healthy. TRI is the other way round (high = stressed), so it uses DIVERGING_R;
* EDO CDI classes have their own colours (CDI_COLOR): yellow-orange-red for the drought
  stages Watch-Warning-Alert, soft greens and blue for recovery.
"""
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, ListedColormap

from esa_lstm.config import CDI_CLASSES, DEFAULT_BBOX
from esa_lstm.preprocess.dekads import dekad_end
from esa_lstm.preprocess.grid import cell_edges

# ---------------------------------------------------------------- colours
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK, INK_2, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df", "#fcfcfb"

DATASET_COLOR = {"MSG LST": SERIES[0], "MODIS NDVI": SERIES[1], "ERA5-Land": SERIES[2],
                 "ECOSTRESS": SERIES[3], "WorldCover": SERIES[4], "EDO CDI": SERIES[6]}
INDEX_COLOR = {"TRI": SERIES[0], "VCI": SERIES[1], "SPI": SERIES[2]}

_BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
_REDS = ["#6e1312", "#99201f", "#c23332", "#e34948", "#e97776", "#f3a6a5"]
SEQUENTIAL = LinearSegmentedColormap.from_list("blues", _BLUES)
DIVERGING = LinearSegmentedColormap.from_list("red_grey_blue", _REDS + ["#f0efec"] + _BLUES[1:])
DIVERGING_R = DIVERGING.reversed()

# EDO CDI classes (config.CDI_CLASSES)
# (yellow-orange-red kept apart from the dataset and index colours above)
CDI_COLOR = {0: "#ebeae6", 1: "#f6d743", 2: "#f39c33", 3: "#c23332",
             4: "#7fc8a9", 5: "#a9cce3", 6: "#cfe8b8"}
CDI_SHARE = LinearSegmentedColormap.from_list(  # share of an area in drought, 0 -> 100 %
    "cdi_share", [CDI_COLOR[0], CDI_COLOR[1], CDI_COLOR[2], CDI_COLOR[3]])
DROUGHT_SHADE = "#fbd5d4"  # background of drought dekads in time series

# WorldCover v200 classes with their official colours
WORLDCOVER = {10: ("Tree cover", "#006400"), 20: ("Shrubland", "#ffbb22"), 30: ("Grassland", "#ffff4c"),
              40: ("Cropland", "#f096ff"), 50: ("Built-up", "#fa0000"), 60: ("Bare / sparse", "#b4b4b4"),
              80: ("Water", "#0064c8"), 90: ("Herbaceous wetland", "#0096a0")}


def use_style():
    mpl.rcParams.update({
        "figure.dpi": 110, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": MUTED, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
        "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
        "axes.prop_cycle": mpl.cycler(color=SERIES), "lines.linewidth": 2, "legend.frameon": False,
        "font.size": 9, "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlelocation": "left",
    })


# ---------------------------------------------------------------- maps
def map_axes(ax, bbox=DEFAULT_BBOX, pad=0.02):
    """Limit a map to the box and keep distances true (1 deg lon is shorter than 1 deg lat here)."""
    lon_min, lat_min, lon_max, lat_max = bbox
    ax.set_xlim(lon_min - pad, lon_max + pad)
    ax.set_ylim(lat_min - pad, lat_max + pad)
    ax.set_aspect(1 / np.cos(np.radians((lat_min + lat_max) / 2)))
    ax.grid(False)


def plot_map(ax, da, cmap=SEQUENTIAL, vmin=None, vmax=None, label=None, bbox=DEFAULT_BBOX):
    """A (lat, lon) field drawn cell by cell, with an optional colour bar."""
    mesh = ax.pcolormesh(cell_edges(da.lon), cell_edges(da.lat), da.values, cmap=cmap,
                         vmin=vmin, vmax=vmax, rasterized=True)
    map_axes(ax, bbox)
    if label is not None:
        bar = plt.colorbar(mesh, ax=ax, shrink=0.8, pad=0.02)
        bar.set_label(label, color=INK_2)
        bar.outline.set_visible(False)
    return mesh


def worldcover_map(ax, classes, bbox=DEFAULT_BBOX, step=1):
    """WorldCover classes in their official colours; step > 1 draws every n-th pixel (faster)."""
    colours = np.ones((256, 4))
    for code, (_, hex_colour) in WORLDCOVER.items():
        colours[code] = mpl.colors.to_rgba(hex_colour)
    sub = classes[::step, ::step]
    ax.imshow(sub.values, cmap=ListedColormap(colours), vmin=0, vmax=255, interpolation="nearest",
              extent=[float(sub.lon[0]), float(sub.lon[-1]), float(sub.lat[-1]), float(sub.lat[0])])
    map_axes(ax, bbox)


def worldcover_legend(ax, codes, **kw):
    handles = [plt.Rectangle((0, 0), 1, 1, color=WORLDCOVER[c][1]) for c in codes if c in WORLDCOVER]
    ax.legend(handles, [WORLDCOVER[c][0] for c in codes if c in WORLDCOVER], **kw)


def cdi_map(ax, cdi, bbox=DEFAULT_BBOX):
    """One EDO CDI class map, cell by cell in the class colours."""
    cmap = ListedColormap([CDI_COLOR[c] for c in CDI_CLASSES])
    ax.pcolormesh(cell_edges(cdi.lon), cell_edges(cdi.lat), cdi.values, cmap=cmap,
                  vmin=-0.5, vmax=max(CDI_CLASSES) + 0.5, rasterized=True)
    map_axes(ax, bbox)


def cdi_legend(ax, codes=tuple(CDI_CLASSES), **kw):
    handles = [plt.Rectangle((0, 0), 1, 1, color=CDI_COLOR[c]) for c in codes]
    ax.legend(handles, [CDI_CLASSES[c] for c in codes], **kw)


def draw_box(ax, bbox, **kw):
    lon_min, lat_min, lon_max, lat_max = bbox
    ax.add_patch(mpl.patches.Rectangle((lon_min, lat_min), lon_max - lon_min, lat_max - lat_min,
                                       fill=False, **kw))


def outline_cells(ax, mask, color=INK, lw=0.8):
    """Draw a box around every True cell of a (lat, lon) mask, e.g. the cropland cells."""
    res = abs(float(mask.lat[1] - mask.lat[0]))
    for lat, lon in mask.where(mask).to_series().dropna().index:
        ax.add_patch(mpl.patches.Rectangle((lon - res / 2, lat - res / 2), res, res,
                                           fill=False, edgecolor=color, lw=lw))


# ---------------------------------------------------------------- time series
def shade_droughts(ax, drought, label=None):
    """Shade the dekads where `drought` is True: a Series indexed by dekad start,
    e.g. from harmonise.drought_dekads(). `label` names the shading in a legend."""
    starts = pd.DatetimeIndex(drought.index[drought.to_numpy(dtype=bool)])
    for i, (start, end) in enumerate(zip(starts, dekad_end(starts) + pd.Timedelta(days=1))):
        ax.axvspan(start, end, color=DROUGHT_SHADE, lw=0, zorder=0, label=label if i == 0 else None)
