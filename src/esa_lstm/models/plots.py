"""
The lag model's figures (docs/PLAN.md section 8), drawn from predictions.parquet.

No cells or dates are picked by hand: the figures cover every predicted cell, and the map
shows the target window with the lowest cropland-mean actual VCI. Each figure is saved to
`path`, or shown when no path is given (notebook 05).
Colours follow esa_lstm.plotting: VCI on red (low) to blue (high), errors on the blue ramp.
"""
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from esa_lstm import plotting
from esa_lstm.config import TARGET_RES
from esa_lstm.models.evaluate import rmse
from esa_lstm.preprocess.grid import target_grid

LINES = {"vci_actual": ("actual", plotting.INK), "vci_predicted": ("predicted", plotting.SERIES[0]),
         "vci_baseline": ("no change", plotting.MUTED)}


def on_grid(per_cell):
    """A Series indexed by (lat, lon) as a map on the common grid (NaN where there is no value)."""
    target = target_grid()
    da = per_cell.to_xarray()
    return da.reindex(lat=target.lat, lon=target.lon, method="nearest", tolerance=TARGET_RES / 4)


def save(fig, path):
    """Write the figure to `path`; without a path (in a notebook), show it instead."""
    if path is None:
        plt.show()
        return
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)


def timeseries(predictions, path=None):
    """Cropland-mean VCI per target window, actual vs predicted vs no change, one panel per season."""
    means = predictions.groupby(["held_out_year", "target_date"])[list(LINES)].mean()
    years = means.index.unique("held_out_year")
    columns = min(4, len(years))
    rows = -(-len(years) // columns)
    fig, axes = plt.subplots(rows, columns, figsize=(3.2 * columns, 2.6 * rows), sharey=True, squeeze=False)
    for ax, year in zip(axes.flat, years):
        season = means.loc[year]
        for column, (label, colour) in LINES.items():
            ax.plot(season.index, season[column], marker="o", ms=3, color=colour, label=label)
        ax.set_title(str(year))
        ax.set_ylim(0, 100)
        ax.xaxis.set_major_locator(mdates.MonthLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for ax in axes.flat[len(years):]:
        ax.set_visible(False)
    for ax in axes[:, 0]:
        ax.set_ylabel("VCI (cropland mean)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="lower center", ncols=len(LINES),
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout()
    save(fig, path)


def skill_map(predictions, path=None):
    """RMSE per cell over all its predictions, model and no change side by side, same colour scale."""
    per_cell = predictions.groupby(["lat", "lon"])
    maps = {name: on_grid(per_cell.apply(lambda p, c=column: rmse(p["vci_actual"], p[c])))
            for name, column in {"model": "vci_predicted", "no change": "vci_baseline"}.items()}
    vmax = max(float(m.max()) for m in maps.values())
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, (name, m) in zip(axes, maps.items()):
        plotting.plot_map(ax, m, vmin=0, vmax=vmax, label="RMSE (VCI points)")
        ax.set_title(f"{name}: RMSE per cell")
    save(fig, path)


def driest_date(predictions):
    """The target window with the lowest cropland-mean actual VCI."""
    return predictions.groupby("target_date")["vci_actual"].mean().idxmin()


def date_map(predictions, date, path=None):
    """Actual vs predicted VCI per cell for one target window."""
    day = predictions[predictions["target_date"] == date].set_index(["lat", "lon"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8))
    for ax, column in zip(axes, ["vci_actual", "vci_predicted"]):
        plotting.plot_map(ax, on_grid(day[column]), cmap=plotting.DIVERGING, vmin=0, vmax=100, label="VCI")
        ax.set_title(f"{LINES[column][0]} VCI, {date:%Y-%m-%d}")
    save(fig, path)


def importance(values, model_name, path=None):
    """One bar per input column, in input order (top to bottom); the axis label is the Series name."""
    fig, ax = plt.subplots(figsize=(5, 0.25 * len(values) + 1))
    colours = np.where(values.to_numpy() < 0, plotting.SERIES[1], plotting.SERIES[0])
    ax.barh(values.index[::-1], values.to_numpy()[::-1], color=colours[::-1])
    ax.axvline(0, color=plotting.MUTED, lw=0.8)
    ax.set_xlabel(values.name)
    ax.set_title(f"{model_name}: importance per input")
    save(fig, path)


def all_figures(predictions, importance_values, model_name, folder):
    """Write the four figures of PLAN section 8 into `folder`."""
    folder.mkdir(parents=True, exist_ok=True)
    plotting.use_style()
    timeseries(predictions, folder / "timeseries.png")
    skill_map(predictions, folder / "skill_map.png")
    date = pd.Timestamp(driest_date(predictions))
    date_map(predictions, date, folder / f"map_{date:%Y-%m-%d}.png")
    importance(importance_values, model_name, folder / "importance.png")
