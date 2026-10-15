"""
Turn the input table into model rows (docs/PLAN.md section 6).

One row per cell and window t where t-2, t-1, t and t+1 all exist in the same cell and
the same year's season:
    X     inputs vci/tri/spi1 at t-2, t-1, t (vci_t2, ..., spi1_t0), plus missing-value
          flags for TRI at t-2, t-1, t and for VCI at t-2, t-1 (1 = missing)
    y     the VCI change from t to t+1; rows without it are dropped (this includes every
          row with VCI missing at t, so VCI at t needs no flag)
    meta  cell_id, lat, lon, year, target_date (the start of window t+1)
Values are never taken from another cell or another year.
"""
import pandas as pd

from esa_lstm.preprocess.windows import WINDOW_DAYS

VARIABLES = ["vci", "tri", "spi1"]
LAGS = {"t2": 2, "t1": 1, "t0": 0}  # column suffix -> windows before t
FLAGS = {"tri": ["t2", "t1", "t0"], "vci": ["t2", "t1"]}


def input_columns():
    """The columns of X, in order: inputs, then flags."""
    inputs = [f"{v}_{s}" for v in VARIABLES for s in LAGS]
    flags = [f"{v}_missing_{s}" for v, suffixes in FLAGS.items() for s in suffixes]
    return inputs + flags


def build_rows(table):
    """X, y and meta (see the module docstring), all with the same index."""
    df = table.sort_values(["cell_id", "window_start"], ignore_index=True)
    df["year"] = df["window_start"].dt.year
    df["window"] = (df["window_start"].dt.dayofyear - 1) // WINDOW_DAYS  # 0 = the year's first window
    group = df.groupby(["cell_id", "year"])

    # A row needs windows t-2, t-1 and t+1 in the same cell and year, with none skipped
    exists = pd.Series(True, index=df.index)
    for lag in (2, 1, -1):
        exists &= group["window"].shift(lag) == df["window"] - lag

    X = pd.DataFrame({f"{v}_{s}": group[v].shift(lag) for v in VARIABLES for s, lag in LAGS.items()})
    for v, suffixes in FLAGS.items():
        for s in suffixes:
            X[f"{v}_missing_{s}"] = X[f"{v}_{s}"].isna().astype(int)
    y = (group["vci"].shift(-1) - df["vci"]).rename("vci_change")
    meta = df[["cell_id", "lat", "lon", "year"]].assign(target_date=group["window_start"].shift(-1))

    keep = exists & y.notna()
    return X.loc[keep, input_columns()], y[keep], meta[keep]
