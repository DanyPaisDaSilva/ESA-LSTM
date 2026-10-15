"""
Read the lag model's input table (docs/PLAN.md section 3) and check it before anything uses it.

The table is built by esa_lstm.preprocess.windows. Any problem stops the run with a message
that lists everything wrong at once.
"""
import pandas as pd
from pandas.api import types

from esa_lstm.config import GROWING_SEASON_END_MONTH, GROWING_SEASON_START_MONTH

# column -> (type check, type name for the message)
COLUMNS = {
    "cell_id": (types.is_string_dtype, "str"),
    "lat": (types.is_float_dtype, "float"),
    "lon": (types.is_float_dtype, "float"),
    "window_start": (types.is_datetime64_any_dtype, "date"),
    "vci": (types.is_float_dtype, "float"),
    "tri": (types.is_float_dtype, "float"),
    "spi1": (types.is_float_dtype, "float"),
}


def load_table(path):
    return check_table(pd.read_parquet(path))


def check_table(table):
    """Return the table unchanged if it is valid; otherwise raise ValueError listing the problems."""
    missing = [c for c in COLUMNS if c not in table]
    if missing:
        raise ValueError(f"Input table: missing column(s) {missing}")
    problems = [f"column {c!r} should be {name}, is {table[c].dtype}"
                for c, (is_type, name) in COLUMNS.items() if not is_type(table[c])]
    if problems:  # the checks below need the right types
        raise ValueError("Input table:\n  " + "\n  ".join(problems))

    duplicates = table.duplicated(["cell_id", "window_start"]).sum()
    if duplicates:
        problems.append(f"{duplicates} duplicate (cell_id, window_start) pairs")
    outside = ((table["vci"] < 0) | (table["vci"] > 100)).sum()  # NaN (a gap) is allowed
    if outside:
        problems.append(f"{outside} VCI values outside [0, 100]")
    month = table["window_start"].dt.month
    off_season = ((month < GROWING_SEASON_START_MONTH) | (month > GROWING_SEASON_END_MONTH)).sum()
    if off_season:
        problems.append(f"{off_season} windows start outside months "
                        f"{GROWING_SEASON_START_MONTH}-{GROWING_SEASON_END_MONTH}")
    if problems:
        raise ValueError("Input table:\n  " + "\n  ".join(problems))
    return table
