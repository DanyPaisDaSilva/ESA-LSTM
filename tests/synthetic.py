"""
Synthetic input table for the lag-model tests, in the format of docs/PLAN.md section 3.

A few cells over several April-September seasons, one row per cell and 16-day window:
* TRI (K) and SPI-1 are random noise that drifts slowly (an AR(1) process per cell-season);
* VCI responds to TRI at a known delay: the change from window t to t+1 is
      -effect * (TRI at t - delay, minus its mean)  +  a pull back towards 50  +  noise,
  so with delay=1 the model should rely most on tri_t1;
* some TRI and VCI values are missing, as with clouds. VCI responds to the true TRI;
  only the values in the table go missing.
"""
import numpy as np
import pandas as pd

from esa_lstm.preprocess.windows import season_windows

TRI_MEAN = 8.0  # K, about the real cropland mean


def drifting_noise(rng, n, sd, persistence=0.8):
    """AR(1) series with mean 0 and standard deviation sd: each value keeps `persistence` of the last."""
    x = np.empty(n)
    x[0] = rng.normal(0, sd)
    for i in range(1, n):
        x[i] = persistence * x[i - 1] + rng.normal(0, sd * np.sqrt(1 - persistence**2))
    return x


def make_table(n_cells=4, years=range(2015, 2021), delay=1, effect=3.0, noise=1.0,
               missing_tri=0.2, missing_vci=0.05, seed=0):
    """The synthetic table: n_cells x len(years) x 12 windows."""
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_cells):
        lat, lon = 49.30, round(5.65 + 0.05 * c, 2)
        for year in years:
            starts = season_windows([year])
            n = len(starts)
            tri_anomaly = drifting_noise(rng, n + delay, sd=2.5)  # `delay` extra values before the season
            vci = np.empty(n)
            vci[0] = 50 + rng.normal(0, 10)
            for t in range(n - 1):
                change = -effect * tri_anomaly[t] + 0.3 * (50 - vci[t]) + rng.normal(0, noise)  # tri_anomaly[t] = TRI at t - delay
                vci[t + 1] = np.clip(vci[t] + change, 0, 100)
            rows.append(pd.DataFrame({
                "cell_id": f"{lat:.2f}_{lon:.2f}", "lat": lat, "lon": lon, "window_start": starts,
                "vci": np.where(rng.random(n) < missing_vci, np.nan, vci),
                "tri": np.where(rng.random(n) < missing_tri, np.nan, TRI_MEAN + tri_anomaly[delay:]),
                "spi1": drifting_noise(rng, n, sd=1.0),
            }))
    return pd.concat(rows, ignore_index=True)
