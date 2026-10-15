"""
Predictions table and errors, for the model and the no-change baseline (docs/PLAN.md sections 7-8).

The baseline predicts no change: VCI at t+1 = VCI at t. Because VCI at t is known, the
error on the predicted change equals the error on the predicted VCI, so all errors are in
VCI points. Model and baseline are always scored on the same rows.
"""
import numpy as np
import pandas as pd


def predictions_table(X, y, meta, change_predicted):
    """One row per predicted row, with the columns of predictions.parquet."""
    rows = change_predicted.index
    vci_t = X.loc[rows, "vci_t0"]
    m = meta.loc[rows]
    return pd.DataFrame({
        "cell_id": m["cell_id"], "lat": m["lat"], "lon": m["lon"], "target_date": m["target_date"],
        "held_out_year": m["year"],
        "change_actual": y.loc[rows], "change_predicted": change_predicted,
        "vci_actual": vci_t + y.loc[rows], "vci_predicted": vci_t + change_predicted,
        "vci_baseline": vci_t,
    })


def rmse(actual, predicted):
    return float(np.sqrt(np.mean((actual - predicted) ** 2)))


def errors(predictions):
    """RMSE in VCI points for the model and the baseline, per held-out year and overall ("all")."""
    def both(p):
        return pd.Series({"model": rmse(p["vci_actual"], p["vci_predicted"]),
                          "baseline": rmse(p["vci_actual"], p["vci_baseline"])})

    table = {str(year): both(p) for year, p in predictions.groupby("held_out_year")}
    table["all"] = both(predictions)
    return pd.DataFrame(table).T.rename_axis("held_out_year")
