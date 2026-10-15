"""
Leave one season out (docs/PLAN.md section 7).

One round per year: train on all other years, predict the held-out year. Rows of the
held-out year with VCI missing at t-2 or t-1 are not predicted, because their VCI input is
a filled-in mean; they are still used for training in the other rounds. Every other row is
predicted exactly once, by a model that never saw its year.
"""
import pandas as pd


def predictable(X):
    """True for the rows that may be predicted: VCI known at t-2 and t-1."""
    return (X["vci_missing_t2"] == 0) & (X["vci_missing_t1"] == 0)


def rounds(X, meta):
    """(held-out year, train index, test index), one per year."""
    for year in sorted(meta["year"].unique()):
        held_out = meta["year"] == year
        yield year, meta.index[~held_out], meta.index[held_out & predictable(X)]


def predict_held_out(new_model, X, y, meta):
    """Predicted VCI change for every predictable row, named change_predicted.

    new_model() must return a new, unfitted model (for example lambda: get_model("linear")).
    """
    predicted = []
    for _, train, test in rounds(X, meta):
        model = new_model().fit(X.loc[train], y.loc[train])
        predicted.append(pd.Series(model.predict(X.loc[test]), index=test))
    return pd.concat(predicted).rename("change_predicted")
