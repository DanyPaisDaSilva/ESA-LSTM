"""Tests for the lag model (docs/PLAN.md section 9), on the synthetic table of synthetic.py."""
import joblib
import numpy as np
import pandas as pd
import pytest
from synthetic import make_table

from esa_lstm.config import ROOT
from esa_lstm.models import data, evaluate, get_model, rows, run, split
from esa_lstm.models.base import BaseModel
from esa_lstm.models.config import load_config

WINDOW = pd.Timedelta(days=16)
# every model that has a config in configs/
MODEL_NAMES = sorted(load_config(p)["model"] for p in (ROOT / "configs").glob("*.yaml") if p.name != "base.yaml")


# ---------------------------------------------------------------- data
def test_data():
    table = make_table()
    assert data.check_table(table) is table  # a valid table passes unchanged

    with pytest.raises(ValueError, match="missing column"):
        data.check_table(table.drop(columns="tri"))
    with pytest.raises(ValueError, match="should be float"):
        data.check_table(table.astype({"vci": str}))
    with pytest.raises(ValueError, match="1 duplicate"):
        data.check_table(pd.concat([table, table.iloc[[5]]]))
    bad = table.copy()
    bad.loc[[0, 1], "vci"] = [-1.0, 100.5]
    with pytest.raises(ValueError, match="2 VCI values outside"):
        data.check_table(bad)
    bad = table.copy()
    bad.loc[0, "window_start"] = pd.Timestamp("2015-03-22")
    with pytest.raises(ValueError, match="1 windows start outside"):
        data.check_table(bad)


# ---------------------------------------------------------------- rows
def test_rows_alignment():
    table = make_table()
    X, y, meta = rows.build_rows(table)
    assert len(X) > 0 and X.index.equals(y.index) and X.index.equals(meta.index)
    value = table.set_index(["cell_id", "window_start"])
    for i in X.index[::7]:  # a sample of rows
        cell, target = meta.at[i, "cell_id"], meta.at[i, "target_date"]
        t = target - WINDOW  # windows are 16 days apart within a year
        for suffix, back in {"t0": 0, "t1": 1, "t2": 2}.items():
            for v in ("vci", "tri", "spi1"):
                np.testing.assert_equal(X.at[i, f"{v}_{suffix}"], value.at[(cell, t - back * WINDOW), v])
        assert y[i] == pytest.approx(value.at[(cell, target), "vci"] - value.at[(cell, t), "vci"])
        assert X.at[i, "tri_missing_t1"] == int(np.isnan(X.at[i, "tri_t1"]))
        assert X.at[i, "vci_missing_t2"] == int(np.isnan(X.at[i, "vci_t2"]))
    assert y.notna().all() and X["vci_t0"].notna().all()  # a missing VCI at t means no target


def test_rows_seasons():
    _, _, meta = rows.build_rows(make_table(missing_vci=0))
    first_input = meta["target_date"] - 3 * WINDOW  # window t-2
    assert (first_input.dt.year == meta["year"]).all()
    assert (meta["target_date"].dt.year == meta["year"]).all()
    assert (first_input.dt.month >= 4).all()  # never reaches back into the previous September
    # 12 windows per season give 9 rows (t = 3rd ... 11th window)
    assert (meta.groupby(["cell_id", "year"]).size() == 9).all()


def test_no_future():
    table = make_table()
    X, _, meta = rows.build_rows(table)
    i = X.index[len(X) // 2]
    cell, t = meta.at[i, "cell_id"], meta.at[i, "target_date"] - WINDOW
    changed = table.copy()
    later = (changed["cell_id"] == cell) & (changed["window_start"] > t)
    changed.loc[later, ["vci", "tri", "spi1"]] += 10.0
    X2, _, meta2 = rows.build_rows(changed)
    row = meta2.index[(meta2["cell_id"] == cell) & (meta2["target_date"] == t + WINDOW)][0]
    pd.testing.assert_series_equal(X.loc[i], X2.loc[row])


# ---------------------------------------------------------------- split
def test_split():
    X, y, meta = rows.build_rows(make_table(missing_vci=0.1))
    eligible = split.predictable(X)
    assert (~eligible).sum() > 10  # some rows have a missing VCI input

    for year, train, test in split.rounds(X, meta):
        assert set(train) == set(meta.index[meta["year"] != year])  # all other years, flagged rows too
        assert set(test) == set(meta.index[(meta["year"] == year) & eligible])

    class SeenYear(BaseModel):  # predicts 1 for a row whose year was in its training data
        def fit(self, X, y):
            self.years = set(meta.loc[X.index, "year"])
            return self

        def predict(self, X):
            return meta.loc[X.index, "year"].isin(self.years).astype(float).to_numpy()

        def importance(self, X, y):
            return pd.Series(0.0, index=X.columns)

    predicted = split.predict_held_out(SeenYear, X, y, meta)
    assert (predicted == 0).all()  # no model saw the year it predicted
    assert predicted.index.is_unique and set(predicted.index) == set(X.index[eligible])


# ---------------------------------------------------------------- models
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_models(name):
    X, y, _ = rows.build_rows(make_table(missing_tri=0.4, missing_vci=0.2))
    assert X["tri_t1"].isna().any() and X["vci_t1"].isna().any()  # the models must cope with gaps
    model = get_model(name).fit(X, y)
    predicted = model.predict(X)
    assert predicted.shape == (len(X),) and np.isfinite(predicted).all()
    importance = model.importance(X, y)
    assert list(importance.index) == list(X.columns) and np.isfinite(importance).all()


@pytest.mark.parametrize("delay", [0, 1, 2])
def test_recovers_delay(delay):
    X, y, meta = rows.build_rows(make_table(delay=delay))
    weights = get_model("linear").fit(X, y).importance(X, y)
    assert weights[["tri_t2", "tri_t1", "tri_t0"]].abs().idxmax() == f"tri_t{delay}"  # the planted delay

    predicted = split.predict_held_out(lambda: get_model("linear"), X, y, meta)
    errors = evaluate.errors(evaluate.predictions_table(X, y, meta, predicted))
    assert errors.at["all", "model"] < errors.at["all", "baseline"]


# ---------------------------------------------------------------- run
@pytest.mark.parametrize("name", MODEL_NAMES)
def test_run(name, tmp_path):
    table = make_table()
    table.to_parquet(tmp_path / "indices.parquet")
    (tmp_path / "base.yaml").write_text(f"data: {tmp_path / 'indices.parquet'}\noutput_dir: {tmp_path / 'outputs'}\n")
    (tmp_path / f"{name}.yaml").write_text(f"model: {name}\n")
    folder = run.run(tmp_path / f"{name}.yaml")

    assert folder == tmp_path / "outputs" / name
    for file in ["model.joblib", "predictions.parquet", "errors.csv", "figures/timeseries.png",
                 "figures/skill_map.png", "figures/importance.png"]:
        assert (folder / file).is_file(), file
    assert len(list(folder.glob("figures/map_*.png"))) == 1

    predictions = pd.read_parquet(folder / "predictions.parquet")
    assert list(predictions.columns) == ["cell_id", "lat", "lon", "target_date", "held_out_year", "change_actual",
                                         "change_predicted", "vci_actual", "vci_predicted", "vci_baseline"]
    errors = pd.read_csv(folder / "errors.csv", index_col=0)
    assert list(errors.columns) == ["model", "baseline"]
    assert list(errors.index) == [str(y) for y in sorted(table["window_start"].dt.year.unique())] + ["all"]
    assert hasattr(joblib.load(folder / "model.joblib"), "predict")
