"""
Run the lag model from one config: uv run python -m esa_lstm.models.run --config configs/linear.yaml

Steps (docs/PLAN.md sections 7-8), writing everything to <output_dir>/<model>/:
1. read and check the input table, build the rows;
2. leave one season out: predict each year with a model trained on the other years;
3. errors for the model and the no-change baseline, per held-out year and overall;
4. train once more on all years, save that model and its importance;
5. figures.
"""
import argparse

import joblib

from esa_lstm.models import data, evaluate, get_model, plots, rows, split
from esa_lstm.models.config import load_config


def run(config_path):
    config = load_config(config_path)
    folder = config["output_dir"] / config["model"]
    folder.mkdir(parents=True, exist_ok=True)

    def new_model():
        return get_model(config["model"], **config["params"])

    X, y, meta = rows.build_rows(data.load_table(config["data"]))
    predictions = evaluate.predictions_table(X, y, meta, split.predict_held_out(new_model, X, y, meta))
    errors = evaluate.errors(predictions)
    model = new_model().fit(X, y)

    joblib.dump(model, folder / "model.joblib")
    predictions.to_parquet(folder / "predictions.parquet", index=False)
    errors.to_csv(folder / "errors.csv")
    plots.all_figures(predictions, model.importance(X, y), config["model"], folder / "figures")

    print(f"{config['model']}: {len(predictions)} of {len(X)} rows predicted; written to {folder}")
    print("RMSE (VCI points):")
    print(errors.round(2).to_string())
    return folder


def main():
    parser = argparse.ArgumentParser(description="Run the lag model from one config.")
    parser.add_argument("--config", required=True, help="a model config, e.g. configs/linear.yaml")
    run(parser.parse_args().config)


if __name__ == "__main__":
    main()
