"""
Run settings of the lag model, read from the YAML files in configs/.

A model config (e.g. configs/linear.yaml) names the model and its parameters; the settings
every run shares (input table, output folder) are in base.yaml in the same folder.
Project-wide settings (paths, season, thresholds) stay in esa_lstm/config.py.
"""
from pathlib import Path

import yaml

from esa_lstm.config import ROOT

REQUIRED = ("data", "output_dir", "model")


def load_config(path):
    """base.yaml, then the model config on top of it. Paths are made absolute (relative to the project root)."""
    path = Path(path)
    config = yaml.safe_load((path.parent / "base.yaml").read_text()) or {}
    config |= yaml.safe_load(path.read_text()) or {}
    missing = [key for key in REQUIRED if key not in config]
    if missing:
        raise ValueError(f"{path}: missing setting(s) {missing}")
    config.setdefault("params", {})
    for key in ("data", "output_dir"):
        config[key] = ROOT / config[key]  # an absolute path stays as it is
    return config
