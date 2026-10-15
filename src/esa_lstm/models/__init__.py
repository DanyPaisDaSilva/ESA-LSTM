"""
The lag model (docs/PLAN.md): predicts the change in VCI over the next 16-day window from
VCI, TRI and SPI-1 at the three windows before, per cropland cell.

Models are registered by name and created only through get_model(), with the name from a
config file. Adding a model = one new file models/<name>.py whose class carries
@register("<name>"), plus one config in configs/; nothing else changes.
"""
import importlib

from esa_lstm.models.base import BaseModel

MODELS: dict[str, type[BaseModel]] = {}


def register(name):
    """Class decorator: make the model available to get_model(name)."""
    def add(cls):
        cls.name = name
        MODELS[name] = cls
        return cls
    return add


def get_model(name, **params):
    """A new, untrained model of the given name, e.g. get_model("linear")."""
    if name not in MODELS:
        importlib.import_module(f"{__name__}.{name}")  # models/<name>.py registers itself on import
    if name not in MODELS:
        raise ValueError(f"models/{name}.py does not register a model called {name!r}")
    return MODELS[name](**params)
