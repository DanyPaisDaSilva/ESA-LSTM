"""
The interface every lag model implements (docs/PLAN.md section 5).

run.py only ever calls fit, predict and importance, so models can be swapped by name.
Any preprocessing (e.g. filling missing TRI or VCI with the training mean) happens inside
fit, from the training rows only, and is reused by predict.
"""
from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BaseModel(ABC):
    name: str  # set by @register("<name>")

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BaseModel":
        """Train on the input columns X and the VCI change y; returns the model itself."""

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """One predicted VCI change per row of X."""

    @abstractmethod
    def importance(self, X: pd.DataFrame, y: pd.Series) -> pd.Series:
        """One value per input column: how much the model relies on it."""
