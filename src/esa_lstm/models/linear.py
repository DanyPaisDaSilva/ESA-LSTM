"""
Linear regression (docs/PLAN.md section 5).

Inside fit: fill missing values with the training mean, scale each input to mean 0 and
standard deviation 1, then fit LinearRegression. Because the inputs are scaled, the weights
are comparable across variables, and they are the importance.
"""
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from esa_lstm.models import register
from esa_lstm.models.base import BaseModel


@register("linear")
class Linear(BaseModel):
    def __init__(self, **params):  # params (from the config) go to LinearRegression
        self.pipeline = make_pipeline(
            SimpleImputer(strategy="mean", keep_empty_features=True),  # keep a column even if all missing
            StandardScaler(),
            LinearRegression(**params),
        )

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        return self

    def predict(self, X):
        return self.pipeline.predict(X)

    def importance(self, X, y):
        """The weight of each scaled input: VCI change per standard deviation of that input."""
        return pd.Series(self.pipeline[-1].coef_, index=self.pipeline.feature_names_in_,
                         name="weight (VCI change per standard deviation of the input)")
