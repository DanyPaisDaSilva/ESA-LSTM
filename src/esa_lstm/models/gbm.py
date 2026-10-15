"""
Gradient boosting (docs/PLAN.md section 5).

HistGradientBoostingRegressor accepts missing values, so there is no preprocessing.
Importance: shuffle one input column at a time and measure how much the RMSE grows
(permutation importance), in VCI points.
"""
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance

from esa_lstm.models import register
from esa_lstm.models.base import BaseModel

SHUFFLES = 10  # times each column is shuffled; the importance is the mean


@register("gbm")
class GBM(BaseModel):
    def __init__(self, **params):  # params (from the config) go to HistGradientBoostingRegressor
        self.model = HistGradientBoostingRegressor(**params)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        return self.model.predict(X)

    def importance(self, X, y):
        """How much the RMSE grows, in VCI points, when the column is shuffled."""
        result = permutation_importance(self.model, X, y, scoring="neg_root_mean_squared_error",
                                        n_repeats=SHUFFLES, random_state=0)
        return pd.Series(result.importances_mean, index=X.columns,
                         name="RMSE increase when shuffled (VCI points)")
