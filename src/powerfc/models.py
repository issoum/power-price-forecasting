"""Forecasting models sharing a fit / predict interface."""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import RidgeCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

NUMERIC = [
    "price_lag24", "price_lag48", "price_lag168", "d1_price_mean", "d1_price_min", "d1_price_max",
    "temperature_2m", "wind_speed_100m", "wind_cubed", "shortwave_radiation", "cloud_cover",
    "holiday", "weekend", "hour_sin1", "hour_cos1", "hour_sin2", "hour_cos2",
]
CATEGORICAL = ["hour", "dow", "month"]


class NaiveDaily:
    """Price of the same hour on D-1."""
    name = "naive_d1"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["price_lag24"].to_numpy()


class NaiveWeekly:
    """Price of the same hour on D-7: the standard benchmark in the EPF literature."""
    name = "naive_d7"

    def fit(self, X, y):
        return self

    def predict(self, X):
        return X["price_lag168"].to_numpy()


class Ridge:
    """Linear model on lagged prices, weather and calendar dummies (LEAR-style baseline)."""
    name = "ridge"

    def __init__(self):
        pre = ColumnTransformer([
            ("num", StandardScaler(), NUMERIC),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL),
        ])
        self.model = make_pipeline(pre, RidgeCV(alphas=np.logspace(-2, 3, 20)))

    def fit(self, X, y):
        self.model.fit(X[NUMERIC + CATEGORICAL], y)
        return self

    def predict(self, X):
        return self.model.predict(X[NUMERIC + CATEGORICAL])


class GradientBoosting:
    """Gradient-boosted trees, able to capture non-linear weather/price interactions."""
    name = "gbm"

    def __init__(self, seed=0):
        self.model = HistGradientBoostingRegressor(
            max_iter=500, learning_rate=0.05, max_leaf_nodes=31, min_samples_leaf=40,
            l2_regularization=1.0, random_state=seed,
        )

    def fit(self, X, y):
        self.model.fit(X[NUMERIC + CATEGORICAL], y)
        return self

    def predict(self, X):
        return self.model.predict(X[NUMERIC + CATEGORICAL])


ALL = [NaiveDaily, NaiveWeekly, Ridge, GradientBoosting]


def walk_forward(X, y, model_cls, test_start, refit="MS"):
    """Expanding-window forecasts: refit at the start of each period on all earlier data only."""
    X = X.dropna()
    y = y.loc[X.index]
    test_days = X.loc[X["delivery_day"] >= pd.Timestamp(test_start), "delivery_day"]
    periods = pd.date_range(pd.Timestamp(test_start), test_days.max() + pd.Timedelta(days=1), freq=refit)
    periods = periods.union([pd.Timestamp(test_start), test_days.max() + pd.Timedelta(days=1)])
    out = []
    for a, b in zip(periods[:-1], periods[1:]):
        train = X["delivery_day"] < a
        test = (X["delivery_day"] >= a) & (X["delivery_day"] < b)
        if not test.any():
            continue
        m = model_cls().fit(X[train], y[train])
        out.append(pd.Series(m.predict(X[test]), index=X.index[test]))
    return pd.concat(out).rename(model_cls.name)
