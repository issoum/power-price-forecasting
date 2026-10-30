import numpy as np
import pandas as pd

from powerfc import backtest, data, features, models
from powerfc.config import WEATHER_VARS


def _synthetic(days=40, tz="Europe/Berlin"):
    idx = pd.date_range("2025-03-01", periods=24 * days, freq="h", tz="UTC")
    rng = np.random.default_rng(0)
    df = pd.DataFrame({v: rng.normal(size=len(idx)) for v in WEATHER_VARS}, index=idx)
    df["price"] = 80 + 20 * np.sin(2 * np.pi * idx.hour / 24) + rng.normal(size=len(idx))
    return df


def test_weather_uses_forecast_issued_before_gate_closure():
    idx = pd.date_range("2025-06-10", periods=48, freq="h", tz="UTC")
    w = pd.DataFrame(index=idx)
    for v in WEATHER_VARS:
        w[f"{v}_previous_day1"] = 1.0
        w[f"{v}_previous_day2"] = 2.0
    out = data.select_available_forecast(w, "Europe/Berlin")
    local = idx.tz_convert("Europe/Berlin")
    gate = (local.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=12)).tz_convert("UTC")
    issued_d1 = idx - pd.Timedelta(hours=24)
    used_d1 = out["temperature_2m"] == 1.0
    assert (issued_d1[used_d1] <= gate[used_d1]).all()
    # early-morning hours can use D-1 forecasts, afternoon hours must fall back to D-2
    assert used_d1.any() and (~used_d1).any()


def test_price_features_only_use_previous_days():
    df = _synthetic()
    X = features.build(df, "DE-LU")
    day = X["delivery_day"]
    # perturb every price of one delivery day: none of that day's features may change
    target = day.iloc[24 * 20]
    df2 = df.copy()
    df2.loc[(day == target).values, "price"] += 1000
    X2 = features.build(df2, "DE-LU")
    same_day = (day == target).values
    cols = [c for c in X.columns if c != "delivery_day"]
    pd.testing.assert_frame_equal(X.loc[same_day, cols], X2.loc[same_day, cols])


def test_walk_forward_never_trains_on_test_period():
    df = _synthetic()
    X = features.build(df, "DE-LU")
    seen = []

    class Spy(models.NaiveDaily):
        name = "spy"

        def fit(self, X, y):
            seen.append(X["delivery_day"].max())
            return self

    fc = models.walk_forward(X, df["price"], Spy, "2025-03-25", refit="W-MON")
    first_test_day = X.loc[fc.index, "delivery_day"].min()
    assert seen and seen[0] < first_test_day


def test_best_pair_respects_charge_before_discharge():
    assert backtest.best_pair([50, 10, 80, 5], efficiency=1.0) == (1, 2)
    assert backtest.best_pair([90, 80, 70, 60], efficiency=1.0) is None


def test_cycle_cost_filters_small_spreads():
    # forecast spread 0.85 * 20 - 10 = 7 EUR: profitable without cost, skipped with a 10 EUR cycle cost
    assert backtest.best_pair([10, 20], efficiency=0.85) == (0, 1)
    assert backtest.best_pair([10, 20], efficiency=0.85, cycle_cost=10) is None
