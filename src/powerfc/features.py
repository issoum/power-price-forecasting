"""Feature matrix for day-ahead price forecasting.

Every feature for delivery day D must be known at D-1 12:00 local time:
  - prices of D-1 (published after the D-2 auction) and earlier days,
  - weather forecasts issued before gate closure (see data.select_available_forecast),
  - calendar information.
"""
import holidays as hol
import numpy as np
import pandas as pd

from .config import WEATHER_VARS, ZONES


def build(df, zone):
    tz = ZONES[zone]["tz"]
    local = df.index.tz_convert(tz)
    X = pd.DataFrame(index=df.index)

    # Same hour on previous days (all fully known before D-1 gate closure)
    X["price_lag24"] = df["price"].shift(24)
    X["price_lag48"] = df["price"].shift(48)
    X["price_lag168"] = df["price"].shift(168)

    # D-1 daily statistics, broadcast to every hour of D
    day = pd.Series(local.date, index=df.index)
    daily = df["price"].groupby(day.values).agg(["mean", "min", "max"])
    prev = daily.shift(1)
    for c in ["mean", "min", "max"]:
        X[f"d1_price_{c}"] = day.map(prev[c]).astype(float)

    for v in WEATHER_VARS:
        X[v] = df[v]
    # Wind power scales roughly with the cube of wind speed below rated speed
    X["wind_cubed"] = df["wind_speed_100m"].clip(upper=45) ** 3 / 1e3

    # Calendar
    X["hour"] = local.hour
    X["dow"] = local.dayofweek
    X["month"] = local.month
    years = range(local.year.min(), local.year.max() + 1)
    hd = hol.country_holidays(ZONES[zone]["holidays"], years=years)
    X["holiday"] = np.array([d in hd for d in local.date], dtype=int)
    X["weekend"] = (local.dayofweek >= 5).astype(int)
    for k in (1, 2):
        X[f"hour_sin{k}"] = np.sin(2 * np.pi * k * local.hour / 24)
        X[f"hour_cos{k}"] = np.cos(2 * np.pi * k * local.hour / 24)

    X["delivery_day"] = pd.to_datetime(day.values)
    return X
