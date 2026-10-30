"""Download and cache day-ahead prices and archived weather forecasts.

Prices: Energy-Charts API (Fraunhofer ISE, data from Bundesnetzagentur | SMARD, CC BY 4.0).
Weather: Open-Meteo previous-runs API, i.e. the forecasts as they were issued one and two
days before delivery, so features never contain information unavailable at gate closure.
"""
import time

import numpy as np
import pandas as pd
import requests

from .config import DATA_DIR, GATE_CLOSURE_HOUR, WEATHER_VARS, ZONES

PRICE_URL = "https://api.energy-charts.info/price"
WEATHER_URL = "https://previous-runs-api.open-meteo.com/v1/forecast"


def _get_json(url, params, retries=6, pause=1.5):
    """GET with a polite pause between calls and backoff on rate limiting (both APIs are free)."""
    for attempt in range(retries):
        time.sleep(pause)
        try:
            r = requests.get(url, params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
        except (requests.RequestException, ValueError):
            pass
        time.sleep(15 * (attempt + 1))
    raise RuntimeError(f"request failed after {retries} attempts: {url} {params}")


def _month_chunks(start, end):
    edges = pd.date_range(pd.Timestamp(start).normalize(), pd.Timestamp(end), freq="MS")
    edges = [pd.Timestamp(start)] + [e for e in edges if e > pd.Timestamp(start)] + [pd.Timestamp(end)]
    return list(zip(edges[:-1], edges[1:]))


def fetch_prices(zone, start, end):
    """Hourly day-ahead prices (EUR/MWh), UTC index. 15-min products (since Oct 2025) are averaged."""
    parts = []
    for a, b in _month_chunks(start, end):
        d = _get_json(PRICE_URL, {"bzn": zone, "start": a.strftime("%Y-%m-%d"), "end": b.strftime("%Y-%m-%d")})
        s = pd.Series(d["price"], index=pd.to_datetime(d["unix_seconds"], unit="s", utc=True), dtype=float)
        parts.append(s)
    s = pd.concat(parts)
    s = s[~s.index.duplicated()].sort_index()
    return s.resample("h").mean().rename("price")


def fetch_weather(zone, start, end):
    """Hourly D-1 and D-2 weather forecasts averaged over the zone's points, UTC index."""
    hourly = [f"{v}_previous_day{k}" for v in WEATHER_VARS for k in (1, 2)]
    frames = []
    for name, lat, lon in ZONES[zone]["points"]:
        parts = []
        for a, b in _month_chunks(start, end):
            d = _get_json(WEATHER_URL, {
                "latitude": lat, "longitude": lon, "timezone": "UTC",
                "start_date": a.strftime("%Y-%m-%d"),
                "end_date": (b - pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                "hourly": ",".join(hourly),
            })
            h = d["hourly"]
            parts.append(pd.DataFrame({k: h[k] for k in hourly}, index=pd.to_datetime(h["time"], utc=True)))
        frames.append(pd.concat(parts).astype(float))
    w = sum(frames) / len(frames)
    return w[~w.index.duplicated()].sort_index()


def select_available_forecast(weather, tz):
    """For each delivery hour keep the most recent forecast issued before D-1 gate closure.

    `*_previous_day1` was issued ~24h before the target hour, which is after gate closure for
    afternoon hours; those hours fall back to `*_previous_day2` (issued ~48h before).
    """
    idx = weather.index
    local = idx.tz_convert(tz)
    gate = (local.normalize() - pd.Timedelta(days=1) + pd.Timedelta(hours=GATE_CLOSURE_HOUR)).tz_convert("UTC")
    day1_ok = (idx - pd.Timedelta(hours=24)) <= gate
    out = pd.DataFrame(index=idx)
    for v in WEATHER_VARS:
        out[v] = np.where(day1_ok, weather[f"{v}_previous_day1"], weather[f"{v}_previous_day2"])
    return out


def load(zone, start, end, refresh=False):
    """Cached hourly frame with price and gate-closure-available weather forecasts."""
    DATA_DIR.mkdir(exist_ok=True)

    def cached(kind, fetch):
        path = DATA_DIR / f"{kind}_{zone}_{start}_{end}.parquet"
        if path.exists() and not refresh:
            return pd.read_parquet(path)
        out = fetch(zone, start, end).to_frame() if kind == "prices" else fetch(zone, start, end)
        out.to_parquet(path)
        return out

    prices = cached("prices", fetch_prices)["price"]
    weather = select_available_forecast(cached("weather", fetch_weather), ZONES[zone]["tz"])
    return weather.join(prices, how="inner")
