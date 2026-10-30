"""Forecast accuracy metrics and the Diebold–Mariano test."""
import numpy as np
import pandas as pd
from scipy import stats


def metrics(y, f, benchmark):
    e = y - f
    mae = np.mean(np.abs(e))
    return {
        "MAE": mae,
        "RMSE": np.sqrt(np.mean(e ** 2)),
        # relative MAE vs the weekly naive forecast (<1 beats the benchmark)
        "rMAE": mae / np.mean(np.abs(y - benchmark)),
        "sMAPE_%": 100 * np.mean(2 * np.abs(e) / (np.abs(y) + np.abs(f)).clip(lower=1e-6)),
    }


def diebold_mariano(y, f1, f2, days):
    """DM test on daily mean absolute errors (H1: f1 more accurate than f2).

    Losses are averaged per delivery day because the 24 hourly errors of one auction are
    strongly correlated; the daily series is then treated as one-step-ahead.
    """
    d = (np.abs(y - f1) - np.abs(y - f2)).groupby(days.values).mean()
    stat = d.mean() / (d.std(ddof=1) / np.sqrt(len(d)))
    return stat, stats.norm.cdf(stat)


def table(y, forecasts, days, benchmark="naive_d7"):
    rows = {name: metrics(y, f, forecasts[benchmark]) for name, f in forecasts.items()}
    res = pd.DataFrame(rows).T
    best = res["MAE"].idxmin()
    res["DM_p_vs_best"] = [
        np.nan if n == best else diebold_mariano(y, forecasts[best], forecasts[n], days)[1]
        for n in res.index
    ]
    return res.sort_values("MAE")
