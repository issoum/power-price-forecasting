"""Battery-arbitrage backtest driven by the price forecasts.

Each delivery day a 1 MW / 1 MWh battery, empty at the start of the day, buys 1 MWh in one
hour and sells in a later hour, both through the day-ahead auction. The hours are chosen on
the forecast; P&L is settled on realised prices. Assumptions: price-taker (no market impact),
round-trip efficiency applied on the sell leg, no degradation cost.
"""
import numpy as np
import pandas as pd


def best_pair(prices, efficiency):
    """Charge hour i < discharge hour j maximising efficiency * p[j] - p[i]."""
    p = np.asarray(prices, dtype=float)
    best, pair = 0.0, None
    running_min, i_min = np.inf, None
    for j in range(len(p)):
        if i_min is not None:
            gain = efficiency * p[j] - running_min
            if gain > best:
                best, pair = gain, (i_min, j)
        if p[j] < running_min:
            running_min, i_min = p[j], j
    return pair


def run(actual, forecast, days, efficiency=0.85):
    """Daily P&L (EUR) of the strategy; trades only when the forecast spread is profitable."""
    df = pd.DataFrame({"actual": actual, "forecast": forecast, "day": days.values}).dropna()
    pnl = {}
    for day, g in df.groupby("day"):
        pair = best_pair(g["forecast"], efficiency)
        if pair is None:
            pnl[day] = 0.0
            continue
        i, j = pair
        pnl[day] = efficiency * g["actual"].iloc[j] - g["actual"].iloc[i]
    return pd.Series(pnl).sort_index()


def summary(pnl, perfect):
    return {
        "total_pnl_eur": pnl.sum(),
        "capture_rate_%": 100 * pnl.sum() / perfect.sum(),
        "hit_rate_%": 100 * (pnl[pnl != 0] > 0).mean(),
        "worst_day_eur": pnl.min(),
        "days": len(pnl),
    }
