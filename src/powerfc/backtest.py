"""Battery-arbitrage backtest driven by the price forecasts.

Each delivery day a 1 MW / 1 MWh battery, empty at the start of the day, buys 1 MWh in one
hour and sells in a later hour, both through the day-ahead auction. The hours are chosen on
the forecast and the trade is only placed if the forecast spread covers losses and the cycle
cost; P&L is settled on realised prices.

Assumptions: price-taker (orders are price-independent, so always filled at the clearing
price), round-trip efficiency applied on the sell leg, a cycle cost per MWh discharged
standing for battery degradation and fees, no other costs.

Most of the P&L is the value of owning the battery (the structural intraday spread), not of
the forecast: the forecast's contribution is the P&L in excess of a naive forecast.
"""
import numpy as np
import pandas as pd


def best_pair(prices, efficiency, cycle_cost=0.0):
    """Charge hour i < discharge hour j maximising efficiency * p[j] - p[i] - cycle_cost.

    Returns None when no pair has a positive expected gain.
    """
    p = np.asarray(prices, dtype=float)
    best, pair = 0.0, None
    running_min, i_min = np.inf, None
    for j in range(len(p)):
        if i_min is not None:
            gain = efficiency * p[j] - running_min - cycle_cost
            if gain > best:
                best, pair = gain, (i_min, j)
        if p[j] < running_min:
            running_min, i_min = p[j], j
    return pair


def run(actual, forecast, days, efficiency=0.85, cycle_cost=0.0):
    """Daily P&L (EUR) of the strategy, net of the cycle cost on traded days."""
    df = pd.DataFrame({"actual": actual, "forecast": forecast, "day": days.values}).dropna()
    pnl = {}
    for day, g in df.groupby("day"):
        pair = best_pair(g["forecast"], efficiency, cycle_cost)
        if pair is None:
            pnl[day] = 0.0
            continue
        i, j = pair
        pnl[day] = efficiency * g["actual"].iloc[j] - g["actual"].iloc[i] - cycle_cost
    return pd.Series(pnl).sort_index()


def summary(pnl, perfect, naive):
    years = len(pnl) / 365
    traded = pnl[pnl != 0]
    return {
        "pnl_eur": pnl.sum(),
        "pnl_eur_per_mw_year": pnl.sum() / years,
        "capture_rate_%": 100 * pnl.sum() / perfect.sum(),
        # value of the forecast itself: P&L in excess of the naive D-7 forecast
        "excess_vs_naive_eur_per_mw_year": (pnl.sum() - naive.sum()) / years,
        "excess_vs_naive_%": 100 * (pnl.sum() / naive.sum() - 1),
        "days_traded_%": 100 * len(traded) / len(pnl),
        "losing_trades_%": 100 * (traded < 0).mean(),
        "worst_day_eur": pnl.min(),
    }
