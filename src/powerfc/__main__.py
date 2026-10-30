"""End-to-end run: download -> features -> walk-forward forecasts -> metrics -> backtest -> plots.

    python -m powerfc --zone DE-LU --test-start 2025-01-01
"""
import argparse

import pandas as pd

from . import backtest, data, evaluate, features, models, plots
from .config import DATA_START, REPORTS_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zone", default="DE-LU")
    ap.add_argument("--start", default=DATA_START)
    ap.add_argument("--end", default=pd.Timestamp.today().normalize().strftime("%Y-%m-%d"))
    ap.add_argument("--test-start", default="2025-01-01")
    ap.add_argument("--refresh", action="store_true", help="re-download instead of using the cache")
    args = ap.parse_args()

    df = data.load(args.zone, args.start, args.end, refresh=args.refresh)
    X = features.build(df, args.zone)
    y = df["price"]

    fc = {m.name: models.walk_forward(X, y, m, args.test_start) for m in models.ALL}
    fc = pd.DataFrame(fc).dropna()
    y_test = y.loc[fc.index]
    days = X.loc[fc.index, "delivery_day"]

    acc = evaluate.table(y_test, {c: fc[c] for c in fc}, days)

    perfect = backtest.run(y_test, y_test, days)
    pnl = {c: backtest.run(y_test, fc[c], days) for c in fc}
    bt = pd.DataFrame({c: backtest.summary(p, perfect) for c, p in pnl.items()}).T.sort_values("total_pnl_eur", ascending=False)

    REPORTS_DIR.mkdir(exist_ok=True)
    tag = f"{args.zone}_{args.test_start}"
    acc.to_csv(REPORTS_DIR / f"accuracy_{tag}.csv")
    bt.to_csv(REPORTS_DIR / f"backtest_{tag}.csv")
    plots.forecast_week(y_test, fc, REPORTS_DIR / f"forecast_week_{tag}.png", args.zone)
    plots.cumulative_pnl(pnl, perfect, REPORTS_DIR / f"cum_pnl_{tag}.png", args.zone)

    pd.set_option("display.float_format", "{:,.3f}".format)
    print(f"\nZone {args.zone} | test delivery days {days.min():%Y-%m-%d} -> {days.max():%Y-%m-%d} | {days.nunique()} days\n")
    print("Forecast accuracy (EUR/MWh)\n", acc, "\n")
    print("Battery arbitrage backtest (1 MW / 1 MWh, 85% round trip)\n", bt)


if __name__ == "__main__":
    main()
