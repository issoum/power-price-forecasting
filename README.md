# Day-Ahead Power Price Forecasting & Battery Arbitrage

Hourly forecasts of the **German day-ahead electricity price (DE-LU bidding zone)** from weather
forecasts, lagged prices and calendar effects, evaluated walk-forward over 22 months. The forecasts
then drive a **battery arbitrage strategy** backtested on realised auction prices.

The focus is methodological: every feature is restricted to what a trader actually knows at
**gate closure (D-1, 12:00 CET)**, and models are only ever trained on the past.

## Results – DE-LU, out-of-sample 1 Jan 2025 → 29 Oct 2026 (667 delivery days)

**Forecast accuracy** (EUR/MWh, hourly). rMAE is the MAE relative to the weekly naive benchmark.
DM p-value: Diebold–Mariano test that the best model is more accurate (daily-averaged losses).

| Model | MAE | RMSE | rMAE | sMAPE | DM p-value vs best |
|---|---:|---:|---:|---:|---:|
| Gradient boosting | **20.2** | **30.7** | **0.55** | **35.6 %** | – |
| Ridge (LEAR-style) | 23.1 | 33.0 | 0.63 | 39.6 % | < 0.001 |
| Naive D-1 | 29.0 | 45.4 | 0.79 | 48.1 % | < 0.001 |
| Naive D-7 | 36.5 | 55.4 | 1.00 | 54.9 % | < 0.001 |

**Battery arbitrage** (1 MW / 1 MWh, one cycle per day, 85 % round-trip efficiency, **15 EUR/MWh cycle
cost** for degradation and fees). Owning a battery earns the structural intraday spread whatever the
forecast. The value of the forecast itself is therefore measured **in excess of the naive D-7
forecast**.

| Forecast used | P&L per MW-year | Excess vs naive D-7 | Excess per MW-year | 95 % CI of total excess* | Losing trades |
|---|---:|---:|---:|---:|---:|
| Ridge | €32.0k | **+9.5 %** | **€2,780** | €3.5k – €7.0k | 6.2 % |
| Gradient boosting | €31.0k | +6.0 % | €1,750 | €1.3k – €5.3k | 6.8 % |
| Naive D-1 | €30.4k | +4.1 % | €1,190 | – | 10.2 % |
| Naive D-7 | €29.2k | – | – | – | 9.8 % |

\*Block bootstrap over weeks, 667 days. Perfect foresight earns €35.1k per MW-year.
The excess value is stable for cycle costs from 0 to 30 EUR/MWh (`reports/cycle_cost_sensitivity_*.csv`).

![Forecast vs actual, last test week](reports/forecast_week_DE-LU_2025-01-01.png)
![Battery P&L in excess of the naive forecast](reports/excess_pnl_DE-LU_2025-01-01.png)

### Takeaways

- Gradient boosting cuts the error of the weekly naive benchmark by **45 %**. An ablation without
  the weather features gives an MAE of 27.0 instead of 20.2, so **weather forecasts alone reduce the
  error by a quarter**. The rest of the gain comes from recent price lags and calendar effects.
- **Most of the battery P&L is not forecasting skill.** The naive forecast already earns 83 % of
  the perfect-foresight P&L, because the German daily shape (solar trough at midday, evening peak)
  is very regular. A better forecast adds about €1.7–2.8k per MW-year (+6 to +10 %), and the
  bootstrap confidence intervals exclude zero.
- **The most accurate model is not the most profitable one.** The battery only cares about which
  hours are the cheapest and the most expensive of the day. Ridge adds more value than gradient
  boosting although gradient boosting is more accurate and ranks the hours slightly better on
  average (intraday Spearman 0.883 vs 0.875). Gradient boosting's excess P&L also stalled in summer
  2026. Accuracy metrics and trading value answer different questions.

## Backtest assumptions and limits

- **Execution:** price-independent orders in the day-ahead auction, filled at the clearing price
  with no market impact. This is realistic for a small battery.
- **Costs:** 15 EUR/MWh discharged for degradation and fees. A trade is only placed when the
  forecast spread covers losses and this cost. Grid charges, auxiliary consumption and outages
  are ignored.
- **Conservative on revenue:** one cycle per day on hourly prices. The auction clears 15-minute
  products since October 2025, and a real battery would also trade intraday and sell balancing
  services.
- The P&L figures value the asset's dispatch, not a return on capital: the battery's capex is
  not modelled.

## Avoiding look-ahead bias

| Input | What is used | Why it is available at D-1 12:00 |
|---|---|---|
| Prices | Same hour on D-1, D-2, D-7; D-1 mean/min/max | D-1 prices are published after the D-2 auction |
| Weather | Archived **forecasts**, not observations | D-1 run if issued before gate closure, otherwise the D-2 run |
| Calendar | Hour, weekday, month, public holidays | Known in advance |
| Models | Expanding window, refit monthly | Trained only on delivery days before the test month |

These rules are checked by unit tests (`tests/test_no_lookahead.py`): perturbing all prices of a
delivery day must leave that day's features unchanged, weather must come from a run issued
before gate closure, and walk-forward training data must end before the first test day.

## Data

- **Day-ahead prices:** [Energy-Charts API](https://api.energy-charts.info) (Fraunhofer ISE; data
  Bundesnetzagentur | SMARD.de, CC BY 4.0). Since 1 Oct 2025 the auction clears 15-minute products,
  which are averaged to hourly values.
- **Weather forecasts:** [Open-Meteo previous-runs API](https://open-meteo.com) – temperature, 100 m
  wind speed, solar radiation and cloud cover, as forecast one and two days ahead, averaged over six
  German locations (demand centres and a North Sea wind cluster). Complete from March 2024.

## Code

```
src/powerfc/
  data.py        downloads, caching, gate-closure-consistent weather selection
  features.py    lagged prices, weather, calendar
  models.py      naive benchmarks, ridge, gradient boosting, walk-forward loop
  evaluate.py    MAE / RMSE / rMAE / sMAPE, Diebold–Mariano test
  backtest.py    battery arbitrage on forecasts with cycle costs, settled on realised prices
  plots.py       report figures
tests/           look-ahead and backtest unit tests
```

```bash
pip install -e ".[dev]"
python -m powerfc --zone DE-LU --test-start 2025-01-01   # also supports --zone FR
pytest
```
