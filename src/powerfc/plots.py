"""Report figures."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

INK, INK_2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
# Fixed colour per model so it never changes between charts or with ranking
COLORS = {"gbm": "#2a78d6", "ridge": "#eb6834", "naive_d1": "#1baf7a", "naive_d7": "#eda100"}


def _style(ax, title, ylabel):
    ax.set_facecolor(SURFACE)
    ax.figure.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", color=INK, fontsize=12)
    ax.set_ylabel(ylabel, color=INK_2)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.tick_params(colors=INK_2)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)


def forecast_week(y, fc, path, zone, models=("gbm", "ridge", "naive_d7")):
    """Actual vs forecast on the most recent full week of the test set."""
    end = y.index.max().normalize()
    sl = slice(end - pd.Timedelta(days=7), end)
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.plot(y.loc[sl].index, y.loc[sl], color=INK, linewidth=2, label="actual")
    for m in models:
        ax.plot(fc.loc[sl].index, fc.loc[sl, m], color=COLORS[m], linewidth=2, label=m)
    _style(ax, f"{zone} day-ahead price – last test week", "EUR/MWh")
    ax.legend(frameon=False, ncol=4, loc="upper left", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def excess_pnl(pnl, path, zone, cycle_cost, benchmark="naive_d7"):
    """Cumulative P&L in excess of the naive benchmark: the value added by the forecast itself."""
    fig, ax = plt.subplots(figsize=(11, 4.2))
    ax.axhline(0, color=INK_2, linewidth=1)
    for m in COLORS:
        if m == benchmark:
            continue
        ax.plot(pnl[m].index, (pnl[m] - pnl[benchmark]).cumsum(), color=COLORS[m], linewidth=2, label=m)
    _style(ax, f"{zone} battery P&L in excess of the naive D-7 forecast (cycle cost {cycle_cost:g} EUR/MWh)", "EUR")
    ax.legend(frameon=False, loc="upper left", labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
