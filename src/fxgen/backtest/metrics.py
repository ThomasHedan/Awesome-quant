"""Performance metrics computed from a trade list.

Trades are measured in R (net pips / initial stop distance in pips) so that
pairs with different volatility are comparable. Daily series attribute each
trade's R to its exit date; Sharpe is annualised from business-day series.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from fxgen.data.splits import Period

TRADING_DAYS = 260


def daily_r(trades: pd.DataFrame, period: Period) -> pd.Series:
    """Business-day series of summed R (weekend exits roll to Monday, zero on idle days)."""
    days = pd.bdate_range(period.start.normalize(), period.end.normalize(), inclusive="left", tz="UTC")
    if trades.empty:
        return pd.Series(0.0, index=days)
    day = trades["exit_time"].dt.normalize()
    dow = day.dt.day_of_week
    day = day + pd.to_timedelta(np.select([dow == 5, dow == 6], [2, 1], 0), unit="D")
    by_day = trades.groupby(day)["r_multiple"].sum()
    return by_day.reindex(days, fill_value=0.0).astype(float)


def sharpe(daily: pd.Series) -> float:
    std = float(daily.std(ddof=1)) if len(daily) > 1 else 0.0
    if std == 0 or not math.isfinite(std):
        return 0.0
    return float(daily.mean() / std * math.sqrt(TRADING_DAYS))


def max_drawdown(r_sequence: np.ndarray) -> float:
    """Largest peak-to-trough drop of the cumulative R curve (positive number)."""
    if r_sequence.size == 0:
        return 0.0
    equity = np.concatenate([[0.0], np.cumsum(r_sequence)])
    return float(np.max(np.maximum.accumulate(equity) - equity))


def compute_metrics(trades: pd.DataFrame, period: Period, risk_pct: float) -> dict[str, Any]:
    """Summary metrics of a trade list over ``period``."""
    years = max((period.end - period.start).days / 365.25, 1e-9)
    n = len(trades)
    if n == 0:
        return {
            "n_trades": 0, "trades_per_year": 0.0, "win_rate": 0.0, "profit_factor": 0.0, "expectancy_r": 0.0,
            "total_r": 0.0, "sharpe": 0.0, "max_dd_r": 0.0, "max_dd_pct": 0.0, "return_pct": 0.0,
            "worst_day_pct": 0.0, "avg_bars_held": 0.0, "long_share": 0.0,
        }
    ordered = trades.sort_values("exit_time")
    r = ordered["r_multiple"].to_numpy()
    gains, losses = r[r > 0].sum(), -r[r < 0].sum()
    daily = daily_r(ordered, period)
    dd = max_drawdown(r)
    return {
        "n_trades": int(n),
        "trades_per_year": float(n / years),
        "win_rate": float((r > 0).mean()),
        "profit_factor": float(gains / losses) if losses > 0 else float("inf") if gains > 0 else 0.0,
        "expectancy_r": float(r.mean()),
        "total_r": float(r.sum()),
        "sharpe": sharpe(daily),
        "max_dd_r": dd,
        "max_dd_pct": dd * risk_pct,
        "return_pct": float(r.sum() * risk_pct),
        "worst_day_pct": float(min(0.0, daily.min()) * risk_pct),
        "avg_bars_held": float(ordered["bars_held"].mean()),
        "long_share": float((ordered["direction"] > 0).mean()),
    }


def per_year(trades: pd.DataFrame, risk_pct: float) -> pd.DataFrame:
    """R, trade count and return per calendar year of exit."""
    if trades.empty:
        return pd.DataFrame(columns=["year", "n_trades", "total_r", "return_pct", "win_rate"])
    grouped = trades.groupby(trades["exit_time"].dt.year)["r_multiple"]
    out = pd.DataFrame(
        {
            "n_trades": grouped.size(),
            "total_r": grouped.sum(),
            "return_pct": grouped.sum() * risk_pct,
            "win_rate": grouped.apply(lambda s: float((s > 0).mean())),
        }
    )
    return out.rename_axis("year").reset_index()
