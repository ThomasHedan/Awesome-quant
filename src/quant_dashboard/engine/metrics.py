"""Backtest metrics + QuantStats tearsheet.

Pulls the headline ratios from vectorbt (which annualizes using the
``freq`` we set in the runner) and rounds out the picture with QuantStats
for distribution-shape metrics (CAGR via QuantStats for cross-check, tail
ratio, VaR / CVaR). Everything is read off of an already-run
:class:`BacktestResult`; we never recompute the backtest here.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from quant_dashboard.engine.runner import BacktestResult


def _safe(fn, default=float("nan")):
    """Some vectorbt ratios are NaN on empty trade sets; tidy those into NaN
    so the dataclass stays JSON-serializable without polluting the surface
    with None / Inf checks at every call site."""
    try:
        val = fn()
        if isinstance(val, pd.Series):
            val = float(val.iloc[0]) if len(val) else default
        else:
            val = float(val)
        if math.isinf(val):
            return default
        return val
    except Exception:
        return default


def _duration_days(td) -> float:
    if td is None or pd.isna(td):
        return float("nan")
    if isinstance(td, pd.Timedelta):
        return float(td.total_seconds()) / 86_400.0
    try:
        return float(td) / 86_400.0
    except Exception:
        return float("nan")


@dataclass
class Metrics:
    """Full set of backtest metrics. All scalars; NaN where undefined."""

    # Returns
    total_return: float
    cagr: float
    annualized_volatility: float

    # Risk-adjusted
    sharpe: float
    sortino: float
    calmar: float

    # Drawdowns
    max_drawdown: float
    max_drawdown_duration_days: float
    avg_drawdown_duration_days: float

    # Tails
    tail_ratio: float
    var_95: float
    cvar_95: float

    # Trade stats
    num_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    best_trade_pnl: float
    worst_trade_pnl: float
    avg_trade_pnl: float

    # Exposure / costs
    time_in_market: float
    total_fees: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def compute_metrics(result: BacktestResult) -> Metrics:
    import quantstats as qs

    pf = result.portfolio
    trades = pf.trades

    # vectorbt's trade PnL — MappedArray, .values gives a 1-D ndarray.
    pnls = np.asarray(trades.pnl.values) if trades.count() > 0 else np.array([])

    if pnls.size > 0:
        best = float(pnls.max())
        worst = float(pnls.min())
        avg = float(pnls.mean())
    else:
        best = worst = avg = float("nan")

    pm = pf.position_mask()
    time_in_market = float(pm.astype(float).mean()) if len(pm) else float("nan")

    orders = pf.orders
    total_fees = float(np.asarray(orders.fees.values).sum()) if hasattr(orders, "fees") else float("nan")

    returns = result.returns

    return Metrics(
        total_return=_safe(pf.total_return),
        cagr=_safe(lambda: qs.stats.cagr(returns)),
        annualized_volatility=_safe(pf.annualized_volatility),
        sharpe=_safe(pf.sharpe_ratio),
        sortino=_safe(pf.sortino_ratio),
        calmar=_safe(pf.calmar_ratio),
        max_drawdown=_safe(pf.max_drawdown),
        max_drawdown_duration_days=_duration_days(_dd(pf, "max_duration")),
        avg_drawdown_duration_days=_duration_days(_dd(pf, "avg_duration")),
        tail_ratio=_safe(lambda: qs.stats.tail_ratio(returns)),
        var_95=_safe(lambda: qs.stats.value_at_risk(returns, sigma=1, confidence=0.95)),
        cvar_95=_safe(lambda: qs.stats.cvar(returns, sigma=1, confidence=0.95)),
        num_trades=int(trades.count()),
        win_rate=_safe(trades.win_rate),
        profit_factor=_safe(trades.profit_factor),
        expectancy=_safe(trades.expectancy),
        best_trade_pnl=best,
        worst_trade_pnl=worst,
        avg_trade_pnl=avg,
        time_in_market=time_in_market,
        total_fees=total_fees,
    )


def _dd(pf, name: str):
    """Pull a drawdowns aggregate without exploding if there are no drawdowns."""
    try:
        return getattr(pf.drawdowns, name)()
    except Exception:
        return None


# ---------------------------------------------------------------------------


def tearsheet_html(
    result: BacktestResult,
    output_path: str | Path,
    *,
    benchmark: pd.Series | None = None,
    title: str | None = None,
) -> Path:
    """Render a QuantStats HTML tearsheet to ``output_path`` and return its path.

    ``benchmark`` is a returns series aligned (loosely) to ``result.returns``;
    QuantStats handles alignment. If omitted, QuantStats falls back to its
    default (SPY) — pass an explicit benchmark for offline / non-equity runs.
    """
    import quantstats as qs

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    returns = result.returns
    bm = benchmark
    if bm is not None and hasattr(bm.index, "tz") and bm.index.tz is not None:
        bm = bm.tz_localize(None)

    qs.reports.html(
        returns,
        benchmark=bm,
        output=str(out),
        title=title or f"{result.strategy_name} — {result.data.spec.symbol}",
        download_filename=str(out.name),
    )
    return out
