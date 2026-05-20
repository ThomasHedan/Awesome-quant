"""Backtest runner — the thin wrapper around ``vbt.Portfolio.from_signals``.

The runner owns the no-look-ahead shift (strategies emit decision-bar signals;
the runner shifts them one bar forward so vectorbt executes at the *next*
bar's close), the cost configuration, and the frequency inference vectorbt
needs for annualization.

For v1, fees and slippage are uniform fractional costs applied per fill. This
is exact for crypto and equities; for futures it requires the user to
calibrate against typical notional — there's a TODO to add per-contract
accounting once we wire the futures contract sizing into the engine.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quant_dashboard.data import OHLCV
from quant_dashboard.strategies.base import Signals, Strategy


class EngineError(RuntimeError):
    """User-actionable backtest configuration errors."""


@dataclass(frozen=True)
class CostsConfig:
    """Fractional fees + slippage per fill, plus sizing.

    ``fees`` and ``slippage`` are fractions of trade notional. For crypto and
    equities this is exact (10 bps = 0.001). For futures, calibrate against
    a typical notional: ``commission_per_contract / (price * multiplier)``.
    """

    fees: float = 0.0005           # 5 bps round-trip half
    slippage: float = 0.0005       # 5 bps
    init_cash: float = 100_000.0
    size: float | None = None      # passed through to vbt; None = 100% on entry
    size_type: str = "percent"     # "percent", "value", "amount"

    def __post_init__(self) -> None:
        if self.fees < 0 or self.slippage < 0:
            raise EngineError("fees and slippage must be non-negative")
        if self.init_cash <= 0:
            raise EngineError("init_cash must be positive")


@dataclass
class BacktestResult:
    """Output of :func:`run_backtest` — the vectorbt portfolio plus the
    information needed to render reports without re-running the backtest."""

    portfolio: Any            # vbt.Portfolio (kept loose to avoid import here)
    data: OHLCV
    strategy_name: str
    params: dict[str, Any]
    costs: CostsConfig
    signals: Signals
    freq: str
    meta: dict = field(default_factory=dict)

    @property
    def returns(self) -> pd.Series:
        r = self.portfolio.returns()
        # vectorbt returns a tz-aware index; QuantStats prefers tz-naive.
        if hasattr(r.index, "tz") and r.index.tz is not None:
            r = r.tz_localize(None)
        return r

    @property
    def equity_curve(self) -> pd.Series:
        v = self.portfolio.value()
        if hasattr(v.index, "tz") and v.index.tz is not None:
            v = v.tz_localize(None)
        return v


# ---------------------------------------------------------------------------


def _infer_freq(index: pd.DatetimeIndex) -> str:
    """Return a vectorbt-compatible frequency string from the index spacing.

    We use the median spacing, not ``infer_freq``, because intraday or
    irregular data (weekends, holidays, halts) commonly defeats inference.
    """
    if len(index) < 2:
        raise EngineError("need at least two bars to infer a frequency")
    deltas = index.to_series().diff().dropna()
    median = deltas.median()
    seconds = int(median.total_seconds())
    if seconds <= 0:
        raise EngineError("non-positive median bar spacing")
    # Map common cases to readable strings; otherwise fall back to seconds.
    table = [
        (60, "1min"),
        (5 * 60, "5min"),
        (15 * 60, "15min"),
        (30 * 60, "30min"),
        (60 * 60, "1h"),
        (4 * 60 * 60, "4h"),
        (24 * 60 * 60, "1D"),
        (7 * 24 * 60 * 60, "1W"),
    ]
    for s, label in table:
        if seconds == s:
            return label
    return f"{seconds}s"


def _shift_signals_for_execution(sig: Signals) -> tuple[pd.Series, pd.Series]:
    """Shift decision-bar signals to next-bar execution.

    Strategies emit signals aligned to the bar whose close drove the decision.
    To avoid look-ahead at the engine boundary, we shift them by one so
    vectorbt executes at the close of the *next* bar. The first bar is
    dropped from being actionable (set False) since there's no prior decision.
    """
    entries = sig.entries.shift(1).fillna(False).astype(bool)
    exits = sig.exits.shift(1).fillna(False).astype(bool)
    return entries, exits


def run_backtest(
    data: OHLCV,
    strategy: Strategy,
    params: dict[str, Any] | None = None,
    costs: CostsConfig | None = None,
) -> BacktestResult:
    """Run a single backtest end-to-end.

    Parameters
    ----------
    data:
        Validated OHLCV bundle from a :class:`DataSource`.
    strategy:
        Strategy instance. Params validated against its schema.
    params:
        Override values; missing keys fall back to ``strategy.default_params()``.
    costs:
        Fees, slippage, sizing. Defaults to a conservative 5 bps / 5 bps.
    """
    import vectorbt as vbt

    params = dict(params or {})
    costs = costs or CostsConfig()

    sig = strategy.generate_signals(data, **params)
    entries, exits = _shift_signals_for_execution(sig)

    freq = _infer_freq(data.df.index)

    pf_kwargs: dict[str, Any] = dict(
        close=data.df["close"],
        entries=entries,
        exits=exits,
        fees=costs.fees,
        slippage=costs.slippage,
        init_cash=costs.init_cash,
        freq=freq,
    )
    if costs.size is not None:
        pf_kwargs["size"] = costs.size
        pf_kwargs["size_type"] = costs.size_type

    try:
        pf = vbt.Portfolio.from_signals(**pf_kwargs)
    except Exception as exc:
        raise EngineError(f"vectorbt backtest failed: {exc}") from exc

    return BacktestResult(
        portfolio=pf,
        data=data,
        strategy_name=strategy.name,
        params=params,
        costs=costs,
        signals=sig,
        freq=freq,
    )
