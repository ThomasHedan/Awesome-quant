"""Multi-chart runner — run one strategy across many (symbol, timeframe) pairs.

Returns a :class:`MultiChartResult` with per-instrument metrics and the raw
:class:`BacktestResult` objects so the caller can drill into any single one.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from quant_dashboard.data.base import DataSource, OHLCV
from quant_dashboard.engine.metrics import Metrics, compute_metrics
from quant_dashboard.engine.runner import BacktestResult, CostsConfig, run_backtest
from quant_dashboard.strategies.base import Strategy


class MultiChartError(RuntimeError):
    """User-actionable multi-chart configuration errors."""


@dataclass
class InstrumentResult:
    symbol: str
    timeframe: str
    result: BacktestResult
    metrics: Metrics
    error: str | None = None


@dataclass
class MultiChartResult:
    """Aggregated output of :func:`run_multi_chart`."""

    strategy_name: str
    params: dict[str, Any]
    costs: CostsConfig
    instruments: list[InstrumentResult] = field(default_factory=list)

    @property
    def summary_df(self) -> pd.DataFrame:
        """One row per instrument: key metrics for comparison."""
        rows = []
        for ir in self.instruments:
            if ir.error:
                rows.append({"symbol": ir.symbol, "timeframe": ir.timeframe,
                              "error": ir.error})
                continue
            m = ir.metrics
            rows.append({
                "symbol": ir.symbol,
                "timeframe": ir.timeframe,
                "total_return_pct": round(m.total_return * 100, 2),
                "cagr_pct": round(m.cagr * 100, 2),
                "sharpe": round(m.sharpe, 3),
                "sortino": round(m.sortino, 3),
                "max_drawdown_pct": round(m.max_drawdown * 100, 2),
                "calmar": round(m.calmar, 3),
                "win_rate_pct": round(m.win_rate * 100, 2),
                "num_trades": m.num_trades,
                "profit_factor": round(m.profit_factor, 3),
            })
        return pd.DataFrame(rows)

    @property
    def successful(self) -> list[InstrumentResult]:
        return [ir for ir in self.instruments if ir.error is None]

    @property
    def failed(self) -> list[InstrumentResult]:
        return [ir for ir in self.instruments if ir.error is not None]


def run_multi_chart(
    sources: list[tuple[DataSource, str, str]],  # (source, symbol, timeframe)
    strategy: Strategy,
    params: dict[str, Any],
    costs: CostsConfig | None = None,
    start: str = "2020-01-01",
    end: str | None = None,
) -> MultiChartResult:
    """Run *strategy* on every (source, symbol, timeframe) combination.

    Args:
        sources: List of ``(DataSource, symbol, timeframe)`` triples.
        strategy: An instantiated :class:`Strategy`.
        params: Strategy parameters dict (falls back to defaults for missing keys).
        costs: Shared :class:`CostsConfig` applied to all instruments. Defaults to
            ``CostsConfig()`` (5 bps fees + 5 bps slippage).
        start: ISO date string for the data window start.
        end: ISO date string for the data window end. ``None`` = today.

    Returns:
        :class:`MultiChartResult` with per-instrument results and a
        ``summary_df`` DataFrame ready for display.

    Raises:
        :class:`MultiChartError`: if ``sources`` is empty or ``strategy`` is not
            a :class:`Strategy` instance.
    """
    if not sources:
        raise MultiChartError("sources must not be empty")
    if not isinstance(strategy, Strategy):
        raise MultiChartError(
            f"strategy must be a Strategy instance, got {type(strategy).__name__}"
        )

    _costs = costs or CostsConfig()
    full_params = {**strategy.default_params(), **params}

    result = MultiChartResult(
        strategy_name=strategy.name,
        params=full_params,
        costs=_costs,
    )

    for source, symbol, timeframe in sources:
        try:
            data: OHLCV = source.fetch(symbol, timeframe, start=start, end=end)
            bt = run_backtest(data, strategy, full_params, costs=_costs)
            m = compute_metrics(bt)
            result.instruments.append(
                InstrumentResult(symbol=symbol, timeframe=timeframe,
                                 result=bt, metrics=m)
            )
        except Exception as exc:  # noqa: BLE001 — surface per-instrument errors
            result.instruments.append(
                InstrumentResult(symbol=symbol, timeframe=timeframe,
                                 result=None,  # type: ignore[arg-type]
                                 metrics=None,  # type: ignore[arg-type]
                                 error=str(exc))
            )

    return result
