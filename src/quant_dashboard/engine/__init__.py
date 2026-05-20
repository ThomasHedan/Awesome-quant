"""Backtest engine, metrics, and (later) benchmark / Monte Carlo / optimizer."""

from quant_dashboard.engine.metrics import Metrics, compute_metrics, tearsheet_html
from quant_dashboard.engine.runner import (
    BacktestResult,
    CostsConfig,
    EngineError,
    run_backtest,
)

__all__ = [
    "BacktestResult",
    "CostsConfig",
    "EngineError",
    "Metrics",
    "compute_metrics",
    "run_backtest",
    "tearsheet_html",
]
