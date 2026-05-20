"""Tests for the metrics module."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from quant_dashboard.data import AssetClass, InstrumentSpec, OHLCV
from quant_dashboard.engine import CostsConfig, compute_metrics, run_backtest, tearsheet_html
from quant_dashboard.engine.metrics import Metrics
from quant_dashboard.strategies import SMACross


def _data(n: int = 500, seed: int = 11, drift: float = 0.0005) -> OHLCV:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC", name="timestamp")
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": np.full(n, 1_000.0),
        },
        index=idx,
    )
    return OHLCV(
        df=df,
        spec=InstrumentSpec(symbol="TEST", asset_class=AssetClass.EQUITY),
        timeframe="1d",
        source="synthetic",
    )


def test_metrics_populated_for_active_strategy():
    result = run_backtest(_data(500), SMACross(), {"fast": 10, "slow": 30})
    m = compute_metrics(result)

    assert isinstance(m, Metrics)
    # Numeric scalars everywhere.
    assert isinstance(m.total_return, float)
    assert isinstance(m.sharpe, float)
    assert isinstance(m.sortino, float)
    assert isinstance(m.calmar, float)
    assert isinstance(m.num_trades, int)

    # If the strategy traded, we should have some trades and a finite win rate.
    if m.num_trades > 0:
        assert 0.0 <= m.win_rate <= 1.0
        assert math.isfinite(m.best_trade_pnl)
        assert math.isfinite(m.worst_trade_pnl)
        # max_drawdown is expressed as a negative number by vectorbt.
        assert m.max_drawdown <= 0.0
        # Time in market should be a fraction in [0,1].
        assert 0.0 <= m.time_in_market <= 1.0
        # Fees are non-negative.
        assert m.total_fees >= 0.0


def test_metrics_to_dict_is_jsonable():
    result = run_backtest(_data(300), SMACross(), {"fast": 5, "slow": 20})
    d = compute_metrics(result).to_dict()
    import json
    # Will raise if any non-serializable types leaked in.
    json.dumps(d, default=str)


def test_metrics_handles_strategy_with_no_trades():
    # fast >= slow => no signals; should give a Metrics with sensible NaN.
    result = run_backtest(_data(100), SMACross(), {"fast": 50, "slow": 10})
    m = compute_metrics(result)
    assert m.num_trades == 0
    assert m.total_return == pytest.approx(0.0, abs=1e-9)
    assert math.isnan(m.best_trade_pnl)
    assert math.isnan(m.worst_trade_pnl)


def test_metrics_higher_fees_reduce_total_return():
    data = _data(500)
    low = compute_metrics(
        run_backtest(data, SMACross(), {"fast": 10, "slow": 30},
                     costs=CostsConfig(fees=0.0, slippage=0.0))
    )
    high = compute_metrics(
        run_backtest(data, SMACross(), {"fast": 10, "slow": 30},
                     costs=CostsConfig(fees=0.01, slippage=0.005))
    )
    if low.num_trades > 0:
        assert high.total_return < low.total_return
        assert high.total_fees > low.total_fees


def test_tearsheet_writes_html(tmp_path: Path):
    result = run_backtest(_data(500), SMACross(), {"fast": 10, "slow": 30})
    out = tmp_path / "tearsheet.html"
    path = tearsheet_html(result, out, title="test")
    assert path.exists()
    assert path.stat().st_size > 1_000
    head = path.read_text(encoding="utf-8")[:200].lower()
    assert "<html" in head or "<!doctype" in head
