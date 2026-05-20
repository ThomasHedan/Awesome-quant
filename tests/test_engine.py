"""Tests for the backtest engine (runner + costs + signal shifting)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_dashboard.data import AssetClass, InstrumentSpec, OHLCV
from quant_dashboard.engine import BacktestResult, CostsConfig, EngineError, run_backtest
from quant_dashboard.engine.runner import _infer_freq, _shift_signals_for_execution
from quant_dashboard.strategies import RSIMeanReversion, SMACross
from quant_dashboard.strategies.base import Signals


def _trending_ohlcv(n: int = 400, seed: int = 0, drift: float = 0.0008) -> OHLCV:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC", name="timestamp")
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


# ---------- frequency inference --------------------------------------------


def test_infer_freq_daily():
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    assert _infer_freq(idx) == "1D"


def test_infer_freq_hourly():
    idx = pd.date_range("2024-01-01", periods=5, freq="1h", tz="UTC")
    assert _infer_freq(idx) == "1h"


def test_infer_freq_with_gaps_uses_median():
    # Weekly business-ish days: median is 1D even with weekend gaps.
    idx = pd.DatetimeIndex(
        pd.to_datetime(
            ["2024-01-02", "2024-01-03", "2024-01-04", "2024-01-05", "2024-01-08"],
            utc=True,
        )
    )
    assert _infer_freq(idx) == "1D"


def test_infer_freq_too_short():
    idx = pd.DatetimeIndex([pd.Timestamp("2024-01-01", tz="UTC")])
    with pytest.raises(EngineError):
        _infer_freq(idx)


# ---------- signal shifting -------------------------------------------------


def test_shift_signals_drops_first_bar():
    idx = pd.date_range("2024-01-01", periods=5, freq="1D", tz="UTC")
    entries = pd.Series([True, False, True, False, False], index=idx)
    exits = pd.Series([False, True, False, True, False], index=idx)
    sig = Signals(entries=entries, exits=exits)
    e2, x2 = _shift_signals_for_execution(sig)
    # The decision at bar 0 must NOT cause a fill at bar 0.
    assert e2.iloc[0] == False
    assert x2.iloc[0] == False
    # Bar 0 decision shows up at bar 1.
    assert e2.iloc[1] == True
    assert x2.iloc[2] == True


# ---------- costs validation -----------------------------------------------


def test_costs_rejects_negative_fees():
    with pytest.raises(EngineError):
        CostsConfig(fees=-0.001)


def test_costs_rejects_nonpositive_init_cash():
    with pytest.raises(EngineError):
        CostsConfig(init_cash=0.0)


# ---------- end-to-end run --------------------------------------------------


def test_run_backtest_returns_populated_result():
    data = _trending_ohlcv(400, drift=0.0008)
    result = run_backtest(data, SMACross(), {"fast": 10, "slow": 30})
    assert isinstance(result, BacktestResult)
    assert result.strategy_name == "sma_cross"
    assert result.params == {"fast": 10, "slow": 30}
    assert result.freq == "1D"
    # The equity curve should align with the OHLCV index (one value per bar).
    eq = result.equity_curve
    assert len(eq) == len(data.df)


def test_run_backtest_no_lookahead_at_engine_boundary():
    """Even at the engine layer, no fill should occur on the very first bar."""
    data = _trending_ohlcv(200)
    result = run_backtest(data, SMACross(), {"fast": 5, "slow": 20})
    # Orders should never be placed at index 0 because the shift drops it.
    orders = result.portfolio.orders
    if orders.count() > 0:
        first_ts = pd.Timestamp(orders.records_readable["Timestamp"].iloc[0])
        if first_ts.tzinfo is None:
            first_ts = first_ts.tz_localize("UTC")
        # The very first bar can't have a fill because there's no prior decision.
        assert first_ts > data.df.index[0]


def test_run_backtest_with_rsi():
    data = _trending_ohlcv(400, drift=0.0)
    result = run_backtest(
        data,
        RSIMeanReversion(),
        {"period": 14, "lower": 30.0, "upper": 70.0},
        costs=CostsConfig(fees=0.001, slippage=0.0005, init_cash=50_000.0),
    )
    assert result.strategy_name == "rsi_meanrev"
    assert result.costs.fees == 0.001
    assert result.costs.init_cash == 50_000.0


def test_run_backtest_zero_signals_runs_cleanly():
    """A strategy that never trades should produce a flat equity curve at init_cash."""
    data = _trending_ohlcv(100)
    # fast >= slow => SMACross emits nothing by construction.
    result = run_backtest(data, SMACross(), {"fast": 50, "slow": 10})
    eq = result.equity_curve
    assert eq.iloc[0] == pytest.approx(eq.iloc[-1], rel=1e-9)
    assert result.portfolio.trades.count() == 0
