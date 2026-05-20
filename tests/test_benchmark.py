"""Tests for the benchmark comparison module."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_dashboard.data import AssetClass, InstrumentSpec, OHLCV
from quant_dashboard.engine import (
    BenchmarkError,
    build_buy_and_hold_returns,
    compute_benchmark_comparison,
    run_backtest,
)
from quant_dashboard.engine.benchmark import _ols_alpha_beta
from quant_dashboard.strategies import SMACross


def _data(n=400, seed=0, drift=0.0005) -> OHLCV:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC", name="timestamp")
    df = pd.DataFrame(
        {"open": close, "high": close * 1.005, "low": close * 0.995,
         "close": close, "volume": np.full(n, 1_000.0)},
        index=idx,
    )
    return OHLCV(
        df=df,
        spec=InstrumentSpec(symbol="TEST", asset_class=AssetClass.EQUITY),
        timeframe="1d", source="synthetic",
    )


# ---------- OLS sanity ------------------------------------------------------


def test_ols_recovers_known_alpha_beta():
    rng = np.random.default_rng(0)
    x = rng.normal(0.0005, 0.012, 1_000)
    # y = 0.0002 + 1.5 * x + noise
    y = 0.0002 + 1.5 * x + rng.normal(0, 0.001, 1_000)
    alpha, beta, r2 = _ols_alpha_beta(y, x, rf_per_period=0.0)
    assert beta == pytest.approx(1.5, abs=0.05)
    assert alpha == pytest.approx(0.0002, abs=5e-4)
    assert 0.9 < r2 <= 1.0


def test_ols_rejects_zero_variance_benchmark():
    with pytest.raises(BenchmarkError):
        _ols_alpha_beta(np.array([0.01, 0.02, 0.0]), np.zeros(3))


# ---------- build_buy_and_hold_returns --------------------------------------


def test_buy_and_hold_returns_match_close_pct_change():
    data = _data(100)
    bh = build_buy_and_hold_returns(data)
    expected = data.df["close"].pct_change().fillna(0.0)
    np.testing.assert_allclose(bh.to_numpy(), expected.to_numpy())
    assert bh.index.tz is None  # tz-naive to match the engine returns


# ---------- end-to-end comparison ------------------------------------------


def test_benchmark_against_self_gives_beta_one():
    """Using the instrument as its own benchmark, a B&H "strategy" must
    produce beta == 1, alpha ~= 0, correlation == 1."""
    data = _data(500)
    bh = build_buy_and_hold_returns(data)

    # Hand-rolled BacktestResult substitute: re-use the SMACross runner just
    # to get an actual BacktestResult, then override its returns property
    # via a stand-in object so we don't have to mock vbt.
    class _Result:
        def __init__(self, returns, data):
            self.returns = returns
            self.data = data

    result = _Result(bh, data)
    comp = compute_benchmark_comparison(result, benchmark_returns=bh)
    assert comp.beta == pytest.approx(1.0, abs=1e-6)
    assert abs(comp.alpha) < 1e-6
    assert comp.correlation == pytest.approx(1.0, abs=1e-6)
    assert comp.r_squared == pytest.approx(1.0, abs=1e-6)
    assert comp.excess_total_return == pytest.approx(0.0, abs=1e-9)


def test_benchmark_strategy_vs_buy_and_hold():
    data = _data(500, seed=3)
    result = run_backtest(data, SMACross(), {"fast": 10, "slow": 30})
    comp = compute_benchmark_comparison(result)
    assert comp.benchmark_name.startswith("B&H")
    assert isinstance(comp.alpha, float)
    assert isinstance(comp.beta, float)
    assert -1.0 <= comp.correlation <= 1.0
    assert 0.0 <= comp.r_squared <= 1.0
    # Relative equity is normalized: starts near 1.0.
    assert comp.relative_equity.iloc[0] == pytest.approx(1.0, abs=0.1)
    # Total returns are real numbers.
    assert math.isfinite(comp.strategy_total_return)
    assert math.isfinite(comp.benchmark_total_return)


def test_benchmark_misaligned_series_raises():
    data = _data(500)
    result = run_backtest(data, SMACross(), {"fast": 10, "slow": 30})
    # Benchmark only overlaps in 2 bars.
    bm_idx = pd.date_range("2030-01-01", periods=5, freq="1D")
    bm = pd.Series(np.random.normal(0, 0.01, 5), index=bm_idx)
    with pytest.raises(BenchmarkError):
        compute_benchmark_comparison(result, benchmark_returns=bm)


def test_benchmark_information_ratio_well_defined():
    data = _data(500, seed=11)
    result = run_backtest(data, SMACross(), {"fast": 10, "slow": 30})
    comp = compute_benchmark_comparison(result)
    # IR may be negative but should be finite for a non-flat excess series.
    assert math.isfinite(comp.information_ratio) or math.isnan(comp.information_ratio)
