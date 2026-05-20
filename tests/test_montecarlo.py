"""Tests for the Monte Carlo edge-analysis module."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from quant_dashboard.data import AssetClass, InstrumentSpec, OHLCV
from quant_dashboard.engine import (
    MonteCarloError,
    MonteCarloResult,
    run_backtest,
    run_monte_carlo,
)
from quant_dashboard.strategies import SMACross


def _data(n=500, seed=11, drift=0.0005) -> OHLCV:
    rng = np.random.default_rng(seed)
    rets = rng.normal(drift, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC", name="timestamp")
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


# ---------- argument validation --------------------------------------------


def test_run_monte_carlo_rejects_tiny_n_simulations():
    result = run_backtest(_data(300), SMACross(), {"fast": 10, "slow": 30})
    with pytest.raises(MonteCarloError):
        run_monte_carlo(result, n_simulations=5)


def test_run_monte_carlo_rejects_bad_ruin_threshold():
    result = run_backtest(_data(300), SMACross(), {"fast": 10, "slow": 30})
    with pytest.raises(MonteCarloError):
        run_monte_carlo(result, n_simulations=100, ruin_threshold=1.5)
    with pytest.raises(MonteCarloError):
        run_monte_carlo(result, n_simulations=100, ruin_threshold=0.0)


# ---------- reproducibility ------------------------------------------------


def test_monte_carlo_is_seed_deterministic():
    result = run_backtest(_data(500), SMACross(), {"fast": 10, "slow": 30})
    a = run_monte_carlo(result, method="bootstrap", n_simulations=200, seed=7)
    b = run_monte_carlo(result, method="bootstrap", n_simulations=200, seed=7)
    np.testing.assert_array_equal(a.terminal_returns, b.terminal_returns)
    np.testing.assert_array_equal(a.sharpes, b.sharpes)
    np.testing.assert_array_equal(a.max_drawdowns, b.max_drawdowns)


def test_monte_carlo_different_seeds_differ():
    result = run_backtest(_data(500), SMACross(), {"fast": 10, "slow": 30})
    a = run_monte_carlo(result, n_simulations=200, seed=1)
    b = run_monte_carlo(result, n_simulations=200, seed=2)
    assert not np.array_equal(a.terminal_returns, b.terminal_returns)


# ---------- bootstrap shape -------------------------------------------------


def test_bootstrap_outputs_have_expected_shape():
    result = run_backtest(_data(500), SMACross(), {"fast": 10, "slow": 30})
    mc = run_monte_carlo(result, method="bootstrap", n_simulations=300, seed=0)
    assert isinstance(mc, MonteCarloResult)
    assert mc.n_simulations == 300
    assert mc.terminal_returns.shape == (300,)
    assert mc.sharpes.shape == (300,)
    assert mc.max_drawdowns.shape == (300,)
    # Equity CI: rows = bars, cols = 5 percentiles.
    assert mc.equity_curves_ci.shape == (len(result.returns), 5)
    assert list(mc.equity_curves_ci.columns) == ["p05", "p25", "p50", "p75", "p95"]
    # Realized values are scalars.
    assert math.isfinite(mc.realized_total_return)
    assert isinstance(mc.realized_max_drawdown, float)


def test_bootstrap_ci_is_monotone_in_percentile():
    result = run_backtest(_data(500), SMACross(), {"fast": 10, "slow": 30})
    mc = run_monte_carlo(result, method="bootstrap", n_simulations=500, seed=0)
    ci = mc.equity_curves_ci
    # p05 <= p25 <= p50 <= p75 <= p95 at every bar.
    assert (ci["p05"] <= ci["p25"]).all()
    assert (ci["p25"] <= ci["p50"]).all()
    assert (ci["p50"] <= ci["p75"]).all()
    assert (ci["p75"] <= ci["p95"]).all()


# ---------- shuffle method --------------------------------------------------


def test_shuffle_method_runs_when_trades_exist():
    result = run_backtest(_data(500), SMACross(), {"fast": 10, "slow": 30})
    n_trades = result.portfolio.trades.count()
    if n_trades < 2:
        pytest.skip("strategy didn't produce enough trades on this seed")
    mc = run_monte_carlo(result, method="shuffle", n_simulations=200, seed=0)
    assert mc.method == "shuffle"
    # Equity curve length matches the number of trades, not bars.
    assert mc.equity_curves_ci.shape == (n_trades, 5)


def test_shuffle_raises_when_no_trades():
    # fast >= slow => SMACross emits nothing.
    result = run_backtest(_data(200), SMACross(), {"fast": 50, "slow": 10})
    with pytest.raises(MonteCarloError):
        run_monte_carlo(result, method="shuffle", n_simulations=100)


# ---------- edge probabilities ---------------------------------------------


def test_prob_goal_and_ruin_in_unit_interval():
    result = run_backtest(_data(500, drift=0.001), SMACross(), {"fast": 10, "slow": 30})
    mc = run_monte_carlo(
        result, method="bootstrap", n_simulations=500, seed=0,
        goal_return=0.10, ruin_threshold=0.30,
    )
    assert 0.0 <= mc.prob_goal <= 1.0
    assert 0.0 <= mc.prob_ruin <= 1.0


def test_percentile_ranks_within_0_100():
    result = run_backtest(_data(500), SMACross(), {"fast": 10, "slow": 30})
    mc = run_monte_carlo(result, method="bootstrap", n_simulations=500, seed=0)
    assert 0.0 <= mc.sharpe_percentile <= 100.0 or math.isnan(mc.sharpe_percentile)
    assert 0.0 <= mc.return_percentile <= 100.0 or math.isnan(mc.return_percentile)


def test_summary_jsonable():
    result = run_backtest(_data(400), SMACross(), {"fast": 10, "slow": 30})
    mc = run_monte_carlo(result, method="bootstrap", n_simulations=200, seed=0)
    import json
    json.dumps(mc.to_summary(), default=str)
