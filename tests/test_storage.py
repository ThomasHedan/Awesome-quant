"""Tests for the SQLite run store."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_dashboard.data import AssetClass, InstrumentSpec, OHLCV
from quant_dashboard.engine import (
    compute_metrics,
    run_backtest,
    run_grid_search,
    run_monte_carlo,
)
from quant_dashboard.storage import RunStore, StorageError
from quant_dashboard.strategies import SMACross


def _data(n=300, seed=0) -> OHLCV:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
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


@pytest.fixture
def store(tmp_path):
    return RunStore(tmp_path / "runs.db")


# ---------- backtest persistence -------------------------------------------


def test_save_and_get_backtest_roundtrip(store):
    result = run_backtest(_data(), SMACross(), {"fast": 10, "slow": 30})
    metrics = compute_metrics(result)
    run_id = store.save_backtest(result, metrics, notes="first run")
    assert run_id >= 1

    row = store.get_run(run_id)
    assert row["strategy_name"] == "sma_cross"
    assert row["symbol"] == "TEST"
    assert row["source"] == "synthetic"
    assert row["params"] == {"fast": 10, "slow": 30}
    assert row["costs"]["fees"] == result.costs.fees
    assert row["spec"]["asset_class"] == "equity"
    assert row["metrics"]["num_trades"] == metrics.num_trades
    assert row["notes"] == "first run"
    assert len(row["config_hash"]) == 16


def test_list_runs_filters_by_strategy(store):
    r = run_backtest(_data(), SMACross(), {"fast": 10, "slow": 30})
    m = compute_metrics(r)
    store.save_backtest(r, m)
    store.save_backtest(r, m)
    df = store.list_runs(strategy_name="sma_cross")
    assert len(df) == 2
    df_empty = store.list_runs(strategy_name="nonexistent")
    assert df_empty.empty


def test_delete_run_also_clears_children(store):
    r = run_backtest(_data(), SMACross(), {"fast": 10, "slow": 30})
    m = compute_metrics(r)
    run_id = store.save_backtest(r, m)
    mc = run_monte_carlo(r, method="bootstrap", n_simulations=50, seed=0)
    store.save_monte_carlo(run_id, mc)

    row_before = store.get_run(run_id)
    assert len(row_before["monte_carlo_runs"]) == 1

    store.delete_run(run_id)
    with pytest.raises(StorageError):
        store.get_run(run_id)


# ---------- config hash stability ------------------------------------------


def test_config_hash_stable_across_saves(store):
    r1 = run_backtest(_data(), SMACross(), {"fast": 10, "slow": 30})
    r2 = run_backtest(_data(), SMACross(), {"fast": 10, "slow": 30})
    m = compute_metrics(r1)
    id1 = store.save_backtest(r1, m)
    id2 = store.save_backtest(r2, m)
    assert store.get_run(id1)["config_hash"] == store.get_run(id2)["config_hash"]


def test_config_hash_changes_when_params_change(store):
    r1 = run_backtest(_data(), SMACross(), {"fast": 10, "slow": 30})
    r2 = run_backtest(_data(), SMACross(), {"fast": 5, "slow": 30})
    m = compute_metrics(r1)
    id1 = store.save_backtest(r1, m)
    id2 = store.save_backtest(r2, m)
    assert store.get_run(id1)["config_hash"] != store.get_run(id2)["config_hash"]


# ---------- Monte Carlo persistence ----------------------------------------


def test_save_monte_carlo_requires_parent(store):
    r = run_backtest(_data(), SMACross(), {"fast": 10, "slow": 30})
    mc = run_monte_carlo(r, method="bootstrap", n_simulations=50, seed=0)
    with pytest.raises(StorageError):
        store.save_monte_carlo(99_999, mc)


def test_monte_carlo_summary_persisted(store):
    r = run_backtest(_data(), SMACross(), {"fast": 10, "slow": 30})
    m = compute_metrics(r)
    run_id = store.save_backtest(r, m)
    mc = run_monte_carlo(
        r, method="bootstrap", n_simulations=100, seed=42,
        goal_return=0.15, ruin_threshold=0.40,
    )
    store.save_monte_carlo(run_id, mc)
    row = store.get_run(run_id)
    assert len(row["monte_carlo_runs"]) == 1
    summary = row["monte_carlo_runs"][0]["summary"]
    assert summary["method"] == "bootstrap"
    assert summary["n_simulations"] == 100
    assert summary["goal_return"] == 0.15
    assert summary["ruin_threshold"] == 0.40


# ---------- optimization persistence ---------------------------------------


def test_save_and_get_optimization(store):
    data = _data(300)
    grids = {"fast": [5, 10], "slow": [30, 50]}
    opt = run_grid_search(data, SMACross, grids, metric="sharpe")
    opt_id = store.save_optimization(
        opt,
        source=data.source, symbol=data.spec.symbol,
        timeframe=data.timeframe,
        start=data.df.index[0], end=data.df.index[-1],
        grids=grids,
    )
    row = store.get_optimization(opt_id)
    assert row["strategy_name"] == "sma_cross"
    assert row["metric"] == "sharpe"
    assert row["n_combos"] == 4
    assert set(row["grids"]) == {"fast", "slow"}
    assert set(row["best_params"]) == {"fast", "slow"}
    assert isinstance(row["rankings"], pd.DataFrame)
    assert len(row["rankings"]) == 4
    assert "sharpe" in row["rankings"].columns


def test_list_optimizations(store):
    data = _data(200)
    grids = {"fast": [5, 10], "slow": [30, 50]}
    opt = run_grid_search(data, SMACross, grids, metric="sharpe")
    store.save_optimization(
        opt, source="synthetic", symbol="TEST", timeframe="1d",
        start=data.df.index[0], end=data.df.index[-1], grids=grids,
    )
    df = store.list_optimizations(strategy_name="sma_cross")
    assert len(df) == 1
    assert df.iloc[0]["strategy_name"] == "sma_cross"
