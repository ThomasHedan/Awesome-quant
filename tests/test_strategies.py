"""Tests for the strategy layer.

The headline test is :func:`test_no_lookahead_for_all_strategies` — it
asserts that every registered strategy's signals at time ``t`` depend only
on data up to ``t``, by mutating future bars and verifying past signals
are unchanged.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_dashboard.data import AssetClass, InstrumentSpec, OHLCV
from quant_dashboard.strategies import (
    ParamSpec,
    RSIMeanReversion,
    SMACross,
    Signals,
    Strategy,
    StrategyError,
    available_strategies,
    get_strategy,
    register,
    strategy_registry,
)
from quant_dashboard.strategies.base import iter_param_grid


def _ohlcv(n: int = 300, seed: int = 1) -> OHLCV:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2023-01-01", periods=n, freq="1D", tz="UTC", name="timestamp")
    df = pd.DataFrame(
        {
            "open": close,
            "high": close * 1.005,
            "low": close * 0.995,
            "close": close,
            "volume": rng.uniform(1_000, 5_000, n),
        },
        index=idx,
    )
    return OHLCV(
        df=df,
        spec=InstrumentSpec(symbol="TEST", asset_class=AssetClass.EQUITY),
        timeframe="1d",
        source="synthetic",
    )


# ---------- ParamSpec -------------------------------------------------------


def test_param_spec_grid_int_inclusive():
    p = ParamSpec(name="x", kind="int", default=2, low=1, high=5, step=1)
    assert p.grid() == [1, 2, 3, 4, 5]


def test_param_spec_grid_float():
    p = ParamSpec(name="x", kind="float", default=0.1, low=0.1, high=0.3, step=0.1)
    grid = p.grid()
    assert len(grid) == 3
    assert grid[0] == pytest.approx(0.1)
    assert grid[-1] == pytest.approx(0.3)


def test_param_spec_choice_validation():
    p = ParamSpec(name="x", kind="choice", default="a", choices=("a", "b"))
    assert p.validate("b") == "b"
    with pytest.raises(StrategyError):
        p.validate("c")


def test_param_spec_int_bounds():
    p = ParamSpec(name="x", kind="int", default=5, low=1, high=10, step=1)
    with pytest.raises(StrategyError):
        p.validate(0)
    with pytest.raises(StrategyError):
        p.validate(11)


# ---------- iter_param_grid -------------------------------------------------


def test_iter_param_grid_cartesian():
    out = list(iter_param_grid({"a": [1, 2], "b": [10, 20]}))
    assert len(out) == 4
    assert {"a": 1, "b": 10} in out
    assert {"a": 2, "b": 20} in out


# ---------- registry --------------------------------------------------------


def test_registry_contains_reference_strategies():
    names = available_strategies()
    assert "sma_cross" in names
    assert "rsi_meanrev" in names


def test_get_strategy_returns_class():
    cls = get_strategy("sma_cross")
    assert cls is SMACross


def test_register_rejects_unnamed_class():
    class Anon(Strategy):
        name = "base"

        @classmethod
        def param_specs(cls):
            return []

        def _compute_signals(self, data, **params):
            raise NotImplementedError

    with pytest.raises(StrategyError):
        register(Anon)


def test_register_rejects_duplicate_name():
    class Dup(Strategy):
        name = "sma_cross"

        @classmethod
        def param_specs(cls):
            return []

        def _compute_signals(self, data, **params):
            raise NotImplementedError

    with pytest.raises(StrategyError):
        register(Dup)


# ---------- SMACross --------------------------------------------------------


def test_sma_cross_produces_signals():
    data = _ohlcv(300)
    sig = SMACross().generate_signals(data, fast=10, slow=30)
    assert isinstance(sig, Signals)
    assert sig.entries.dtype == bool
    assert sig.exits.dtype == bool
    assert sig.entries.index.equals(data.df.index)
    # On a 300-bar drifting-up series, we expect at least one cross of each.
    assert sig.entries.sum() >= 1
    assert sig.exits.sum() >= 1


def test_sma_cross_degenerate_fast_ge_slow_emits_nothing():
    data = _ohlcv(100)
    sig = SMACross().generate_signals(data, fast=30, slow=10)
    assert sig.entries.sum() == 0
    assert sig.exits.sum() == 0


def test_sma_cross_rejects_unknown_param():
    data = _ohlcv(50)
    with pytest.raises(StrategyError):
        SMACross().generate_signals(data, fast=5, slow=20, bogus=1)


# ---------- RSIMeanReversion ------------------------------------------------


def test_rsi_meanrev_produces_signals():
    data = _ohlcv(400, seed=2)
    sig = RSIMeanReversion().generate_signals(data, period=14, lower=30.0, upper=70.0)
    assert sig.entries.dtype == bool
    assert sig.exits.dtype == bool
    # We aren't asserting a specific count — only that the index lines up and
    # we got at least *some* oversold crosses on a 400-bar random walk.
    assert sig.entries.sum() >= 1


def test_rsi_meanrev_param_bounds_reject_inverted_thresholds():
    # The schema's bounds (lower in [5,45], upper in [55,95]) make it
    # impossible to construct an inverted pair through validate_params.
    data = _ohlcv(100)
    with pytest.raises(StrategyError):
        RSIMeanReversion().generate_signals(data, period=14, lower=70.0, upper=30.0)


# ---------- no-look-ahead invariant -----------------------------------------


def _strategy_params_for_no_lookahead(cls):
    """Pick a parameter set per strategy whose signals on the test data are
    non-trivially populated, so the comparison actually has something to
    check."""
    if cls is SMACross:
        return {"fast": 5, "slow": 20}
    if cls is RSIMeanReversion:
        return {"period": 14, "lower": 30.0, "upper": 70.0}
    return cls.default_params()


@pytest.mark.parametrize("name", available_strategies())
def test_no_lookahead_for_all_strategies(name):
    """Mutating future bars must not change past signals.

    For each registered strategy: run on the full series, then run on a
    series where bars after a cut-point have been replaced with adversarial
    values. The first ``cut`` signals must be byte-for-byte identical.
    """
    cls = get_strategy(name)
    strat = cls()
    params = _strategy_params_for_no_lookahead(cls)

    data = _ohlcv(300, seed=7)
    full_sig = strat.generate_signals(data, **params)

    cut = 150
    mutated_df = data.df.copy()
    # Slam future bars with values wildly different from the history; if any
    # indicator peeks forward, the past signals will shift.
    mutated_df.iloc[cut:, mutated_df.columns.get_loc("close")] *= 1e3
    mutated_df.iloc[cut:, mutated_df.columns.get_loc("open")] *= 1e3
    mutated_df.iloc[cut:, mutated_df.columns.get_loc("high")] *= 1e3
    mutated_df.iloc[cut:, mutated_df.columns.get_loc("low")] *= 1e3

    mutated = OHLCV(
        df=mutated_df,
        spec=data.spec,
        timeframe=data.timeframe,
        source=data.source,
    )
    mutated_sig = strat.generate_signals(mutated, **params)

    pd.testing.assert_series_equal(
        full_sig.entries.iloc[:cut],
        mutated_sig.entries.iloc[:cut],
        check_names=False,
    )
    pd.testing.assert_series_equal(
        full_sig.exits.iloc[:cut],
        mutated_sig.exits.iloc[:cut],
        check_names=False,
    )


# ---------- discoverability check ------------------------------------------


def test_registry_param_schemas_introspectable():
    """The dashboard needs the schema, not just the class, to render the UI."""
    for name in available_strategies():
        cls = strategy_registry[name]
        specs = cls.param_specs()
        assert isinstance(specs, list)
        assert all(isinstance(s, ParamSpec) for s in specs)
        # default_params() should produce values that pass validation.
        cls._validate_params(cls.default_params())
