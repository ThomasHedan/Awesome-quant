"""Tests for the multi-chart runner."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant_dashboard.data import AssetClass, InstrumentSpec, OHLCV
from quant_dashboard.data.base import DataSource
from quant_dashboard.engine import CostsConfig, MultiChartError, run_multi_chart
from quant_dashboard.strategies import SMACross


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _ohlcv(n: int = 300, seed: int = 1, symbol: str = "TEST") -> OHLCV:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0005, 0.012, n)
    close = 100 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC", name="timestamp")
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
        spec=InstrumentSpec(symbol=symbol, asset_class=AssetClass.EQUITY),
        timeframe="1d",
        source="synthetic",
    )


class _FakeSource(DataSource):
    """Deterministic in-memory data source for tests."""

    def __init__(self, seed: int = 1, fail: bool = False) -> None:
        super().__init__()
        self._seed = seed
        self._fail = fail

    def _fetch_raw(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> pd.DataFrame:
        if self._fail:
            raise RuntimeError(f"Simulated fetch failure for {symbol}")
        return _ohlcv(seed=self._seed, symbol=symbol).df


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_run_multi_chart_basic() -> None:
    sources = [
        (_FakeSource(seed=1), "AAA", "1d"),
        (_FakeSource(seed=2), "BBB", "1d"),
        (_FakeSource(seed=3), "CCC", "1d"),
    ]
    result = run_multi_chart(
        sources, SMACross(), params={"fast": 10, "slow": 30},
        costs=CostsConfig(), start="2022-01-01",
    )
    assert result.strategy_name == "sma_cross"
    assert len(result.instruments) == 3
    assert len(result.successful) == 3
    assert len(result.failed) == 0


def test_summary_df_columns() -> None:
    sources = [(_FakeSource(seed=1), "SPY", "1d")]
    result = run_multi_chart(sources, SMACross(), params={"fast": 10, "slow": 30})
    df = result.summary_df
    for col in ("symbol", "timeframe", "sharpe", "cagr_pct", "max_drawdown_pct"):
        assert col in df.columns, f"Missing column: {col}"


def test_per_instrument_failure_does_not_abort() -> None:
    sources = [
        (_FakeSource(seed=1), "GOOD", "1d"),
        (_FakeSource(fail=True), "BAD", "1d"),
    ]
    result = run_multi_chart(sources, SMACross(), params={"fast": 10, "slow": 30})
    assert len(result.successful) == 1
    assert len(result.failed) == 1
    assert result.failed[0].symbol == "BAD"
    assert "Simulated fetch failure" in result.failed[0].error


def test_empty_sources_raises() -> None:
    with pytest.raises(MultiChartError, match="must not be empty"):
        run_multi_chart([], SMACross(), params={})


def test_non_strategy_raises() -> None:
    with pytest.raises(MultiChartError, match="Strategy instance"):
        run_multi_chart([(_FakeSource(), "X", "1d")], object(), params={})  # type: ignore[arg-type]


def test_default_params_used_when_params_empty() -> None:
    sources = [(_FakeSource(seed=1), "SPY", "1d")]
    result = run_multi_chart(sources, SMACross(), params={})
    defaults = SMACross().default_params()
    assert result.params == defaults


def test_summary_df_all_failed_has_error_col() -> None:
    sources = [(_FakeSource(fail=True), "X", "1d")]
    result = run_multi_chart(sources, SMACross(), params={"fast": 10, "slow": 30})
    assert len(result.failed) == 1
    df = result.summary_df
    assert "error" in df.columns
