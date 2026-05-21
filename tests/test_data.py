"""Tests for the data layer.

These exercise the normalization contract, the parquet cache, the file
adapter (real), and use mocking to drive the network-bound adapters
(ccxt, yfinance, alpaca, databento, ibkr) without touching the network.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import pytest

from quant_dashboard.data import (
    AlpacaSource,
    AssetClass,
    CCXTSource,
    DatabentoSource,
    DataSourceError,
    FileSource,
    IBKRSource,
    InstrumentSpec,
    OHLCV,
    ParquetCache,
    YFinanceSource,
    normalize_ohlcv,
)
from quant_dashboard.data.futures_source import _root_from_symbol


# ---------- fixtures --------------------------------------------------------


@pytest.fixture
def cache_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("QUANT_DASHBOARD_CACHE_DIR", str(tmp_path / "cache"))
    return tmp_path / "cache"


def _synthetic_df(n: int = 60, start: str = "2024-01-01") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="1D", tz="UTC", name="timestamp")
    rng = np.random.default_rng(0)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    high = close + rng.uniform(0.1, 0.5, n)
    low = close - rng.uniform(0.1, 0.5, n)
    open_ = close + rng.normal(0, 0.2, n)
    vol = rng.uniform(1_000, 5_000, n)
    return pd.DataFrame(
        {"open": open_, "high": high, "low": low, "close": close, "volume": vol},
        index=idx,
    )


# ---------- normalize_ohlcv -------------------------------------------------


def test_normalize_lowercases_and_orders_columns():
    raw = _synthetic_df(10).rename(
        columns={"open": "Open", "high": "HIGH", "low": "Low", "close": "Close", "volume": "VOLUME"}
    )
    out = normalize_ohlcv(raw)
    assert list(out.columns) == ["open", "high", "low", "close", "volume"]
    assert str(out.index.tz) == "UTC"
    assert out.index.is_monotonic_increasing


def test_normalize_uses_timestamp_column():
    df = _synthetic_df(5)
    df.index.name = "ts"
    raw = df.reset_index()
    out = normalize_ohlcv(raw, timestamp_col="ts")
    assert isinstance(out.index, pd.DatetimeIndex)
    assert str(out.index.tz) == "UTC"


def test_normalize_drops_duplicates_and_sorts():
    df = _synthetic_df(5)
    shuffled = pd.concat([df.iloc[[3, 1, 4, 2, 0, 0]]])  # out of order + dup
    out = normalize_ohlcv(shuffled)
    assert out.index.is_monotonic_increasing
    assert not out.index.has_duplicates
    assert len(out) == 5


def test_normalize_missing_column_raises():
    raw = _synthetic_df(3).drop(columns=["volume"])
    with pytest.raises(DataSourceError):
        normalize_ohlcv(raw)


# ---------- OHLCV validation ------------------------------------------------


def test_ohlcv_rejects_naive_index():
    df = _synthetic_df(3).tz_convert(None)
    with pytest.raises(DataSourceError):
        OHLCV(df=df, spec=InstrumentSpec("X", AssetClass.OTHER), timeframe="1d", source="t")


def test_ohlcv_rejects_nan_ohlc():
    df = _synthetic_df(3).copy()
    df.iloc[1, df.columns.get_loc("close")] = np.nan
    with pytest.raises(DataSourceError):
        OHLCV(df=df, spec=InstrumentSpec("X", AssetClass.OTHER), timeframe="1d", source="t")


# ---------- ParquetCache ----------------------------------------------------


def test_parquet_cache_roundtrip(cache_dir):
    cache = ParquetCache.default()
    df = _synthetic_df(10)
    cache.save("ccxt/BTC_USDT/1d/A_B", df)
    assert cache.has("ccxt/BTC_USDT/1d/A_B")
    loaded = cache.load("ccxt/BTC_USDT/1d/A_B")
    pd.testing.assert_frame_equal(loaded, df, check_freq=False)
    assert cache.clear("ccxt/BTC_USDT") == 1
    assert not cache.has("ccxt/BTC_USDT/1d/A_B")


# ---------- FileSource (real) -----------------------------------------------


def test_file_source_csv_roundtrip(tmp_path, cache_dir):
    df = _synthetic_df(8)
    csv = tmp_path / "spy.csv"
    out = df.reset_index().rename(columns={"timestamp": "timestamp"})
    out.to_csv(csv, index=False)

    src = FileSource(asset_class=AssetClass.EQUITY)
    bundle = src.fetch(str(csv), "1d", start="2024-01-01", end="2024-01-31")
    assert isinstance(bundle, OHLCV)
    assert bundle.spec.asset_class == AssetClass.EQUITY
    assert bundle.spec.symbol == str(csv)
    assert bundle.spec.meta["display_name"] == "spy"
    assert list(bundle.df.columns) == ["open", "high", "low", "close", "volume"]
    assert bundle.df.index.is_monotonic_increasing


def test_file_source_parquet_roundtrip(tmp_path, cache_dir):
    df = _synthetic_df(8)
    pq = tmp_path / "btc.parquet"
    df.to_parquet(pq)

    src = FileSource(asset_class=AssetClass.CRYPTO)
    bundle = src.fetch(str(pq), "1d")
    assert bundle.spec.asset_class == AssetClass.CRYPTO
    assert len(bundle.df) == 8


def test_file_source_missing_file_raises(cache_dir):
    src = FileSource()
    with pytest.raises(DataSourceError):
        src.fetch("/nonexistent/path.csv", "1d")


# ---------- CCXTSource (mocked) ---------------------------------------------


def test_ccxt_fetch_mocked(cache_dir):
    src = CCXTSource(exchange="binance")

    fake_rows = [
        [int(pd.Timestamp("2024-01-01", tz="UTC").timestamp() * 1000) + i * 86_400_000,
         100 + i, 101 + i, 99 + i, 100.5 + i, 1000 + i]
        for i in range(20)
    ]

    fake_exchange = mock.MagicMock()
    fake_exchange.has = {"fetchOHLCV": True}
    fake_exchange.fetch_ohlcv.return_value = fake_rows
    src._exchange = fake_exchange

    bundle = src.fetch("BTC/USDT", "1d", start="2024-01-01", end="2024-01-20")
    assert bundle.source == "ccxt"
    assert bundle.spec.asset_class == AssetClass.CRYPTO
    assert bundle.spec.quote_currency == "USDT"
    assert bundle.spec.session == "24x7"
    assert len(bundle.df) == 20
    assert list(bundle.df.columns) == ["open", "high", "low", "close", "volume"]


def test_ccxt_unknown_timeframe_raises(cache_dir):
    src = CCXTSource()
    src._exchange = mock.MagicMock(has={"fetchOHLCV": True})
    with pytest.raises(DataSourceError):
        src.fetch("BTC/USDT", "7m", start="2024-01-01")


# ---------- YFinanceSource (mocked) -----------------------------------------


def test_yfinance_fetch_mocked(cache_dir):
    src = YFinanceSource()
    fake = _synthetic_df(10).rename(
        columns={"open": "Open", "high": "High", "low": "Low", "close": "Close", "volume": "Volume"}
    )
    with mock.patch("yfinance.download", return_value=fake):
        bundle = src.fetch("SPY", "1d", start="2024-01-01", end="2024-01-31")
    assert bundle.spec.asset_class == AssetClass.EQUITY
    assert bundle.spec.exchange == "yahoo"
    assert len(bundle.df) == 10


def test_yfinance_empty_raises(cache_dir):
    src = YFinanceSource()
    with mock.patch("yfinance.download", return_value=pd.DataFrame()):
        with pytest.raises(DataSourceError):
            src.fetch("ZZZZ", "1d", start="2024-01-01", end="2024-01-31")


# ---------- AlpacaSource (gating) -------------------------------------------


def test_alpaca_missing_keys_raises(cache_dir, monkeypatch):
    monkeypatch.delenv("ALPACA_API_KEY", raising=False)
    monkeypatch.delenv("ALPACA_API_SECRET", raising=False)
    src = AlpacaSource()
    with pytest.raises(DataSourceError, match="ALPACA_API_KEY"):
        src.fetch("SPY", "1d", start="2024-01-01", end="2024-01-31")


# ---------- DatabentoSource (gating + spec) ---------------------------------


def test_databento_missing_key_raises(cache_dir, monkeypatch):
    monkeypatch.delenv("DATABENTO_API_KEY", raising=False)
    src = DatabentoSource()
    with pytest.raises(DataSourceError, match="DATABENTO_API_KEY"):
        src.fetch("ESZ4", "1d", start="2024-01-01", end="2024-01-31")


def test_databento_requires_explicit_range(cache_dir, monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "fake")
    src = DatabentoSource()
    src._client = lambda: mock.MagicMock()  # noqa: ARG005
    with pytest.raises(DataSourceError, match="explicit start and end"):
        src.fetch("ESZ4", "1d")


def test_futures_spec_from_registry(cache_dir, monkeypatch):
    monkeypatch.setenv("DATABENTO_API_KEY", "fake")
    src = DatabentoSource()
    spec = src.instrument_spec("ESZ4")
    assert spec.asset_class == AssetClass.FUTURE
    assert spec.multiplier == 50.0
    assert spec.tick_size == 0.25
    assert spec.tick_value == 12.50
    assert spec.meta["root"] == "ES"


def test_futures_spec_overrides_win(cache_dir):
    src = DatabentoSource(spec_overrides={"multiplier": 7.0, "tick_value": 1.0})
    spec = src.instrument_spec("XYZH5")
    assert spec.multiplier == 7.0
    assert spec.tick_value == 1.0


def test_root_extraction():
    assert _root_from_symbol("ESZ4") == "ES"
    assert _root_from_symbol("MES.c.0") == "MES"
    assert _root_from_symbol("ZN") == "ZN"
    assert _root_from_symbol("123") == "123"


# ---------- IBKRSource (gating) ---------------------------------------------


def test_ibkr_connect_failure_raises(cache_dir, monkeypatch):
    monkeypatch.setenv("IBKR_HOST", "127.0.0.1")
    monkeypatch.setenv("IBKR_PORT", "59999")  # unlikely to be open
    src = IBKRSource()

    fake_ib = mock.MagicMock()
    fake_ib.connect.side_effect = ConnectionError("refused")
    with mock.patch("ib_insync.IB", return_value=fake_ib):
        with pytest.raises(DataSourceError, match="Gateway"):
            src.fetch("ES", "1d", start="2024-01-01", end="2024-01-31")


def test_ibkr_spec_is_futures(cache_dir):
    src = IBKRSource(contract_type="future", contract_kwargs={"exchange": "CME"})
    spec = src.instrument_spec("ES")
    assert spec.asset_class == AssetClass.FUTURE
    assert spec.exchange == "CME"
    assert spec.multiplier == 50.0


# ---------- caching wiring (via FileSource for determinism) -----------------


def test_cache_is_hit_on_second_fetch(tmp_path, cache_dir):
    df = _synthetic_df(6)
    csv = tmp_path / "x.csv"
    df.reset_index().to_csv(csv, index=False)

    src = FileSource()
    first = src.fetch(str(csv), "1d")
    # Mutate the source file; if the cache works, the second read returns the
    # original contents because the cached parquet is loaded instead.
    df2 = _synthetic_df(6, start="2030-01-01")
    df2.reset_index().to_csv(csv, index=False)
    second = src.fetch(str(csv), "1d")

    pd.testing.assert_frame_equal(first.df, second.df, check_freq=False)
