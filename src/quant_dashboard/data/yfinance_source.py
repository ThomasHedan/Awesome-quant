"""Equities adapters: yfinance (free) and Alpaca (keyed).

Both produce the same normalized OHLCV. Choose yfinance for daily/long
history; Alpaca when you have keys and want intraday with no rate-limit
roulette.
"""

from __future__ import annotations

import os

import pandas as pd

from quant_dashboard.data.base import (
    AssetClass,
    DataSource,
    DataSourceError,
    InstrumentSpec,
)


# ---------- yfinance --------------------------------------------------------

# yfinance's interval strings map almost 1:1 to ours, but a few differ.
_YF_INTERVAL: dict[str, str] = {
    "1m": "1m",
    "2m": "2m",
    "5m": "5m",
    "15m": "15m",
    "30m": "30m",
    "60m": "60m",
    "1h": "60m",
    "1d": "1d",
    "1w": "1wk",
    "1mo": "1mo",
}


class YFinanceSource(DataSource):
    """Equities / ETFs / indices via Yahoo Finance."""

    name = "yfinance"
    default_asset_class = AssetClass.EQUITY

    def _fetch_raw(self, symbol, timeframe, start, end) -> pd.DataFrame:
        if timeframe not in _YF_INTERVAL:
            raise DataSourceError(
                f"yfinance: unsupported timeframe '{timeframe}'. "
                f"Try one of: {sorted(_YF_INTERVAL)}"
            )
        try:
            import yfinance as yf
        except ImportError as exc:  # pragma: no cover
            raise DataSourceError("yfinance is not installed.") from exc

        interval = _YF_INTERVAL[timeframe]
        kwargs = {
            "interval": interval,
            "auto_adjust": True,
            "progress": False,
            "actions": False,
        }
        if start is not None:
            kwargs["start"] = start.tz_convert("UTC").to_pydatetime()
        if end is not None:
            kwargs["end"] = end.tz_convert("UTC").to_pydatetime()

        try:
            df = yf.download(symbol, **kwargs)
        except Exception as exc:
            raise DataSourceError(f"yfinance download failed: {exc}") from exc

        if df is None or df.empty:
            raise DataSourceError(
                f"yfinance: empty result for {symbol} ({interval}). "
                f"Check the symbol and date range."
            )

        # yfinance returns a MultiIndex when multiple tickers are requested;
        # for a single ticker it may still wrap columns. Flatten defensively.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df.columns = [str(c).lower() for c in df.columns]
        if "adj close" in df.columns and "close" not in df.columns:
            df = df.rename(columns={"adj close": "close"})
        return df

    def instrument_spec(self, symbol: str) -> InstrumentSpec:
        return InstrumentSpec(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            quote_currency="USD",
            multiplier=1.0,
            tick_size=0.01,
            tick_value=0.01,
            exchange="yahoo",
            session="regular",
        )


# ---------- Alpaca ----------------------------------------------------------

_ALPACA_TIMEFRAME = {
    "1m": ("Minute", 1),
    "5m": ("Minute", 5),
    "15m": ("Minute", 15),
    "30m": ("Minute", 30),
    "1h": ("Hour", 1),
    "1d": ("Day", 1),
}


class AlpacaSource(DataSource):
    """Equities OHLCV via the Alpaca Market Data API.

    Reads credentials from ``ALPACA_API_KEY`` / ``ALPACA_API_SECRET``. Fails
    loudly with an actionable message if either is missing.
    """

    name = "alpaca"
    default_asset_class = AssetClass.EQUITY

    def __init__(self, *, feed: str = "iex", **kwargs) -> None:
        super().__init__(**kwargs)
        self.feed = feed

    def _client(self):
        try:
            from alpaca.data.historical import StockHistoricalDataClient
        except ImportError as exc:  # pragma: no cover
            raise DataSourceError("alpaca-py is not installed.") from exc
        key = os.environ.get("ALPACA_API_KEY")
        secret = os.environ.get("ALPACA_API_SECRET")
        if not key or not secret:
            raise DataSourceError(
                "Alpaca credentials missing: set ALPACA_API_KEY and "
                "ALPACA_API_SECRET in your environment (see .env.sample)."
            )
        return StockHistoricalDataClient(key, secret)

    def _fetch_raw(self, symbol, timeframe, start, end) -> pd.DataFrame:
        if timeframe not in _ALPACA_TIMEFRAME:
            raise DataSourceError(
                f"alpaca: unsupported timeframe '{timeframe}'. "
                f"Try one of: {sorted(_ALPACA_TIMEFRAME)}"
            )
        from alpaca.data.requests import StockBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit

        unit_name, amount = _ALPACA_TIMEFRAME[timeframe]
        tf = TimeFrame(amount, getattr(TimeFrameUnit, unit_name))
        client = self._client()
        try:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=tf,
                start=start.to_pydatetime() if start is not None else None,
                end=end.to_pydatetime() if end is not None else None,
                feed=self.feed,
            )
            resp = client.get_stock_bars(req)
        except Exception as exc:
            raise DataSourceError(f"alpaca fetch failed: {exc}") from exc

        df = resp.df
        if df is None or df.empty:
            raise DataSourceError(f"alpaca: empty result for {symbol} {timeframe}")

        # Alpaca returns a MultiIndex (symbol, timestamp).
        if isinstance(df.index, pd.MultiIndex):
            df = df.xs(symbol, level=0)
        return df

    def instrument_spec(self, symbol: str) -> InstrumentSpec:
        return InstrumentSpec(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            quote_currency="USD",
            multiplier=1.0,
            tick_size=0.01,
            tick_value=0.01,
            exchange="alpaca",
            session="regular",
        )
