"""CCXT crypto adapter.

OHLCV from any CCXT exchange (Binance by default). Public endpoints only; no
API keys required for historical bars.
"""

from __future__ import annotations

import time

import pandas as pd

from quant_dashboard.data.base import (
    AssetClass,
    DataSource,
    DataSourceError,
    InstrumentSpec,
)

# CCXT timeframe in ms — keeps us from importing ccxt just to compute deltas.
_TF_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
    "3d": 3 * 24 * 60 * 60_000,
    "1w": 7 * 24 * 60 * 60_000,
}


class CCXTSource(DataSource):
    """Crypto OHLCV via CCXT. ``symbol`` follows CCXT's ``BASE/QUOTE``."""

    name = "ccxt"
    default_asset_class = AssetClass.CRYPTO

    def __init__(
        self,
        exchange: str = "binance",
        *,
        page_limit: int = 1000,
        rate_limit_sleep: float = 0.25,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.exchange_id = exchange
        self.page_limit = page_limit
        self.rate_limit_sleep = rate_limit_sleep
        self._exchange = None  # lazy

    def _client(self):
        if self._exchange is not None:
            return self._exchange
        try:
            import ccxt
        except ImportError as exc:  # pragma: no cover - dependency pinned
            raise DataSourceError("ccxt is not installed.") from exc
        if not hasattr(ccxt, self.exchange_id):
            raise DataSourceError(
                f"ccxt has no exchange named '{self.exchange_id}'."
            )
        klass = getattr(ccxt, self.exchange_id)
        self._exchange = klass({"enableRateLimit": True})
        return self._exchange

    def _fetch_raw(self, symbol, timeframe, start, end) -> pd.DataFrame:
        if timeframe not in _TF_MS:
            raise DataSourceError(
                f"ccxt: unsupported timeframe '{timeframe}'. "
                f"Try one of: {sorted(_TF_MS)}"
            )
        ex = self._client()
        if not getattr(ex, "has", {}).get("fetchOHLCV", False):
            raise DataSourceError(
                f"ccxt: exchange '{self.exchange_id}' does not support OHLCV"
            )

        step = _TF_MS[timeframe]
        since = int(start.timestamp() * 1000) if start is not None else None
        # ccxt only paginates if we give it a since; fall back to one page.
        if since is None:
            try:
                rows = ex.fetch_ohlcv(symbol, timeframe=timeframe, limit=self.page_limit)
            except Exception as exc:
                raise DataSourceError(f"ccxt fetch failed: {exc}") from exc
        else:
            end_ms = int(end.timestamp() * 1000) if end is not None else None
            rows = []
            cursor = since
            while True:
                try:
                    batch = ex.fetch_ohlcv(
                        symbol,
                        timeframe=timeframe,
                        since=cursor,
                        limit=self.page_limit,
                    )
                except Exception as exc:
                    raise DataSourceError(f"ccxt fetch failed: {exc}") from exc
                if not batch:
                    break
                rows.extend(batch)
                last_ts = batch[-1][0]
                if end_ms is not None and last_ts >= end_ms:
                    break
                if len(batch) < self.page_limit:
                    break
                cursor = last_ts + step
                time.sleep(self.rate_limit_sleep)

        if not rows:
            raise DataSourceError(f"ccxt: empty result for {symbol} {timeframe}")

        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume"])
        df.index = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        df = df.drop(columns=["timestamp"])
        return df

    def instrument_spec(self, symbol: str) -> InstrumentSpec:
        quote = symbol.split("/")[-1] if "/" in symbol else "USD"
        return InstrumentSpec(
            symbol=symbol,
            asset_class=AssetClass.CRYPTO,
            quote_currency=quote,
            multiplier=1.0,
            tick_size=0.01,
            tick_value=0.01,
            exchange=self.exchange_id,
            session="24x7",
        )
