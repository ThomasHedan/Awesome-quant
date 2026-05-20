"""Futures adapters: Databento (historical) and IBKR (live/paper via Gateway).

Both produce normalized OHLCV with a futures :class:`InstrumentSpec` carrying
``multiplier``, ``tick_size``, and ``tick_value`` so downstream P&L is in the
right units. A minimal contract registry lets common roots resolve without
the caller passing specs every time; unknown roots fall back to sane defaults
and accept ``spec_overrides`` to plug the gap.

Both adapters fail fast with a plain-English error if their dependencies
(API key, running Gateway) aren't available. They never crash the dashboard.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import pandas as pd

from quant_dashboard.data.base import (
    AssetClass,
    DataSource,
    DataSourceError,
    InstrumentSpec,
)


@dataclass(frozen=True)
class _ContractSpec:
    multiplier: float
    tick_size: float
    tick_value: float
    quote_currency: str = "USD"


# Minimal seed of common roots. Extend by passing ``spec_overrides`` to
# the adapter or by editing this table — both adapters consult it before
# falling back to a generic default.
_FUTURES_REGISTRY: dict[str, _ContractSpec] = {
    "ES": _ContractSpec(multiplier=50.0, tick_size=0.25, tick_value=12.50),
    "MES": _ContractSpec(multiplier=5.0, tick_size=0.25, tick_value=1.25),
    "NQ": _ContractSpec(multiplier=20.0, tick_size=0.25, tick_value=5.00),
    "MNQ": _ContractSpec(multiplier=2.0, tick_size=0.25, tick_value=0.50),
    "CL": _ContractSpec(multiplier=1000.0, tick_size=0.01, tick_value=10.00),
    "GC": _ContractSpec(multiplier=100.0, tick_size=0.10, tick_value=10.00),
    "ZN": _ContractSpec(multiplier=1000.0, tick_size=1 / 64, tick_value=15.625),
}


_MONTH_CODES = set("FGHJKMNQUVXZ")


def _root_from_symbol(symbol: str) -> str:
    """Extract the futures root from a symbol like ``ESZ4`` or ``MES.c.0``.

    Rule: take the leading alphabetic run; if the symbol continues with digits
    or another segment (``.c.0``) and the last alpha is a futures month
    letter, drop it. Standalone roots (``ZN``) are preserved.
    """
    out: list[str] = []
    for ch in symbol:
        if ch.isalpha():
            out.append(ch)
        else:
            break
    prefix = "".join(out).upper()
    if not prefix:
        return symbol.upper()
    rest = symbol[len(out):]
    if rest and len(prefix) > 1 and prefix[-1] in _MONTH_CODES:
        return prefix[:-1]
    return prefix


def _build_futures_spec(
    symbol: str,
    *,
    exchange: str | None,
    spec_overrides: dict | None = None,
) -> InstrumentSpec:
    root = _root_from_symbol(symbol)
    base = _FUTURES_REGISTRY.get(root, _ContractSpec(1.0, 0.01, 0.01))
    overrides = dict(spec_overrides or {})
    return InstrumentSpec(
        symbol=symbol,
        asset_class=AssetClass.FUTURE,
        quote_currency=overrides.pop("quote_currency", base.quote_currency),
        multiplier=overrides.pop("multiplier", base.multiplier),
        tick_size=overrides.pop("tick_size", base.tick_size),
        tick_value=overrides.pop("tick_value", base.tick_value),
        exchange=overrides.pop("exchange", exchange),
        session=overrides.pop("session", "futures"),
        meta={"root": root, **overrides},
    )


# ---------- Databento -------------------------------------------------------

_DATABENTO_SCHEMA = {
    "1m": "ohlcv-1m",
    "5m": "ohlcv-1m",   # resampled below if needed
    "1h": "ohlcv-1h",
    "1d": "ohlcv-1d",
}

_RESAMPLE_RULE = {"5m": "5min", "15m": "15min", "30m": "30min"}


class DatabentoSource(DataSource):
    """Futures OHLCV via Databento's Historical API.

    By default uses the ``GLBX.MDP3`` dataset (CME Globex). For continuous
    contracts, pass e.g. ``ES.c.0`` and set ``stype_in='continuous'``.
    """

    name = "databento"
    default_asset_class = AssetClass.FUTURE

    def __init__(
        self,
        *,
        dataset: str = "GLBX.MDP3",
        stype_in: str = "raw_symbol",
        spec_overrides: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.dataset = dataset
        self.stype_in = stype_in
        self._spec_overrides = dict(spec_overrides or {})

    def _client(self):
        try:
            import databento as db
        except ImportError as exc:  # pragma: no cover
            raise DataSourceError("databento is not installed.") from exc
        key = os.environ.get("DATABENTO_API_KEY")
        if not key:
            raise DataSourceError(
                "Databento API key missing: set DATABENTO_API_KEY in your "
                "environment (see .env.sample)."
            )
        return db.Historical(key)

    def _fetch_raw(self, symbol, timeframe, start, end) -> pd.DataFrame:
        if start is None or end is None:
            raise DataSourceError("databento: explicit start and end are required.")
        schema = _DATABENTO_SCHEMA.get(timeframe)
        if schema is None:
            raise DataSourceError(
                f"databento: unsupported timeframe '{timeframe}'. "
                f"Try one of: {sorted(_DATABENTO_SCHEMA)}"
            )
        client = self._client()
        try:
            data = client.timeseries.get_range(
                dataset=self.dataset,
                symbols=[symbol],
                schema=schema,
                stype_in=self.stype_in,
                start=start.tz_convert("UTC").to_pydatetime(),
                end=end.tz_convert("UTC").to_pydatetime(),
            )
            df = data.to_df()
        except Exception as exc:
            raise DataSourceError(f"databento fetch failed: {exc}") from exc

        if df is None or df.empty:
            raise DataSourceError(f"databento: empty result for {symbol} {timeframe}")

        # Databento DataFrames are tz-aware UTC and use 'ts_event' as the index.
        if df.index.name != "timestamp":
            df.index.name = "timestamp"

        rule = _RESAMPLE_RULE.get(timeframe)
        if rule is not None:
            df = (
                df[["open", "high", "low", "close", "volume"]]
                .resample(rule)
                .agg({
                    "open": "first",
                    "high": "max",
                    "low": "min",
                    "close": "last",
                    "volume": "sum",
                })
                .dropna(subset=["open", "high", "low", "close"])
            )
        return df

    def instrument_spec(self, symbol: str) -> InstrumentSpec:
        return _build_futures_spec(
            symbol,
            exchange=self.dataset,
            spec_overrides=self._spec_overrides,
        )


# ---------- Interactive Brokers --------------------------------------------

_IB_BAR_SIZE = {
    "1m": "1 min",
    "5m": "5 mins",
    "15m": "15 mins",
    "30m": "30 mins",
    "1h": "1 hour",
    "1d": "1 day",
}


class IBKRSource(DataSource):
    """Futures (or stock) OHLCV via IBKR's TWS / Gateway, through ib_insync.

    Requires a running Gateway or TWS reachable at ``IBKR_HOST``:``IBKR_PORT``
    with the API enabled. Use ``contract_kwargs`` to override the default
    ``Future`` contract (e.g. for a specific expiry).
    """

    name = "ibkr"
    default_asset_class = AssetClass.FUTURE

    def __init__(
        self,
        *,
        contract_type: str = "future",
        contract_kwargs: dict | None = None,
        spec_overrides: dict | None = None,
        what_to_show: str = "TRADES",
        use_rth: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.contract_type = contract_type
        self.contract_kwargs = dict(contract_kwargs or {})
        self._spec_overrides = dict(spec_overrides or {})
        self.what_to_show = what_to_show
        self.use_rth = use_rth

    def _connect(self):
        try:
            from ib_insync import IB
        except ImportError as exc:  # pragma: no cover
            raise DataSourceError("ib_insync is not installed.") from exc
        host = os.environ.get("IBKR_HOST", "127.0.0.1")
        port = int(os.environ.get("IBKR_PORT", "7497"))
        client_id = int(os.environ.get("IBKR_CLIENT_ID", "17"))
        ib = IB()
        try:
            ib.connect(host, port, clientId=client_id, timeout=10)
        except Exception as exc:
            raise DataSourceError(
                f"IBKR connection failed at {host}:{port} — is Gateway / TWS "
                f"running with the API enabled? ({exc})"
            ) from exc
        return ib

    def _build_contract(self, symbol: str):
        from ib_insync import ContFuture, Future, Stock

        if self.contract_type == "future":
            return Future(symbol=symbol, **self.contract_kwargs)
        if self.contract_type == "cont_future":
            return ContFuture(symbol=symbol, **self.contract_kwargs)
        if self.contract_type == "stock":
            kw = {"exchange": "SMART", "currency": "USD", **self.contract_kwargs}
            return Stock(symbol=symbol, **kw)
        raise DataSourceError(f"ibkr: unknown contract_type '{self.contract_type}'")

    def _duration_for(self, start: pd.Timestamp, end: pd.Timestamp) -> str:
        # IB wants e.g. "30 D", "2 W", "1 M", "1 Y".
        days = max(1, int((end - start).total_seconds() // 86_400) + 1)
        if days <= 31:
            return f"{days} D"
        if days <= 365:
            weeks = max(1, days // 7)
            return f"{weeks} W"
        years = max(1, days // 365)
        return f"{years} Y"

    def _fetch_raw(self, symbol, timeframe, start, end) -> pd.DataFrame:
        if start is None or end is None:
            raise DataSourceError("ibkr: explicit start and end are required.")
        if timeframe not in _IB_BAR_SIZE:
            raise DataSourceError(
                f"ibkr: unsupported timeframe '{timeframe}'. "
                f"Try one of: {sorted(_IB_BAR_SIZE)}"
            )

        ib = self._connect()
        try:
            contract = self._build_contract(symbol)
            ib.qualifyContracts(contract)
            bars = ib.reqHistoricalData(
                contract,
                endDateTime=end.tz_convert("UTC").to_pydatetime(),
                durationStr=self._duration_for(start, end),
                barSizeSetting=_IB_BAR_SIZE[timeframe],
                whatToShow=self.what_to_show,
                useRTH=self.use_rth,
                formatDate=2,  # epoch seconds
            )
        except DataSourceError:
            raise
        except Exception as exc:
            raise DataSourceError(f"ibkr historical request failed: {exc}") from exc
        finally:
            try:
                ib.disconnect()
            except Exception:
                pass

        if not bars:
            raise DataSourceError(f"ibkr: empty result for {symbol} {timeframe}")

        rows = []
        for b in bars:
            rows.append(
                {
                    "timestamp": pd.Timestamp(b.date, tz="UTC"),
                    "open": float(b.open),
                    "high": float(b.high),
                    "low": float(b.low),
                    "close": float(b.close),
                    "volume": float(b.volume) if b.volume is not None else 0.0,
                }
            )
        df = pd.DataFrame(rows).set_index("timestamp")
        return df

    def instrument_spec(self, symbol: str) -> InstrumentSpec:
        if self.contract_type in {"future", "cont_future"}:
            return _build_futures_spec(
                symbol,
                exchange=self.contract_kwargs.get("exchange"),
                spec_overrides=self._spec_overrides,
            )
        return InstrumentSpec(
            symbol=symbol,
            asset_class=AssetClass.EQUITY,
            multiplier=1.0,
            tick_size=0.01,
            tick_value=0.01,
            exchange=self.contract_kwargs.get("exchange", "SMART"),
            session="regular",
        )
