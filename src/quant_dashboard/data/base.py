"""Core data-layer types: :class:`InstrumentSpec`, :class:`OHLCV`, and the
:class:`DataSource` abstract base.

All adapters MUST return a normalized OHLCV DataFrame:

* UTC ``DatetimeIndex`` named ``timestamp``
* columns ``[open, high, low, close, volume]`` as ``float64``
* monotonically increasing, deduplicated
* no NaN in OHLC (volume may be 0)

Downstream code (strategies, engine, metrics) trusts this contract, so any
new adapter must funnel its raw response through :func:`normalize_ohlcv`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Iterable

import pandas as pd

OHLCV_COLUMNS: tuple[str, ...] = ("open", "high", "low", "close", "volume")


class DataSourceError(RuntimeError):
    """Raised by adapters for any user-actionable failure: missing API keys,
    unreachable gateways, empty result sets, bad symbols, etc.

    The dashboard catches this and surfaces ``str(exc)`` directly to the user,
    so messages should be plain English and tell the user what to do.
    """


class AssetClass(str, Enum):
    CRYPTO = "crypto"
    EQUITY = "equity"
    ETF = "etf"
    FUTURE = "future"
    OTHER = "other"


@dataclass(frozen=True)
class InstrumentSpec:
    """Travels alongside the OHLCV frame so fees, slippage, and P&L are
    computed in the right units regardless of asset class.

    For crypto and equities ``multiplier`` is 1.0 and ``tick_value`` equals
    ``tick_size`` in the quote currency. Futures override both.
    """

    symbol: str
    asset_class: AssetClass
    quote_currency: str = "USD"
    multiplier: float = 1.0
    tick_size: float = 0.01
    tick_value: float = 0.01
    exchange: str | None = None
    session: str | None = None  # "24x7", "regular", "futures", ...
    meta: dict = field(default_factory=dict)


@dataclass(frozen=True)
class OHLCV:
    """A normalized OHLCV frame plus the spec of the instrument it describes."""

    df: pd.DataFrame
    spec: InstrumentSpec
    timeframe: str
    source: str

    def __post_init__(self) -> None:
        _validate_ohlcv(self.df)

    @property
    def close(self) -> pd.Series:
        return self.df["close"]


def _validate_ohlcv(df: pd.DataFrame) -> None:
    if not isinstance(df.index, pd.DatetimeIndex):
        raise DataSourceError("OHLCV index must be a DatetimeIndex.")
    if df.index.tz is None or str(df.index.tz) != "UTC":
        raise DataSourceError("OHLCV index must be tz-aware UTC.")
    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataSourceError(f"OHLCV missing required columns: {missing}.")
    if not df.index.is_monotonic_increasing:
        raise DataSourceError("OHLCV index must be sorted ascending.")
    if df.index.has_duplicates:
        raise DataSourceError("OHLCV index has duplicate timestamps.")
    ohlc = df[["open", "high", "low", "close"]]
    if ohlc.isna().any().any():
        raise DataSourceError("OHLCV has NaN in OHLC columns.")


def _to_utc(ts: datetime | str | pd.Timestamp | None) -> pd.Timestamp | None:
    if ts is None:
        return None
    out = pd.Timestamp(ts)
    if out.tzinfo is None:
        out = out.tz_localize(timezone.utc)
    else:
        out = out.tz_convert("UTC")
    return out


def normalize_ohlcv(
    raw: pd.DataFrame,
    *,
    column_map: dict[str, str] | None = None,
    timestamp_col: str | None = None,
) -> pd.DataFrame:
    """Coerce a vendor-shaped frame into the canonical OHLCV layout.

    Parameters
    ----------
    raw:
        Source frame. May have a column for the timestamp or use the index.
    column_map:
        Optional mapping of source-column -> canonical-column. Case-insensitive
        lookups are also tried, so common shapes (Open/High/Low/Close/Volume,
        OPEN/HIGH/...) work without an explicit map.
    timestamp_col:
        If set, the named column becomes the index (parsed as UTC). Otherwise
        the existing index is used.
    """
    df = raw.copy()

    if timestamp_col is not None:
        if timestamp_col not in df.columns:
            raise DataSourceError(f"timestamp column '{timestamp_col}' not in frame")
        df.index = pd.to_datetime(df[timestamp_col], utc=True)
        df = df.drop(columns=[timestamp_col])
    else:
        df.index = pd.to_datetime(df.index, utc=True)

    df.index.name = "timestamp"

    rename: dict[str, str] = {}
    if column_map:
        rename.update(column_map)
    # Fallback: case-insensitive match for the canonical column names.
    lower = {c.lower(): c for c in df.columns}
    for canon in OHLCV_COLUMNS:
        if canon in df.columns:
            continue
        if canon in lower:
            rename[lower[canon]] = canon
    if rename:
        df = df.rename(columns=rename)

    missing = [c for c in OHLCV_COLUMNS if c not in df.columns]
    if missing:
        raise DataSourceError(
            f"raw frame is missing required columns after normalization: {missing}"
        )

    df = df[list(OHLCV_COLUMNS)].astype("float64")
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    # Drop rows where OHLC are all NaN (some vendors emit these for halts).
    df = df.dropna(subset=["open", "high", "low", "close"], how="any")
    df["volume"] = df["volume"].fillna(0.0)
    return df


def slice_window(
    df: pd.DataFrame,
    start: datetime | str | None,
    end: datetime | str | None,
) -> pd.DataFrame:
    """Inclusive slice by UTC timestamp."""
    s = _to_utc(start)
    e = _to_utc(end)
    if s is not None:
        df = df.loc[df.index >= s]
    if e is not None:
        df = df.loc[df.index <= e]
    return df


class DataSource(ABC):
    """Abstract base class for every OHLCV adapter.

    Subclasses implement :meth:`_fetch_raw` and (usually) declare a default
    :class:`InstrumentSpec`. :meth:`fetch` is the public entry point — it
    coordinates caching, normalization, and validation so adapters stay small.
    """

    name: str = "base"
    default_asset_class: AssetClass = AssetClass.OTHER

    def __init__(self, *, cache: "ParquetCache | None" = None) -> None:
        from quant_dashboard.data.cache import ParquetCache

        self._cache = cache if cache is not None else ParquetCache.default()

    # --- public API ------------------------------------------------------

    def fetch(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | str | None = None,
        end: datetime | str | None = None,
        *,
        use_cache: bool = True,
    ) -> OHLCV:
        """Return a validated :class:`OHLCV` for ``symbol``.

        Adapters are responsible for connecting to the vendor and producing a
        raw DataFrame; this method handles caching, normalization, slicing,
        validation, and attaching an :class:`InstrumentSpec`.
        """
        key = self._cache_key(symbol, timeframe, start, end)
        if use_cache and self._cache.has(key):
            df = self._cache.load(key)
        else:
            df = self._fetch_raw(symbol, timeframe, _to_utc(start), _to_utc(end))
            df = normalize_ohlcv(df)
            if use_cache:
                self._cache.save(key, df)

        df = slice_window(df, start, end)
        if df.empty:
            raise DataSourceError(
                f"{self.name}: no rows returned for {symbol} {timeframe} "
                f"[{start}..{end}]"
            )
        spec = self.instrument_spec(symbol)
        return OHLCV(df=df, spec=spec, timeframe=timeframe, source=self.name)

    def instrument_spec(self, symbol: str) -> InstrumentSpec:
        """Default spec — adapters override for asset-class-specific contract
        details (futures multipliers, tick values, etc.).
        """
        return InstrumentSpec(symbol=symbol, asset_class=self.default_asset_class)

    # --- subclass hooks --------------------------------------------------

    @abstractmethod
    def _fetch_raw(
        self,
        symbol: str,
        timeframe: str,
        start: pd.Timestamp | None,
        end: pd.Timestamp | None,
    ) -> pd.DataFrame:
        """Vendor-specific fetch; return a frame normalize_ohlcv can chew on."""

    # --- helpers ---------------------------------------------------------

    def _cache_key(
        self,
        symbol: str,
        timeframe: str,
        start: datetime | str | None,
        end: datetime | str | None,
    ) -> str:
        s = _to_utc(start)
        e = _to_utc(end)
        s_txt = s.strftime("%Y%m%dT%H%M%S") if s is not None else "open"
        e_txt = e.strftime("%Y%m%dT%H%M%S") if e is not None else "open"
        safe_sym = symbol.replace("/", "_").replace(":", "_").replace(" ", "_")
        return f"{self.name}/{safe_sym}/{timeframe}/{s_txt}_{e_txt}"


def concat_ohlcv(parts: Iterable[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate already-normalized OHLCV frames, deduping by timestamp."""
    parts = [p for p in parts if p is not None and not p.empty]
    if not parts:
        return pd.DataFrame(columns=list(OHLCV_COLUMNS))
    out = pd.concat(parts, axis=0).sort_index()
    out = out[~out.index.duplicated(keep="last")]
    return out
