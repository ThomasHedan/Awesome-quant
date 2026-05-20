"""Data adapters.

Every adapter implements :class:`DataSource` and returns a normalized
:class:`OHLCV` (a DataFrame plus an :class:`InstrumentSpec`) so the rest of
the system stays asset-agnostic.
"""

from quant_dashboard.data.base import (
    AssetClass,
    DataSource,
    DataSourceError,
    InstrumentSpec,
    OHLCV,
    OHLCV_COLUMNS,
    normalize_ohlcv,
)
from quant_dashboard.data.cache import ParquetCache
from quant_dashboard.data.ccxt_source import CCXTSource
from quant_dashboard.data.file_source import FileSource
from quant_dashboard.data.futures_source import DatabentoSource, IBKRSource
from quant_dashboard.data.yfinance_source import AlpacaSource, YFinanceSource

__all__ = [
    "AlpacaSource",
    "AssetClass",
    "CCXTSource",
    "DatabentoSource",
    "DataSource",
    "DataSourceError",
    "FileSource",
    "IBKRSource",
    "InstrumentSpec",
    "OHLCV",
    "OHLCV_COLUMNS",
    "ParquetCache",
    "YFinanceSource",
    "normalize_ohlcv",
]
