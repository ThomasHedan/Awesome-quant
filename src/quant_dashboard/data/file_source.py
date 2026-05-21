"""Local CSV / Parquet adapter.

Expects a file with columns (case-insensitive):
``timestamp, open, high, low, close, volume``. ``timestamp`` is parsed as UTC.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from quant_dashboard.data.base import (
    AssetClass,
    DataSource,
    DataSourceError,
    InstrumentSpec,
)


class FileSource(DataSource):
    """Loads OHLCV from a local file. ``symbol`` is the file path."""

    name = "file"
    default_asset_class = AssetClass.OTHER

    def __init__(
        self,
        *,
        asset_class: AssetClass = AssetClass.OTHER,
        spec_overrides: dict | None = None,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.default_asset_class = asset_class
        self._spec_overrides = dict(spec_overrides or {})

    def _fetch_raw(self, symbol, timeframe, start, end) -> pd.DataFrame:
        path = Path(symbol).expanduser()
        if not path.exists():
            raise DataSourceError(f"file not found: {path}")
        suffix = path.suffix.lower()
        if suffix in {".parquet", ".pq"}:
            df = pd.read_parquet(path)
            # Find the timestamp column if it isn't already the index.
            ts_col = None
            for c in df.columns:
                if c.lower() == "timestamp":
                    ts_col = c
                    break
            if ts_col is not None:
                df.index = pd.to_datetime(df[ts_col], utc=True)
                df = df.drop(columns=[ts_col])
        elif suffix in {".csv", ".tsv", ".txt"}:
            sep = "\t" if suffix == ".tsv" else ","
            df = pd.read_csv(path, sep=sep)
            ts_col = None
            for c in df.columns:
                if c.lower() == "timestamp":
                    ts_col = c
                    break
            if ts_col is None:
                raise DataSourceError(
                    f"{path}: CSV must include a 'timestamp' column"
                )
            df.index = pd.to_datetime(df[ts_col], utc=True)
            df = df.drop(columns=[ts_col])
        else:
            raise DataSourceError(
                f"{path}: unsupported file type (use .csv, .tsv, or .parquet)"
            )
        return df

    def instrument_spec(self, symbol: str) -> InstrumentSpec:
        # Keep the full path as the symbol so a run can be replayed from disk
        # via fetch(symbol, ...). Use Path(symbol).stem for display only.
        return InstrumentSpec(
            symbol=symbol,
            asset_class=self.default_asset_class,
            meta={"display_name": Path(symbol).stem},
            **self._spec_overrides,
        )
