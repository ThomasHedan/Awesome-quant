"""Parquet-backed OHLCV cache.

Keyed by an opaque string (built by :meth:`DataSource._cache_key`). We don't
attempt smart partial-range merging — different windows are separate cache
entries. Simple, predictable, deletable.
"""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_DEFAULT_DIR = ".cache/quant-dashboard"


def _key_to_path(root: Path, key: str) -> Path:
    # The key already uses '/' as a separator; whitelist the rest.
    safe = re.sub(r"[^A-Za-z0-9_./-]", "_", key)
    return (root / f"{safe}.parquet").resolve()


@dataclass
class ParquetCache:
    root: Path

    @classmethod
    def default(cls) -> "ParquetCache":
        env = os.environ.get("QUANT_DASHBOARD_CACHE_DIR", _DEFAULT_DIR)
        root = Path(env).expanduser()
        root.mkdir(parents=True, exist_ok=True)
        return cls(root=root)

    def has(self, key: str) -> bool:
        return _key_to_path(self.root, key).exists()

    def load(self, key: str) -> pd.DataFrame:
        path = _key_to_path(self.root, key)
        df = pd.read_parquet(path)
        # Parquet round-trips tz-aware indices, but be defensive.
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        return df

    def save(self, key: str, df: pd.DataFrame) -> None:
        path = _key_to_path(self.root, key)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(path)

    def clear(self, prefix: str | None = None) -> int:
        """Delete cached entries. Returns the number removed."""
        removed = 0
        for p in self.root.rglob("*.parquet"):
            rel = p.relative_to(self.root).as_posix().removesuffix(".parquet")
            if prefix is None or rel.startswith(prefix):
                p.unlink()
                removed += 1
        return removed
