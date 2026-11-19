"""Resampling of bars into higher timeframes."""

from __future__ import annotations

import pandas as pd


def resample(bars: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample lower-timeframe bars into H4 or D1 aligned on the 22:00 UTC daily roll."""
    rule = {"H4": "4h", "D1": "24h"}[timeframe]
    out = bars.resample(rule, origin="epoch", offset="22h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open"])
