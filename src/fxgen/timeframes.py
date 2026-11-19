"""Timeframe durations and FX instrument helpers."""

from __future__ import annotations

import pandas as pd

from fxgen.errors import DataError

TIMEFRAME_DURATION: dict[str, pd.Timedelta] = {
    "H1": pd.Timedelta(hours=1),
    "H4": pd.Timedelta(hours=4),
    "D1": pd.Timedelta(days=1),
}
BARS_PER_YEAR: dict[str, float] = {"H1": 24 * 260, "H4": 6 * 260, "D1": 260}


def duration(timeframe: str) -> pd.Timedelta:
    """Return the bar duration of a timeframe."""
    try:
        return TIMEFRAME_DURATION[timeframe]
    except KeyError as exc:
        raise DataError(f"unknown timeframe {timeframe!r}; use one of {sorted(TIMEFRAME_DURATION)}") from exc


def pip_size(pair: str) -> float:
    """Pip size of an FX pair (0.01 for JPY quotes, 0.0001 otherwise)."""
    return 0.01 if pair.upper().endswith("JPY") else 0.0001
