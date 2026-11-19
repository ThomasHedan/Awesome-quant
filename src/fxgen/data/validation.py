"""Data-quality checks on OHLCV frames: duplicates, gaps, broken bars, timezone."""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from fxgen.errors import DataError
from fxgen.timeframes import duration

logger = logging.getLogger(__name__)

OHLC = ["open", "high", "low", "close"]


@dataclass(frozen=True)
class DataQuality:
    """Summary of the checks run on one pair/timeframe file."""

    pair: str
    timeframe: str
    n_bars: int
    start: str
    end: str
    duplicates_removed: int
    nan_rows_removed: int
    ohlc_repaired: int
    n_gaps: int
    max_gap_hours: float

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _is_weekend_gap(prev: pd.Timestamp, nxt: pd.Timestamp) -> bool:
    if (nxt - prev) > pd.Timedelta(days=4):
        return False
    days = pd.date_range(prev.normalize(), nxt.normalize(), freq="D")
    return bool((days.day_of_week == 5).any())


def count_gaps(index: pd.DatetimeIndex, timeframe: str) -> tuple[int, float]:
    """Count missing-bar gaps that are not explained by the FX weekend close."""
    if len(index) < 2:
        return 0, 0.0
    step = duration(timeframe)
    deltas = index[1:] - index[:-1]
    candidates = np.flatnonzero(deltas > step)
    gaps = [i for i in candidates if not _is_weekend_gap(index[i], index[i + 1])]
    max_gap = max((deltas[i] / pd.Timedelta(hours=1) for i in gaps), default=0.0)
    return len(gaps), float(max_gap)


def clean_and_validate(df: pd.DataFrame, pair: str, timeframe: str) -> tuple[pd.DataFrame, DataQuality]:
    """Sort, dedupe, drop NaN rows, repair inconsistent high/low and report quality.

    The input must have a tz-aware DatetimeIndex (bar open time) and OHLC columns.
    """
    if not isinstance(df.index, pd.DatetimeIndex) or df.index.tz is None:
        raise DataError(f"{pair} {timeframe}: index must be a timezone-aware DatetimeIndex")
    df = df.tz_convert("UTC") if str(df.index.tz) != "UTC" else df
    df = df.sort_index()

    duplicated = df.index.duplicated(keep="last")
    n_dup = int(duplicated.sum())
    if n_dup:
        logger.warning("%s %s: removed %d duplicated timestamps", pair, timeframe, n_dup)
        df = df[~duplicated]

    nan_mask = df[OHLC].isna().any(axis=1)
    n_nan = int(nan_mask.sum())
    if n_nan:
        logger.warning("%s %s: removed %d rows with NaN prices", pair, timeframe, n_nan)
        df = df[~nan_mask]

    if df.empty:
        raise DataError(f"{pair} {timeframe}: no valid bars after cleaning")
    if (df[OHLC] <= 0).any().any():
        raise DataError(f"{pair} {timeframe}: non-positive prices found; check the column mapping")

    body_high = df[["open", "close"]].max(axis=1)
    body_low = df[["open", "close"]].min(axis=1)
    broken = (df["high"] < body_high) | (df["low"] > body_low) | (df["high"] < df["low"])
    n_broken = int(broken.sum())
    if n_broken:
        logger.warning("%s %s: repaired %d bars with inconsistent high/low", pair, timeframe, n_broken)
        df = df.copy()
        df["high"] = np.maximum(df["high"], body_high)
        df["low"] = np.minimum(df["low"], body_low)

    n_gaps, max_gap = count_gaps(df.index, timeframe)
    if n_gaps:
        logger.info("%s %s: %d non-weekend gaps (max %.0f h)", pair, timeframe, n_gaps, max_gap)

    quality = DataQuality(
        pair=pair,
        timeframe=timeframe,
        n_bars=len(df),
        start=df.index[0].isoformat(),
        end=df.index[-1].isoformat(),
        duplicates_removed=n_dup,
        nan_rows_removed=n_nan,
        ohlc_repaired=n_broken,
        n_gaps=n_gaps,
        max_gap_hours=max_gap,
    )
    return df, quality
