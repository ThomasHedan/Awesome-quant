"""Price-structure and location features."""

from __future__ import annotations

import numpy as np
import pandas as pd

from fxgen.indicators import core
from fxgen.indicators.base import Feature


def _donchian_pos(length: int) -> Feature:
    def compute(df: pd.DataFrame) -> pd.Series:
        lo, hi = core.lowest(df["low"], length), core.highest(df["high"], length)
        return (df["close"] - lo) / (hi - lo).replace(0, np.nan)

    return Feature(
        name=f"donchian_pos{length}",
        family="structure",
        label=f"position in Donchian({length}) channel",
        compute=compute,
        pine=f"(close - ta.lowest(low, {length})) / (ta.highest(high, {length}) - ta.lowest(low, {length}))",
    )


def _dist_high(length: int) -> Feature:
    return Feature(
        name=f"dist_high{length}",
        family="structure",
        label=f"(highest high {length} - close) / ATR14",
        compute=lambda df: (core.highest(df["high"], length) - df["close"]) / core.atr(df, 14),
        pine=f"(ta.highest(high, {length}) - close) / ta.atr(14)",
    )


def _dist_low(length: int) -> Feature:
    return Feature(
        name=f"dist_low{length}",
        family="structure",
        label=f"(close - lowest low {length}) / ATR14",
        compute=lambda df: (df["close"] - core.lowest(df["low"], length)) / core.atr(df, 14),
        pine=f"(close - ta.lowest(low, {length})) / ta.atr(14)",
    )


def _bar_range(df: pd.DataFrame) -> pd.Series:
    return (df["high"] - df["low"]).replace(0, np.nan)


FEATURES: list[Feature] = [
    _donchian_pos(20),
    _donchian_pos(55),
    _dist_high(20),
    _dist_low(20),
    _dist_high(100),
    _dist_low(100),
    Feature(
        name="bars_since_high20",
        family="structure",
        label="bars since 20-bar high",
        compute=lambda df: core.bars_since_highest(df["high"], 20),
        pine="-ta.highestbars(high, 20)",
    ),
    Feature(
        name="bars_since_low20",
        family="structure",
        label="bars since 20-bar low",
        compute=lambda df: core.bars_since_lowest(df["low"], 20),
        pine="-ta.lowestbars(low, 20)",
    ),
    Feature(
        name="close_location",
        family="structure",
        label="(close - low) / (high - low)",
        compute=lambda df: (df["close"] - df["low"]) / _bar_range(df),
        pine="(close - low) / (high - low)",
    ),
    Feature(
        name="body_ratio",
        family="structure",
        label="(close - open) / (high - low)",
        compute=lambda df: (df["close"] - df["open"]) / _bar_range(df),
        pine="(close - open) / (high - low)",
        signed=True,
    ),
    Feature(
        name="upper_wick",
        family="structure",
        label="upper wick / range",
        compute=lambda df: (df["high"] - df[["open", "close"]].max(axis=1)) / _bar_range(df),
        pine="(high - math.max(open, close)) / (high - low)",
    ),
    Feature(
        name="lower_wick",
        family="structure",
        label="lower wick / range",
        compute=lambda df: (df[["open", "close"]].min(axis=1) - df["low"]) / _bar_range(df),
        pine="(math.min(open, close) - low) / (high - low)",
    ),
    Feature(
        name="prev_bar_return",
        family="structure",
        label="(close[1] - open[1]) / ATR14",
        compute=lambda df: (df["close"].shift(1) - df["open"].shift(1)) / core.atr(df, 14),
        pine="(close[1] - open[1]) / ta.atr(14)",
        signed=True,
    ),
]
