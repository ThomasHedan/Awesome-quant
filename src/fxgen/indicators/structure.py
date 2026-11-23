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


def _bars_since_high(length: int) -> Feature:
    return Feature(
        name=f"bars_since_high{length}",
        family="structure",
        label=f"bars since {length}-bar high",
        compute=lambda df: core.bars_since_highest(df["high"], length),
        pine=f"-ta.highestbars(high, {length})",
    )


def _bars_since_low(length: int) -> Feature:
    return Feature(
        name=f"bars_since_low{length}",
        family="structure",
        label=f"bars since {length}-bar low",
        compute=lambda df: core.bars_since_lowest(df["low"], length),
        pine=f"-ta.lowestbars(low, {length})",
    )


def _range_pos_close(length: int) -> Feature:
    def compute(df: pd.DataFrame) -> pd.Series:
        lo, hi = core.lowest(df["close"], length), core.highest(df["close"], length)
        return (df["close"] - lo) / (hi - lo).replace(0, np.nan)

    return Feature(
        name=f"close_range_pos{length}",
        family="structure",
        label=f"position of close in its {length}-bar closing range",
        compute=compute,
        pine=f"(close - ta.lowest(close, {length})) / (ta.highest(close, {length}) - ta.lowest(close, {length}))",
        levels=(0.5,),
    )


FEATURES += [
    _donchian_pos(10),
    _donchian_pos(100),
    _dist_high(10),
    _dist_low(10),
    _dist_high(50),
    _dist_low(50),
    _dist_high(250),
    _dist_low(250),
    _bars_since_high(50),
    _bars_since_low(50),
    _bars_since_high(100),
    _bars_since_low(100),
    _range_pos_close(20),
    _range_pos_close(100),
    Feature(
        name="streak",
        family="structure",
        label="consecutive up (+) / down (-) closes",
        compute=lambda df: core.streak(df["close"]),
        pine=("var float _s = 0.0\n_s := close > close[1] ? (_s > 0 ? _s + 1 : 1) : close < close[1] ? "
              "(_s < 0 ? _s - 1 : -1) : 0\n_s"),
        signed=True,
        levels=(-3.0, 3.0),
    ),
    Feature(
        name="dist_donchian_mid20",
        family="structure",
        label="(close - Donchian(20) middle) / ATR14",
        compute=lambda df: (df["close"] - core.donchian_mid(df, 20)) / core.atr(df, 14),
        pine="(close - (ta.highest(high, 20) + ta.lowest(low, 20)) / 2) / ta.atr(14)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="breakout_up20",
        family="structure",
        label="close above the previous 20-bar high (in ATR)",
        compute=lambda df: (df["close"] - core.highest(df["high"], 20).shift(1)) / core.atr(df, 14),
        pine="(close - ta.highest(high, 20)[1]) / ta.atr(14)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="breakout_down20",
        family="structure",
        label="close below the previous 20-bar low (in ATR, positive = below)",
        compute=lambda df: (core.lowest(df["low"], 20).shift(1) - df["close"]) / core.atr(df, 14),
        pine="(ta.lowest(low, 20)[1] - close) / ta.atr(14)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="breakout_up55",
        family="structure",
        label="close above the previous 55-bar high (in ATR)",
        compute=lambda df: (df["close"] - core.highest(df["high"], 55).shift(1)) / core.atr(df, 14),
        pine="(close - ta.highest(high, 55)[1]) / ta.atr(14)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="breakout_down55",
        family="structure",
        label="close below the previous 55-bar low (in ATR, positive = below)",
        compute=lambda df: (core.lowest(df["low"], 55).shift(1) - df["close"]) / core.atr(df, 14),
        pine="(ta.lowest(low, 55)[1] - close) / ta.atr(14)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="body_atr",
        family="structure",
        label="(close - open) / ATR14",
        compute=lambda df: (df["close"] - df["open"]) / core.atr(df, 14),
        pine="(close - open) / ta.atr(14)",
        signed=True,
    ),
    Feature(
        name="body_sum5",
        family="structure",
        label="sum of the last 5 bodies / ATR14",
        compute=lambda df: (df["close"] - df["open"]).rolling(5, min_periods=5).sum() / core.atr(df, 14),
        pine="math.sum(close - open, 5) / ta.atr(14)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="up_bars10",
        family="structure",
        label="share of up closes over 10 bars",
        compute=lambda df: (df["close"].diff() > 0).astype(float).rolling(10, min_periods=10).mean(),
        pine="math.sum(close > close[1] ? 1 : 0, 10) / 10",
        levels=(0.3, 0.7),
    ),
]
