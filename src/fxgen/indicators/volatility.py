"""Volatility features."""

from __future__ import annotations

import numpy as np

from fxgen.indicators import core
from fxgen.indicators.base import Feature


def _bb_width(df):  # type: ignore[no-untyped-def]
    basis = core.sma(df["close"], 20)
    return 4 * core.stdev(df["close"], 20) / basis


def _bb_pctb(df):  # type: ignore[no-untyped-def]
    basis = core.sma(df["close"], 20)
    dev = 2 * core.stdev(df["close"], 20)
    return (df["close"] - (basis - dev)) / (2 * dev)


def _ret_vol_ratio(df):  # type: ignore[no-untyped-def]
    ret = np.log(df["close"]).diff()
    return core.stdev(ret, 10) / core.stdev(ret, 100)


FEATURES: list[Feature] = [
    Feature(
        name="atr_rank",
        family="volatility",
        label="percentrank(ATR14, 250)",
        compute=lambda df: core.percentrank(core.atr(df, 14), 250),
        pine="ta.percentrank(ta.atr(14), 250)",
    ),
    Feature(
        name="atr_ratio",
        family="volatility",
        label="ATR5 / ATR50",
        compute=lambda df: core.atr(df, 5) / core.atr(df, 50),
        pine="ta.atr(5) / ta.atr(50)",
    ),
    Feature(
        name="bb_width_rank",
        family="volatility",
        label="percentrank(Bollinger width(20,2), 250)",
        compute=lambda df: core.percentrank(_bb_width(df), 250),
        pine="ta.percentrank(4 * ta.stdev(close, 20) / ta.sma(close, 20), 250)",
    ),
    Feature(
        name="bb_pctb",
        family="volatility",
        label="Bollinger %B(20,2)",
        compute=_bb_pctb,
        pine="(close - (ta.sma(close, 20) - 2 * ta.stdev(close, 20))) / (4 * ta.stdev(close, 20))",
    ),
    Feature(
        name="range_atr",
        family="volatility",
        label="(high - low) / ATR14",
        compute=lambda df: (df["high"] - df["low"]) / core.atr(df, 14),
        pine="(high - low) / ta.atr(14)",
    ),
    Feature(
        name="ret_vol_ratio",
        family="volatility",
        label="stdev(log ret, 10) / stdev(log ret, 100)",
        compute=_ret_vol_ratio,
        pine="ta.stdev(math.log(close / close[1]), 10) / ta.stdev(math.log(close / close[1]), 100)",
    ),
    Feature(
        name="chop14",
        family="volatility",
        label="Choppiness index(14)",
        compute=lambda df: core.choppiness(df, 14),
        pine="100 * math.log10(math.sum(ta.atr(1), 14) / (ta.highest(high, 14) - ta.lowest(low, 14))) / math.log10(14)",
    ),
    Feature(
        name="atr_z",
        family="volatility",
        label="zscore(ATR14, 250)",
        compute=lambda df: core.zscore(core.atr(df, 14), 250),
        pine="(ta.atr(14) - ta.sma(ta.atr(14), 250)) / ta.stdev(ta.atr(14), 250)",
        signed=True,
    ),
]
