"""Volume features (tick volume for FX). Skipped when the data has no volume."""

from __future__ import annotations

import pandas as pd

from fxgen.indicators import core
from fxgen.indicators.base import Feature


def _obv_flow(df: pd.DataFrame) -> pd.Series:
    obv = core.obv(df)
    return (obv - obv.shift(10)) / df["volume"].rolling(10, min_periods=10).sum()


FEATURES: list[Feature] = [
    Feature(
        name="volume_z",
        family="volume",
        label="zscore(volume, 50)",
        compute=lambda df: core.zscore(df["volume"], 50),
        pine="(volume - ta.sma(volume, 50)) / ta.stdev(volume, 50)",
        signed=True,
        needs_volume=True,
    ),
    Feature(
        name="obv_flow10",
        family="volume",
        label="10-bar OBV change / 10-bar volume",
        compute=_obv_flow,
        pine="(ta.obv - ta.obv[10]) / math.sum(volume, 10)",
        signed=True,
        needs_volume=True,
    ),
    Feature(
        name="mfi14",
        family="volume",
        label="Money flow index(14)",
        levels=(20.0, 80.0),
        compute=lambda df: core.mfi(df, 14),
        pine="ta.mfi(hlc3, 14)",
        needs_volume=True,
    ),
]


def _cmf(df: pd.DataFrame) -> pd.Series:
    rng = (df["high"] - df["low"]).replace(0, float("nan"))
    mfv = ((df["close"] - df["low"]) - (df["high"] - df["close"])) / rng * df["volume"]
    return mfv.rolling(20, min_periods=20).sum() / df["volume"].rolling(20, min_periods=20).sum()


def _vwma(df: pd.DataFrame, n: int) -> pd.Series:
    return (df["close"] * df["volume"]).rolling(n, min_periods=n).sum() / df["volume"].rolling(n, min_periods=n).sum()


FEATURES += [
    Feature(
        name="volume_rank",
        family="volume",
        label="percentrank(volume, 100)",
        compute=lambda df: core.percentrank(df["volume"], 100),
        pine="ta.percentrank(volume, 100)",
        levels=(10.0, 90.0),
        needs_volume=True,
    ),
    Feature(
        name="volume_ratio",
        family="volume",
        label="SMA(volume, 5) / SMA(volume, 50)",
        compute=lambda df: core.sma(df["volume"], 5) / core.sma(df["volume"], 50),
        pine="ta.sma(volume, 5) / ta.sma(volume, 50)",
        levels=(1.0,),
        needs_volume=True,
    ),
    Feature(
        name="cmf20",
        family="volume",
        label="Chaikin money flow(20)",
        compute=_cmf,
        pine=("_mfv = ((close - low) - (high - close)) / (high - low) * volume\n"
              "math.sum(_mfv, 20) / math.sum(volume, 20)"),
        signed=True,
        crosses=True,
        needs_volume=True,
    ),
    Feature(
        name="dist_vwma20",
        family="volume",
        label="(close - VWMA20) / ATR14",
        compute=lambda df: (df["close"] - _vwma(df, 20)) / core.atr(df, 14),
        pine="(close - ta.vwma(close, 20)) / ta.atr(14)",
        signed=True,
        crosses=True,
        needs_volume=True,
    ),
    Feature(
        name="force13",
        family="volume",
        label="EMA13(change x volume) / (SMA13(volume) x ATR14)",
        compute=lambda df: core.ema(df["close"].diff() * df["volume"], 13)
        / (core.sma(df["volume"], 13) * core.atr(df, 14)),
        pine="ta.ema(ta.change(close) * volume, 13) / (ta.sma(volume, 13) * ta.atr(14))",
        signed=True,
        crosses=True,
        needs_volume=True,
    ),
]
