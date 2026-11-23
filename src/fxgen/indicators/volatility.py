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
        levels=(0.0, 0.5, 1.0),
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
        levels=(38.2, 61.8),
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


def _ulcer(df, n):  # type: ignore[no-untyped-def]
    peak = core.highest(df["close"], n)
    dd = 100 * (df["close"] - peak) / peak
    return np.sqrt((dd**2).rolling(n, min_periods=n).sum() / n)


def _mass_index(df):  # type: ignore[no-untyped-def]
    rng = df["high"] - df["low"]
    e1 = core.ema(rng, 9)
    return (e1 / core.ema(e1, 9)).rolling(25, min_periods=25).sum()


def _bb_pctb_len(df, n):  # type: ignore[no-untyped-def]
    basis = core.sma(df["close"], n)
    dev = 2 * core.stdev(df["close"], n)
    return (df["close"] - (basis - dev)) / (2 * dev)


FEATURES += [
    Feature(
        name="atr_rank50",
        family="volatility",
        label="percentrank(ATR14, 50)",
        compute=lambda df: core.percentrank(core.atr(df, 14), 50),
        pine="ta.percentrank(ta.atr(14), 50)",
    ),
    Feature(
        name="atr_ratio14_100",
        family="volatility",
        label="ATR14 / ATR100",
        compute=lambda df: core.atr(df, 14) / core.atr(df, 100),
        pine="ta.atr(14) / ta.atr(100)",
        levels=(1.0,),
    ),
    Feature(
        name="hv_rank",
        family="volatility",
        label="percentrank(stdev(log ret, 20), 250)",
        compute=lambda df: core.percentrank(core.stdev(np.log(df["close"]).diff(), 20), 250),
        pine="ta.percentrank(ta.stdev(math.log(close / close[1]), 20), 250)",
    ),
    Feature(
        name="bb_pctb50",
        family="volatility",
        label="Bollinger %B(50,2)",
        compute=lambda df: _bb_pctb_len(df, 50),
        pine="(close - (ta.sma(close, 50) - 2 * ta.stdev(close, 50))) / (4 * ta.stdev(close, 50))",
        levels=(0.0, 0.5, 1.0),
    ),
    Feature(
        name="bb_width_z",
        family="volatility",
        label="zscore(Bollinger width(20,2), 100)",
        compute=lambda df: core.zscore(4 * core.stdev(df["close"], 20) / core.sma(df["close"], 20), 100),
        pine="_w = 4 * ta.stdev(close, 20) / ta.sma(close, 20)\n(_w - ta.sma(_w, 100)) / ta.stdev(_w, 100)",
        signed=True,
    ),
    Feature(
        name="squeeze",
        family="volatility",
        label="Bollinger(20,2) width / Keltner(20,1.5 ATR) width (< 1 = squeeze)",
        compute=lambda df: (2 * core.stdev(df["close"], 20)) / (1.5 * core.atr(df, 20)),
        pine="(2 * ta.stdev(close, 20)) / (1.5 * ta.atr(20))",
        levels=(1.0,),
    ),
    Feature(
        name="keltner_pos",
        family="volatility",
        label="(close - EMA20) / (2 x ATR10) (Keltner position)",
        compute=lambda df: (df["close"] - core.ema(df["close"], 20)) / (2 * core.atr(df, 10)),
        pine="(close - ta.ema(close, 20)) / (2 * ta.atr(10))",
        signed=True,
        levels=(-1.0, 1.0),
    ),
    Feature(
        name="ulcer14",
        family="volatility",
        label="Ulcer index(14) (%)",
        compute=lambda df: _ulcer(df, 14),
        pine="_peak = ta.highest(close, 14)\nmath.sqrt(math.sum(math.pow(100 * (close - _peak) / _peak, 2), 14) / 14)",
    ),
    Feature(
        name="mass_index",
        family="volatility",
        label="Mass index(9, 25)",
        compute=_mass_index,
        pine="_e1 = ta.ema(high - low, 9)\nmath.sum(_e1 / ta.ema(_e1, 9), 25)",
        levels=(26.5, 27.0),
    ),
    Feature(
        name="chop50",
        family="volatility",
        label="Choppiness index(50)",
        compute=lambda df: core.choppiness(df, 50),
        pine="100 * math.log10(math.sum(ta.atr(1), 50) / (ta.highest(high, 50) - ta.lowest(low, 50))) / math.log10(50)",
        levels=(38.2, 61.8),
    ),
    Feature(
        name="range_rank",
        family="volatility",
        label="percentrank(high - low, 100)",
        compute=lambda df: core.percentrank(df["high"] - df["low"], 100),
        pine="ta.percentrank(high - low, 100)",
        levels=(10.0, 90.0),
    ),
    Feature(
        name="range_ratio5_20",
        family="volatility",
        label="mean range 5 / mean range 20",
        compute=lambda df: core.sma(df["high"] - df["low"], 5) / core.sma(df["high"] - df["low"], 20),
        pine="ta.sma(high - low, 5) / ta.sma(high - low, 20)",
        levels=(1.0,),
    ),
]
