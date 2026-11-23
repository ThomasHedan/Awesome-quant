"""Momentum features."""

from __future__ import annotations

import numpy as np

from fxgen.indicators import core
from fxgen.indicators.base import Feature


def _rsi(length: int, levels: tuple[float, ...] = (30.0, 50.0, 70.0)) -> Feature:
    return Feature(
        name=f"rsi{length}",
        family="momentum",
        label=f"RSI({length})",
        levels=levels,
        compute=lambda df: core.rsi(df["close"], length),
        pine=f"ta.rsi(close, {length})",
    )


def _stoch(length: int) -> Feature:
    return Feature(
        name=f"stoch{length}",
        family="momentum",
        label=f"Stochastic %K({length})",
        levels=(20.0, 80.0),
        compute=lambda df: core.stoch(df, length),
        pine=f"ta.stoch(close, high, low, {length})",
    )


def _cci(length: int) -> Feature:
    return Feature(
        name=f"cci{length}",
        family="momentum",
        label=f"CCI({length})",
        crosses=True,
        levels=(-100.0, 100.0),
        compute=lambda df: core.cci(df, length),
        pine=f"ta.cci(hlc3, {length})",
        signed=True,
    )


def _roc_atr(length: int) -> Feature:
    return Feature(
        name=f"roc{length}_atr",
        family="momentum",
        label=f"(close - close[{length}]) / ATR14",
        compute=lambda df: (df["close"] - df["close"].shift(length)) / core.atr(df, 14),
        pine=f"(close - close[{length}]) / ta.atr(14)",
        signed=True,
        crosses=length == 20,
    )


def _roc_rank(length: int, window: int) -> Feature:
    return Feature(
        name=f"roc{length}_rank",
        family="momentum",
        label=f"percentrank(ROC({length}), {window})",
        compute=lambda df: core.percentrank(df["close"] - df["close"].shift(length), window),
        pine=f"ta.percentrank(close - close[{length}], {window})",
    )


FEATURES: list[Feature] = [
    _rsi(2, (10.0, 90.0)),
    _rsi(7),
    _rsi(14),
    _stoch(5),
    _stoch(14),
    _cci(20),
    _roc_atr(1),
    _roc_atr(5),
    _roc_atr(20),
    _roc_atr(60),
    _roc_rank(20, 250),
]


def _slow_stoch(length: int, smooth: int) -> Feature:
    return Feature(
        name=f"stoch_slow{length}",
        family="momentum",
        label=f"Slow stochastic %K({length},{smooth})",
        compute=lambda df: core.sma(core.stoch(df, length), smooth),
        pine=f"ta.sma(ta.stoch(close, high, low, {length}), {smooth})",
        levels=(20.0, 50.0, 80.0),
    )


def _stoch_kd(length: int, smooth: int) -> Feature:
    def compute(df):  # type: ignore[no-untyped-def]
        k = core.sma(core.stoch(df, length), smooth)
        return k - core.sma(k, smooth)

    return Feature(
        name=f"stoch_kd{length}",
        family="momentum",
        label=f"Slow stochastic %K - %D ({length},{smooth},{smooth})",
        compute=compute,
        pine=f"_k = ta.sma(ta.stoch(close, high, low, {length}), {smooth})\n_k - ta.sma(_k, {smooth})",
        signed=True,
        crosses=True,
    )


def _stoch_rsi_k(df):  # type: ignore[no-untyped-def]
    r = core.rsi(df["close"], 14)
    lo, hi = core.lowest(r, 14), core.highest(r, 14)
    return core.sma(100 * (r - lo) / (hi - lo).replace(0, np.nan), 3)


_STOCH_RSI_PINE = "_r = ta.rsi(close, 14)\n_k = ta.sma(ta.stoch(_r, _r, _r, 14), 3)"


def _macd_hist_variant(fast: int, slow: int, signal: int) -> Feature:
    def compute(df):  # type: ignore[no-untyped-def]
        line = core.ema(df["close"], fast) - core.ema(df["close"], slow)
        return (line - core.ema(line, signal)) / core.atr(df, 14)

    return Feature(
        name=f"macd_hist_{fast}_{slow}_{signal}",
        family="momentum",
        label=f"MACD({fast},{slow},{signal}) histogram / ATR14",
        compute=compute,
        pine=f"[_macd, _signal, _hist] = ta.macd(close, {fast}, {slow}, {signal})\n_hist / ta.atr(14)",
        signed=True,
        crosses=True,
    )


def _macd_line_variant(fast: int, slow: int, signal: int) -> Feature:
    return Feature(
        name=f"macd_line_{fast}_{slow}",
        family="momentum",
        label=f"MACD({fast},{slow}) line / ATR14",
        compute=lambda df: (core.ema(df["close"], fast) - core.ema(df["close"], slow)) / core.atr(df, 14),
        pine=f"[_macd, _signal, _hist] = ta.macd(close, {fast}, {slow}, {signal})\n_macd / ta.atr(14)",
        signed=True,
        crosses=True,
    )


def _roc_pct(df, n):  # type: ignore[no-untyped-def]
    return 100 * (df["close"] - df["close"].shift(n)) / df["close"].shift(n)


FEATURES += [
    _rsi(21),
    _roc_atr(10),
    _roc_rank(5, 250),
    _roc_rank(60, 250),
    _slow_stoch(14, 3),
    _slow_stoch(5, 3),
    _slow_stoch(21, 5),
    _stoch_kd(14, 3),
    _stoch_kd(5, 3),
    _cci(50),
    _macd_hist_variant(5, 35, 5),
    _macd_hist_variant(8, 17, 9),
    _macd_hist_variant(19, 39, 9),
    _macd_line_variant(5, 35, 5),
    _macd_line_variant(8, 17, 9),
    Feature(
        name="stoch_rsi",
        family="momentum",
        label="Stochastic RSI %K(14,14,3)",
        compute=_stoch_rsi_k,
        pine=_STOCH_RSI_PINE + "\n_k",
        levels=(20.0, 50.0, 80.0),
    ),
    Feature(
        name="stoch_rsi_kd",
        family="momentum",
        label="Stochastic RSI %K - %D",
        compute=lambda df: _stoch_rsi_k(df) - core.sma(_stoch_rsi_k(df), 3),
        pine=_STOCH_RSI_PINE + "\n_k - ta.sma(_k, 3)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="rsi_signal",
        family="momentum",
        label="RSI(14) - SMA(RSI(14), 9)",
        compute=lambda df: core.rsi(df["close"], 14) - core.sma(core.rsi(df["close"], 14), 9),
        pine="_r = ta.rsi(close, 14)\n_r - ta.sma(_r, 9)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="williams_r14",
        family="momentum",
        label="Williams %R(14)",
        compute=lambda df: core.stoch(df, 14) - 100,
        pine="ta.wpr(14)",
        levels=(-80.0, -20.0),
    ),
    Feature(
        name="tsi",
        family="momentum",
        label="True strength index(13, 25)",
        compute=lambda df: core.tsi(df["close"], 13, 25),
        pine="ta.tsi(close, 13, 25)",
        signed=True,
        crosses=True,
        levels=(-0.25, 0.25),
    ),
    Feature(
        name="tsi_signal",
        family="momentum",
        label="TSI - EMA(TSI, 7)",
        compute=lambda df: core.tsi(df["close"], 13, 25) - core.ema(core.tsi(df["close"], 13, 25), 7),
        pine="_t = ta.tsi(close, 13, 25)\n_t - ta.ema(_t, 7)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="cmo14",
        family="momentum",
        label="Chande momentum oscillator(14)",
        compute=lambda df: core.cmo(df["close"], 14),
        pine="ta.cmo(close, 14)",
        signed=True,
        crosses=True,
        levels=(-50.0, 50.0),
    ),
    Feature(
        name="ultimate",
        family="momentum",
        label="Ultimate oscillator(7, 14, 28)",
        compute=lambda df: core.ultimate(df),
        pine=("_hi = math.max(high, close[1])\n_lo = math.min(low, close[1])\n_bp = close - _lo\n_tr = _hi - _lo\n"
              "100 * (4 * math.sum(_bp, 7) / math.sum(_tr, 7) + 2 * math.sum(_bp, 14) / math.sum(_tr, 14)"
              " + math.sum(_bp, 28) / math.sum(_tr, 28)) / 7"),
        levels=(30.0, 50.0, 70.0),
    ),
    Feature(
        name="fisher10",
        family="momentum",
        label="Fisher transform(10) of hl2",
        compute=lambda df: core.fisher(df, 10),
        pine=("_hi = ta.highest(hl2, 10)\n_lo = ta.lowest(hl2, 10)\nvar float _v = 0.0\nvar float _f = 0.0\n"
              "_raw = 0.66 * ((hl2 - _lo) / (_hi - _lo) - 0.5) + 0.67 * _v\n"
              "_raw := _raw > 0.99 ? 0.999 : _raw < -0.99 ? -0.999 : _raw\n"
              "_fish = 0.5 * math.log((1 + _raw) / (1 - _raw)) + 0.5 * _f\n"
              "_v := na(_raw) ? 0.0 : _raw\n_f := na(_fish) ? 0.0 : _fish\n_fish"),
        signed=True,
        crosses=True,
        levels=(-1.5, 1.5),
    ),
    Feature(
        name="coppock",
        family="momentum",
        label="Coppock curve (WMA10 of ROC14 + ROC11)",
        compute=lambda df: core.wma(_roc_pct(df, 14) + _roc_pct(df, 11), 10),
        pine="ta.wma(ta.roc(close, 14) + ta.roc(close, 11), 10)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="awesome",
        family="momentum",
        label="Awesome oscillator / ATR14",
        compute=lambda df: (core.sma((df["high"] + df["low"]) / 2, 5) - core.sma((df["high"] + df["low"]) / 2, 34))
        / core.atr(df, 14),
        pine="(ta.sma(hl2, 5) - ta.sma(hl2, 34)) / ta.atr(14)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="elder_bull",
        family="momentum",
        label="Elder bull power (high - EMA13) / ATR14",
        compute=lambda df: (df["high"] - core.ema(df["close"], 13)) / core.atr(df, 14),
        pine="(high - ta.ema(close, 13)) / ta.atr(14)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="elder_bear",
        family="momentum",
        label="Elder bear power (low - EMA13) / ATR14",
        compute=lambda df: (df["low"] - core.ema(df["close"], 13)) / core.atr(df, 14),
        pine="(low - ta.ema(close, 13)) / ta.atr(14)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="momentum_rank",
        family="momentum",
        label="percentrank(close - close[10], 100)",
        compute=lambda df: core.percentrank(df["close"] - df["close"].shift(10), 100),
        pine="ta.percentrank(close - close[10], 100)",
        levels=(10.0, 90.0),
    ),
]
