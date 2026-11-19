"""Momentum features."""

from __future__ import annotations

from fxgen.indicators import core
from fxgen.indicators.base import Feature


def _rsi(length: int) -> Feature:
    return Feature(
        name=f"rsi{length}",
        family="momentum",
        label=f"RSI({length})",
        compute=lambda df: core.rsi(df["close"], length),
        pine=f"ta.rsi(close, {length})",
    )


def _stoch(length: int) -> Feature:
    return Feature(
        name=f"stoch{length}",
        family="momentum",
        label=f"Stochastic %K({length})",
        compute=lambda df: core.stoch(df, length),
        pine=f"ta.stoch(close, high, low, {length})",
    )


def _cci(length: int) -> Feature:
    return Feature(
        name=f"cci{length}",
        family="momentum",
        label=f"CCI({length})",
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
    _rsi(2),
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
