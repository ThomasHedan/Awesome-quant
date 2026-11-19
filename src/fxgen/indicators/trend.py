"""Trend features."""

from __future__ import annotations

from fxgen.indicators import core
from fxgen.indicators.base import Feature

_ATR = "ta.atr(14)"


def _dist_ema(length: int) -> Feature:
    return Feature(
        name=f"dist_ema{length}",
        family="trend",
        label=f"(close - EMA{length}) / ATR14",
        compute=lambda df: (df["close"] - core.ema(df["close"], length)) / core.atr(df, 14),
        pine=f"(close - ta.ema(close, {length})) / {_ATR}",
        signed=True,
        crosses=True,
    )


def _ema_spread(fast: int, slow: int) -> Feature:
    return Feature(
        name=f"ema{fast}_{slow}",
        family="trend",
        label=f"(EMA{fast} - EMA{slow}) / ATR14",
        compute=lambda df: (core.ema(df["close"], fast) - core.ema(df["close"], slow)) / core.atr(df, 14),
        pine=f"(ta.ema(close, {fast}) - ta.ema(close, {slow})) / {_ATR}",
        signed=True,
        crosses=True,
    )


def _ema_slope(length: int, lag: int) -> Feature:
    def compute(df):  # type: ignore[no-untyped-def]
        e = core.ema(df["close"], length)
        return (e - e.shift(lag)) / core.atr(df, 14)

    return Feature(
        name=f"slope_ema{length}",
        family="trend",
        label=f"(EMA{length} - EMA{length}[{lag}]) / ATR14",
        compute=compute,
        pine=f"(ta.ema(close, {length}) - ta.ema(close, {length})[{lag}]) / {_ATR}",
        signed=True,
    )


def _linreg(length: int) -> Feature:
    return Feature(
        name=f"linreg_slope{length}",
        family="trend",
        label=f"LinReg slope({length}) / ATR14",
        compute=lambda df: core.linreg_slope(df["close"], length) / core.atr(df, 14),
        pine=f"(ta.linreg(close, {length}, 0) - ta.linreg(close, {length}, 1)) / {_ATR}",
        signed=True,
    )


def _adx(length: int) -> Feature:
    return Feature(
        name=f"adx{length}",
        family="trend",
        label=f"ADX({length})",
        compute=lambda df: core.dmi(df, length)[2],
        pine=f"ta.dmi({length}, {length})[2]",
    )


def _di_diff(length: int) -> Feature:
    def compute(df):  # type: ignore[no-untyped-def]
        plus, minus, _ = core.dmi(df, length)
        return plus - minus

    return Feature(
        name=f"di_diff{length}",
        family="trend",
        label=f"+DI({length}) - -DI({length})",
        compute=compute,
        pine=f"ta.dmi({length}, {length})[0] - ta.dmi({length}, {length})[1]",
        signed=True,
        crosses=True,
    )


def _macd_hist() -> Feature:
    def compute(df):  # type: ignore[no-untyped-def]
        line = core.ema(df["close"], 12) - core.ema(df["close"], 26)
        return (line - core.ema(line, 9)) / core.atr(df, 14)

    return Feature(
        name="macd_hist",
        family="trend",
        label="MACD(12,26,9) histogram / ATR14",
        compute=compute,
        pine=f"ta.macd(close, 12, 26, 9)[2] / {_ATR}",
        signed=True,
        crosses=True,
    )


def _macd_line() -> Feature:
    return Feature(
        name="macd_line",
        family="trend",
        label="MACD(12,26) line / ATR14",
        compute=lambda df: (core.ema(df["close"], 12) - core.ema(df["close"], 26)) / core.atr(df, 14),
        pine=f"ta.macd(close, 12, 26, 9)[0] / {_ATR}",
        signed=True,
    )


def _aroon(length: int) -> Feature:
    return Feature(
        name=f"aroon_osc{length}",
        family="trend",
        label=f"Aroon oscillator({length})",
        compute=lambda df: core.aroon_oscillator(df, length),
        pine=(
            f"(100.0 * ({length} + ta.highestbars(high, {length + 1})) / {length}) - "
            f"(100.0 * ({length} + ta.lowestbars(low, {length + 1})) / {length})"
        ),
        signed=True,
    )


def _efficiency(length: int) -> Feature:
    return Feature(
        name=f"efficiency{length}",
        family="trend",
        label=f"Kaufman efficiency ratio({length})",
        compute=lambda df: core.efficiency_ratio(df["close"], length),
        pine=f"math.abs(close - close[{length}]) / math.sum(math.abs(ta.change(close)), {length})",
    )


FEATURES: list[Feature] = [
    _dist_ema(20),
    _dist_ema(50),
    _dist_ema(200),
    _ema_spread(20, 50),
    _ema_spread(50, 200),
    _ema_slope(50, 5),
    _ema_slope(200, 10),
    _linreg(20),
    _adx(14),
    _di_diff(14),
    _macd_hist(),
    _macd_line(),
    _aroon(25),
    _efficiency(20),
]
