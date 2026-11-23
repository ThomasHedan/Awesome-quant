"""Trend features."""

from __future__ import annotations

import numpy as np

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
        levels=(20.0, 25.0, 40.0),
        compute=lambda df: core.dmi(df, length)[2],
        pine=f"[_diPlus, _diMinus, _adx] = ta.dmi({length}, {length})\n_adx",
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
        pine=f"[_diPlus, _diMinus, _adx] = ta.dmi({length}, {length})\n_diPlus - _diMinus",
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
        pine=f"[_macd, _signal, _hist] = ta.macd(close, 12, 26, 9)\n_hist / {_ATR}",
        signed=True,
        crosses=True,
    )


def _macd_line() -> Feature:
    return Feature(
        name="macd_line",
        family="trend",
        label="MACD(12,26) line / ATR14",
        compute=lambda df: (core.ema(df["close"], 12) - core.ema(df["close"], 26)) / core.atr(df, 14),
        pine=f"[_macd, _signal, _hist] = ta.macd(close, 12, 26, 9)\n_macd / {_ATR}",
        signed=True,
    )


def _aroon(length: int) -> Feature:
    return Feature(
        name=f"aroon_osc{length}",
        family="trend",
        levels=(-50.0, 50.0),
        crosses=True,
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


def _compose(pine_ma: str, template: str) -> str:
    """Insert a (possibly multi-line) Pine body into ``template`` at ``{ma}``."""
    *setup, value = pine_ma.split("\n")
    return "\n".join([*setup, template.format(ma=value)])


def _ma_dist(kind: str, length: int, compute_ma, pine_ma: str) -> Feature:  # type: ignore[no-untyped-def]
    return Feature(
        name=f"dist_{kind}{length}",
        family="trend",
        label=f"(close - {kind.upper()}{length}) / ATR14",
        compute=lambda df: (df["close"] - compute_ma(df["close"], length)) / core.atr(df, 14),
        pine=_compose(pine_ma, "(close - {ma}) / " + _ATR),
        signed=True,
        crosses=True,
    )


def _ma_spread(kind: str, fast: int, slow: int, compute_ma, pine_fast: str, pine_slow: str) -> Feature:  # type: ignore[no-untyped-def]
    return Feature(
        name=f"{kind}{fast}_{slow}",
        family="trend",
        label=f"({kind.upper()}{fast} - {kind.upper()}{slow}) / ATR14",
        compute=lambda df: (compute_ma(df["close"], fast) - compute_ma(df["close"], slow)) / core.atr(df, 14),
        pine=f"({pine_fast} - {pine_slow}) / {_ATR}",
        signed=True,
        crosses=True,
    )


def _ma_slope(kind: str, length: int, lag: int, compute_ma, pine_ma: str) -> Feature:  # type: ignore[no-untyped-def]
    def compute(df):  # type: ignore[no-untyped-def]
        m = compute_ma(df["close"], length)
        return (m - m.shift(lag)) / core.atr(df, 14)

    return Feature(
        name=f"slope_{kind}{length}",
        family="trend",
        label=f"({kind.upper()}{length} - {kind.upper()}{length}[{lag}]) / ATR14",
        compute=compute,
        pine=_compose(pine_ma, "_m = {ma}") + f"\n(_m - _m[{lag}]) / {_ATR}",
        signed=True,
        crosses=True,
    )


def _sma_pine(n: int) -> str:
    return f"ta.sma(close, {n})"


def _ema_pine(n: int) -> str:
    return f"ta.ema(close, {n})"


def _hma_pine(n: int) -> str:
    return f"ta.hma(close, {n})"


def _ichimoku_cloud_mid(df):  # type: ignore[no-untyped-def]
    tenkan, kijun = core.donchian_mid(df, 9), core.donchian_mid(df, 26)
    span_a = (tenkan + kijun) / 2
    span_b = core.donchian_mid(df, 52)
    return ((span_a + span_b) / 2).shift(25)


def _ichimoku_thickness(df):  # type: ignore[no-untyped-def]
    tenkan, kijun = core.donchian_mid(df, 9), core.donchian_mid(df, 26)
    return ((tenkan + kijun) / 2 - core.donchian_mid(df, 52)).shift(25)


_DONCHIAN_MID_PINE = "(ta.highest(high, {n}) + ta.lowest(low, {n})) / 2"
_TENKAN = _DONCHIAN_MID_PINE.format(n=9)
_KIJUN = _DONCHIAN_MID_PINE.format(n=26)
_SENKOU_B = _DONCHIAN_MID_PINE.format(n=52)

FEATURES += [
    _ma_dist("sma", 20, core.sma, _sma_pine(20)),
    _ma_dist("sma", 50, core.sma, _sma_pine(50)),
    _ma_dist("sma", 100, core.sma, _sma_pine(100)),
    _ma_dist("sma", 200, core.sma, _sma_pine(200)),
    _ma_dist("ema", 9, core.ema, _ema_pine(9)),
    _ma_dist("ema", 100, core.ema, _ema_pine(100)),
    _ma_dist("wma", 20, core.wma, "ta.wma(close, 20)"),
    _ma_dist("hma", 9, core.hma, _hma_pine(9)),
    _ma_dist("hma", 21, core.hma, _hma_pine(21)),
    _ma_dist("hma", 55, core.hma, _hma_pine(55)),
    _ma_dist("hma", 100, core.hma, _hma_pine(100)),
    _ma_dist("alma", 9, core.alma, "ta.alma(close, 9, 0.85, 6)"),
    _ma_dist("alma", 50, core.alma, "ta.alma(close, 50, 0.85, 6)"),
    _ma_dist("dema", 20, core.dema, "_e1 = ta.ema(close, 20)\n_dema = 2 * _e1 - ta.ema(_e1, 20)\n_dema"),
    _ma_dist("tema", 20, core.tema,
             "_e1 = ta.ema(close, 20)\n_e2 = ta.ema(_e1, 20)\n_tema = 3 * _e1 - 3 * _e2 + ta.ema(_e2, 20)\n_tema"),
    _ma_dist("kama", 10, core.kama,
             "_er = math.abs(close - close[10]) / math.sum(math.abs(ta.change(close)), 10)\n"
             "_sc = math.pow(_er * (2.0 / 3.0 - 2.0 / 31.0) + 2.0 / 31.0, 2)\n"
             "var float _k = na\n_k := na(_k) ? close : _k + nz(_sc) * (close - _k)\n_k"),
    _ma_spread("ema", 9, 21, core.ema, _ema_pine(9), _ema_pine(21)),
    _ma_spread("ema", 12, 26, core.ema, _ema_pine(12), _ema_pine(26)),
    _ma_spread("ema", 21, 55, core.ema, _ema_pine(21), _ema_pine(55)),
    _ma_spread("sma", 10, 30, core.sma, _sma_pine(10), _sma_pine(30)),
    _ma_spread("sma", 20, 50, core.sma, _sma_pine(20), _sma_pine(50)),
    _ma_spread("sma", 50, 200, core.sma, _sma_pine(50), _sma_pine(200)),
    _ma_spread("hma", 9, 21, core.hma, _hma_pine(9), _hma_pine(21)),
    _ma_spread("hma", 21, 55, core.hma, _hma_pine(21), _hma_pine(55)),
    _ma_slope("ema", 20, 3, core.ema, _ema_pine(20)),
    _ma_slope("sma", 50, 5, core.sma, _sma_pine(50)),
    _ma_slope("hma", 9, 1, core.hma, _hma_pine(9)),
    _ma_slope("hma", 21, 1, core.hma, _hma_pine(21)),
    _ma_slope("hma", 55, 1, core.hma, _hma_pine(55)),
    _linreg(50),
    _linreg(100),
    _efficiency(10),
    _efficiency(50),
    _adx(7),
    _di_diff(7),
    _aroon(14),
    Feature(
        name="adx14_slope",
        family="trend",
        label="ADX(14) - ADX(14)[5]",
        compute=lambda df: core.dmi(df, 14)[2].diff(5),
        pine="[_diPlus, _diMinus, _adx] = ta.dmi(14, 14)\n_adx - _adx[5]",
        signed=True,
    ),
    Feature(
        name="dist_supertrend10_3",
        family="trend",
        label="(close - SuperTrend(10, 3)) / ATR14",
        compute=lambda df: (df["close"] - core.supertrend(df, 3.0, 10)) / core.atr(df, 14),
        pine=f"[_st, _dir] = ta.supertrend(3, 10)\n(close - _st) / {_ATR}",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="dist_supertrend20_2",
        family="trend",
        label="(close - SuperTrend(20, 2)) / ATR14",
        compute=lambda df: (df["close"] - core.supertrend(df, 2.0, 20)) / core.atr(df, 14),
        pine=f"[_st, _dir] = ta.supertrend(2, 20)\n(close - _st) / {_ATR}",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="dist_psar",
        family="trend",
        label="(close - Parabolic SAR(0.02, 0.02, 0.2)) / ATR14",
        compute=lambda df: (df["close"] - core.sar(df)) / core.atr(df, 14),
        pine=f"(close - ta.sar(0.02, 0.02, 0.2)) / {_ATR}",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="ichimoku_tk",
        family="trend",
        label="(Tenkan - Kijun) / ATR14",
        compute=lambda df: (core.donchian_mid(df, 9) - core.donchian_mid(df, 26)) / core.atr(df, 14),
        pine=f"({_TENKAN} - {_KIJUN}) / {_ATR}",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="dist_kijun",
        family="trend",
        label="(close - Kijun) / ATR14",
        compute=lambda df: (df["close"] - core.donchian_mid(df, 26)) / core.atr(df, 14),
        pine=f"(close - {_KIJUN}) / {_ATR}",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="dist_cloud",
        family="trend",
        label="(close - Ichimoku cloud middle) / ATR14",
        compute=lambda df: (df["close"] - _ichimoku_cloud_mid(df)) / core.atr(df, 14),
        pine=f"_a = ({_TENKAN} + {_KIJUN}) / 2\n_b = {_SENKOU_B}\n(close - ((_a + _b) / 2)[25]) / {_ATR}",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="cloud_thickness",
        family="trend",
        label="(Senkou A - Senkou B) / ATR14 (bullish cloud > 0)",
        compute=lambda df: _ichimoku_thickness(df) / core.atr(df, 14),
        pine=f"_a = ({_TENKAN} + {_KIJUN}) / 2\n_b = {_SENKOU_B}\n(_a - _b)[25] / {_ATR}",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="vortex14",
        family="trend",
        label="Vortex VI+ - VI- (14)",
        compute=lambda df: core.vortex(df, 14),
        pine=("(math.sum(math.abs(high - low[1]), 14) - math.sum(math.abs(low - high[1]), 14)) / "
              "math.sum(ta.atr(1), 14)"),
        signed=True,
        crosses=True,
    ),
    Feature(
        name="trix15",
        family="trend",
        label="TRIX(15) of log close (x10000)",
        compute=lambda df: core.trix(df["close"], 15),
        pine="_t = ta.ema(ta.ema(ta.ema(math.log(close), 15), 15), 15)\n10000 * ta.change(_t)",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="dpo20",
        family="trend",
        label="Detrended price oscillator(20) / ATR14 (non-centred)",
        compute=lambda df: (df["close"] - core.sma(df["close"], 20).shift(11)) / core.atr(df, 14),
        pine=f"(close - ta.sma(close, 20)[11]) / {_ATR}",
        signed=True,
        crosses=True,
    ),
    Feature(
        name="ema_stack",
        family="trend",
        label="EMA alignment score: sign(EMA9-EMA21) + sign(EMA21-EMA50) + sign(EMA50-EMA200)",
        compute=lambda df: (np.sign(core.ema(df["close"], 9) - core.ema(df["close"], 21))
                            + np.sign(core.ema(df["close"], 21) - core.ema(df["close"], 50))
                            + np.sign(core.ema(df["close"], 50) - core.ema(df["close"], 200))),
        pine=("math.sign(ta.ema(close, 9) - ta.ema(close, 21)) + math.sign(ta.ema(close, 21) - ta.ema(close, 50))"
              " + math.sign(ta.ema(close, 50) - ta.ema(close, 200))"),
        signed=True,
        levels=(-2.5, 2.5),
    ),
]
