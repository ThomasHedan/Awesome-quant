"""Candle and swing patterns as 0/1 flags (binary features)."""

from __future__ import annotations

import pandas as pd

from fxgen.indicators.base import Feature


def _flag(name: str, label: str, compute, pine: str) -> Feature:  # type: ignore[no-untyped-def]
    return Feature(
        name=name,
        family="structure",
        label=label,
        compute=lambda df: compute(df).astype(float).where(df["close"].shift(2).notna()),
        pine=f"({pine}) ? 1.0 : 0.0",
        binary=True,
    )


def _rng(df: pd.DataFrame) -> pd.Series:
    return df["high"] - df["low"]


def _body(df: pd.DataFrame) -> pd.Series:
    return (df["close"] - df["open"]).abs()


FEATURES: list[Feature] = [
    _flag("inside_bar", "inside bar", lambda d: (d["high"] < d["high"].shift(1)) & (d["low"] > d["low"].shift(1)),
          "high < high[1] and low > low[1]"),
    _flag("outside_bar", "outside bar", lambda d: (d["high"] > d["high"].shift(1)) & (d["low"] < d["low"].shift(1)),
          "high > high[1] and low < low[1]"),
    _flag("higher_high", "higher high and higher low",
          lambda d: (d["high"] > d["high"].shift(1)) & (d["low"] > d["low"].shift(1)),
          "high > high[1] and low > low[1]"),
    _flag("lower_low", "lower high and lower low",
          lambda d: (d["high"] < d["high"].shift(1)) & (d["low"] < d["low"].shift(1)),
          "high < high[1] and low < low[1]"),
    _flag("bull_engulfing", "bullish engulfing",
          lambda d: (d["close"] > d["open"]) & (d["close"].shift(1) < d["open"].shift(1))
          & (d["close"] >= d["open"].shift(1)) & (d["open"] <= d["close"].shift(1)),
          "close > open and close[1] < open[1] and close >= open[1] and open <= close[1]"),
    _flag("bear_engulfing", "bearish engulfing",
          lambda d: (d["close"] < d["open"]) & (d["close"].shift(1) > d["open"].shift(1))
          & (d["close"] <= d["open"].shift(1)) & (d["open"] >= d["close"].shift(1)),
          "close < open and close[1] > open[1] and close <= open[1] and open >= close[1]"),
    _flag("bull_pinbar", "bullish pin bar (lower wick > 60% of range)",
          lambda d: (d[["open", "close"]].min(axis=1) - d["low"]) > 0.6 * _rng(d),
          "math.min(open, close) - low > 0.6 * (high - low)"),
    _flag("bear_pinbar", "bearish pin bar (upper wick > 60% of range)",
          lambda d: (d["high"] - d[["open", "close"]].max(axis=1)) > 0.6 * _rng(d),
          "high - math.max(open, close) > 0.6 * (high - low)"),
    _flag("doji", "doji (body < 10% of range)", lambda d: _body(d) < 0.1 * _rng(d),
          "math.abs(close - open) < 0.1 * (high - low)"),
    _flag("marubozu_up", "bullish marubozu (body > 85% of range)",
          lambda d: (d["close"] > d["open"]) & (_body(d) > 0.85 * _rng(d)),
          "close > open and math.abs(close - open) > 0.85 * (high - low)"),
    _flag("marubozu_down", "bearish marubozu (body > 85% of range)",
          lambda d: (d["close"] < d["open"]) & (_body(d) > 0.85 * _rng(d)),
          "close < open and math.abs(close - open) > 0.85 * (high - low)"),
    _flag("three_up", "three higher closes", lambda d: (d["close"].diff() > 0) & (d["close"].diff().shift(1) > 0)
          & (d["close"].diff().shift(2) > 0), "close > close[1] and close[1] > close[2] and close[2] > close[3]"),
    _flag("three_down", "three lower closes", lambda d: (d["close"].diff() < 0) & (d["close"].diff().shift(1) < 0)
          & (d["close"].diff().shift(2) < 0), "close < close[1] and close[1] < close[2] and close[2] < close[3]"),
    _flag("nr7", "narrowest range of the last 7 bars", lambda d: _rng(d) <= _rng(d).rolling(7, min_periods=7).min(),
          "(high - low) <= ta.lowest(high - low, 7)"),
]
