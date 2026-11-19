"""Causal indicator primitives, matching TradingView Pine ``ta.*`` semantics.

Every function only uses values at or before ``t`` to produce the value at
``t``. Seeding follows Pine: ``ema`` starts at the first value, ``rma`` starts
with the simple average of the first ``length`` values, ``stdev`` is the
population standard deviation, ``percentrank`` compares to the previous
``length`` values.
"""

from __future__ import annotations

import numba
import numpy as np
import pandas as pd


def sma(x: pd.Series, length: int) -> pd.Series:
    return x.rolling(length, min_periods=length).mean()


def ema(x: pd.Series, length: int) -> pd.Series:
    return x.ewm(span=length, adjust=False, min_periods=1).mean()


@numba.njit(cache=True)
def _rma(values: np.ndarray, length: int) -> np.ndarray:
    out = np.full(values.shape[0], np.nan)
    alpha = 1.0 / length
    acc = 0.0
    count = 0
    state = np.nan
    for i in range(values.shape[0]):
        v = values[i]
        if np.isnan(v):
            out[i] = state
            continue
        if count < length:
            acc += v
            count += 1
            if count == length:
                state = acc / length
                out[i] = state
        else:
            state = alpha * v + (1.0 - alpha) * state
            out[i] = state
    return out


def rma(x: pd.Series, length: int) -> pd.Series:
    return pd.Series(_rma(x.to_numpy(np.float64), length), index=x.index)


def stdev(x: pd.Series, length: int) -> pd.Series:
    return x.rolling(length, min_periods=length).std(ddof=0)


def highest(x: pd.Series, length: int) -> pd.Series:
    return x.rolling(length, min_periods=length).max()


def lowest(x: pd.Series, length: int) -> pd.Series:
    return x.rolling(length, min_periods=length).min()


def true_range(df: pd.DataFrame) -> pd.Series:
    prev_close = df["close"].shift(1)
    ranges = pd.concat(
        [df["high"] - df["low"], (df["high"] - prev_close).abs(), (df["low"] - prev_close).abs()], axis=1
    )
    tr = ranges.max(axis=1)
    tr.iloc[0] = df["high"].iloc[0] - df["low"].iloc[0]
    return tr


def atr(df: pd.DataFrame, length: int = 14) -> pd.Series:
    return rma(true_range(df), length)


def rsi(x: pd.Series, length: int) -> pd.Series:
    change = x.diff()
    up = rma(change.clip(lower=0), length)
    down = rma((-change).clip(lower=0), length)
    rs = up / down.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(down != 0, 100.0).where(up.notna())


def stoch(df: pd.DataFrame, length: int) -> pd.Series:
    lo, hi = lowest(df["low"], length), highest(df["high"], length)
    return 100 * (df["close"] - lo) / (hi - lo).replace(0, np.nan)


def cci(df: pd.DataFrame, length: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    mean = sma(tp, length)
    mad = tp.rolling(length, min_periods=length).apply(lambda w: np.abs(w - w.mean()).mean(), raw=True)
    return (tp - mean) / (0.015 * mad.replace(0, np.nan))


def dmi(df: pd.DataFrame, length: int) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (+DI, -DI, ADX) as in Pine ``ta.dmi(length, length)``."""
    up = df["high"].diff()
    down = -df["low"].diff()
    plus_dm = pd.Series(np.where((up > down) & (up > 0), up, 0.0), index=df.index)
    minus_dm = pd.Series(np.where((down > up) & (down > 0), down, 0.0), index=df.index)
    tr = rma(true_range(df), length)
    plus = 100 * rma(plus_dm, length) / tr
    minus = 100 * rma(minus_dm, length) / tr
    dx = 100 * (plus - minus).abs() / (plus + minus).replace(0, np.nan)
    return plus, minus, rma(dx.fillna(0.0), length)


@numba.njit(cache=True)
def _percentrank(values: np.ndarray, length: int) -> np.ndarray:
    n = values.shape[0]
    out = np.full(n, np.nan)
    for i in range(length, n):
        current = values[i]
        if np.isnan(current):
            continue
        count = 0
        valid = True
        for j in range(1, length + 1):
            prev = values[i - j]
            if np.isnan(prev):
                valid = False
                break
            if prev <= current:
                count += 1
        if valid:
            out[i] = 100.0 * count / length
    return out


def percentrank(x: pd.Series, length: int) -> pd.Series:
    return pd.Series(_percentrank(x.to_numpy(np.float64), length), index=x.index)


def zscore(x: pd.Series, length: int) -> pd.Series:
    return (x - sma(x, length)) / stdev(x, length).replace(0, np.nan)


def linreg_slope(x: pd.Series, length: int) -> pd.Series:
    """Equals Pine ``ta.linreg(x, length, 0) - ta.linreg(x, length, 1)``."""
    t = np.arange(length, dtype=np.float64)
    t_centered = t - t.mean()
    denom = (t_centered**2).sum()
    return x.rolling(length, min_periods=length).apply(lambda w: (t_centered * w).sum() / denom, raw=True)


@numba.njit(cache=True)
def _bars_since_extreme(values: np.ndarray, length: int, find_max: bool) -> np.ndarray:
    n = values.shape[0]
    out = np.full(n, np.nan)
    for i in range(length - 1, n):
        best = values[i]
        offset = 0
        for j in range(1, length):
            v = values[i - j]
            if (find_max and v > best) or (not find_max and v < best):
                best = v
                offset = j
        out[i] = offset
    return out


def bars_since_highest(x: pd.Series, length: int) -> pd.Series:
    """Equals Pine ``-ta.highestbars(x, length)``."""
    return pd.Series(_bars_since_extreme(x.to_numpy(np.float64), length, True), index=x.index)


def bars_since_lowest(x: pd.Series, length: int) -> pd.Series:
    return pd.Series(_bars_since_extreme(x.to_numpy(np.float64), length, False), index=x.index)


def efficiency_ratio(x: pd.Series, length: int) -> pd.Series:
    net = (x - x.shift(length)).abs()
    path = x.diff().abs().rolling(length, min_periods=length).sum()
    return net / path.replace(0, np.nan)


def choppiness(df: pd.DataFrame, length: int) -> pd.Series:
    tr_sum = atr(df, 1).rolling(length, min_periods=length).sum()
    rng = highest(df["high"], length) - lowest(df["low"], length)
    return 100 * np.log10(tr_sum / rng.replace(0, np.nan)) / np.log10(length)


def aroon_oscillator(df: pd.DataFrame, length: int) -> pd.Series:
    up = 100 * (length - bars_since_highest(df["high"], length + 1)) / length
    down = 100 * (length - bars_since_lowest(df["low"], length + 1)) / length
    return up - down


def mfi(df: pd.DataFrame, length: int) -> pd.Series:
    tp = (df["high"] + df["low"] + df["close"]) / 3
    flow = tp * df["volume"]
    change = tp.diff()
    pos = flow.where(change > 0, 0.0).rolling(length, min_periods=length).sum()
    neg = flow.where(change < 0, 0.0).rolling(length, min_periods=length).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def obv(df: pd.DataFrame) -> pd.Series:
    direction = np.sign(df["close"].diff().fillna(0.0))
    return (direction * df["volume"]).cumsum()
