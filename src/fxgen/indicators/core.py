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


def _windows(x: pd.Series, length: int) -> tuple[np.ndarray, np.ndarray]:
    values = x.to_numpy(np.float64)
    out = np.full(values.shape[0], np.nan)
    if values.shape[0] >= length:
        return np.lib.stride_tricks.sliding_window_view(values, length), out
    return np.empty((0, length)), out


def _weighted(x: pd.Series, weights: np.ndarray) -> pd.Series:
    """Weighted moving average; ``weights`` ordered oldest to newest, NaN-propagating."""
    windows, out = _windows(x, len(weights))
    if windows.shape[0]:
        out[len(weights) - 1 :] = windows @ (weights / weights.sum())
    return pd.Series(out, index=x.index)


def wma(x: pd.Series, length: int) -> pd.Series:
    """Pine ``ta.wma``: linear weights, the newest bar weighs ``length``."""
    return _weighted(x, np.arange(1, length + 1, dtype=np.float64))


def hma(x: pd.Series, length: int) -> pd.Series:
    """Pine ``ta.hma``: wma(2 * wma(x, n/2) - wma(x, n), sqrt(n))."""
    return wma(2 * wma(x, length // 2) - wma(x, length), int(np.floor(np.sqrt(length))))


def alma(x: pd.Series, length: int, offset: float = 0.85, sigma: float = 6.0) -> pd.Series:
    """Pine ``ta.alma``."""
    m = offset * (length - 1)
    s = length / sigma
    i = np.arange(length, dtype=np.float64)
    return _weighted(x, np.exp(-((i - m) ** 2) / (2 * s * s)))


def dema(x: pd.Series, length: int) -> pd.Series:
    e1 = ema(x, length)
    return 2 * e1 - ema(e1, length)


def tema(x: pd.Series, length: int) -> pd.Series:
    e1 = ema(x, length)
    e2 = ema(e1, length)
    return 3 * e1 - 3 * e2 + ema(e2, length)


@numba.njit(cache=True)
def _kama(close: np.ndarray, sc: np.ndarray) -> np.ndarray:
    out = np.empty(close.shape[0])
    k = close[0]
    for i in range(close.shape[0]):
        s = 0.0 if np.isnan(sc[i]) else sc[i]
        k = k + s * (close[i] - k)
        out[i] = k
    return out


def kama(x: pd.Series, length: int = 10, fast: int = 2, slow: int = 30) -> pd.Series:
    """Kaufman adaptive moving average, seeded with the first close."""
    fast_sc, slow_sc = 2.0 / (fast + 1), 2.0 / (slow + 1)
    sc = (efficiency_ratio(x, length) * (fast_sc - slow_sc) + slow_sc) ** 2
    return pd.Series(_kama(x.to_numpy(np.float64), sc.to_numpy(np.float64)), index=x.index)


@numba.njit(cache=True)
def _supertrend(hl2: np.ndarray, close: np.ndarray, atr_: np.ndarray, factor: float) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan)
    prev_lower = 0.0
    prev_upper = 0.0
    prev_st = np.nan
    for i in range(n):
        upper = hl2[i] + factor * atr_[i]
        lower = hl2[i] - factor * atr_[i]
        if i > 0:
            if not (lower > prev_lower or close[i - 1] < prev_lower):
                lower = prev_lower
            if not (upper < prev_upper or close[i - 1] > prev_upper):
                upper = prev_upper
        if i == 0 or np.isnan(atr_[i - 1]):
            direction = 1
        elif prev_st == prev_upper:
            direction = -1 if close[i] > upper else 1
        else:
            direction = 1 if close[i] < lower else -1
        st = lower if direction == -1 else upper
        out[i] = st
        prev_lower = 0.0 if np.isnan(lower) else lower
        prev_upper = 0.0 if np.isnan(upper) else upper
        prev_st = st
    return out


def supertrend(df: pd.DataFrame, factor: float = 3.0, period: int = 10) -> pd.Series:
    """Pine ``ta.supertrend(factor, period)`` line (below price in an uptrend)."""
    hl2 = ((df["high"] + df["low"]) / 2).to_numpy(np.float64)
    return pd.Series(_supertrend(hl2, df["close"].to_numpy(np.float64), atr(df, period).to_numpy(np.float64), factor),
                     index=df.index)


@numba.njit(cache=True)
def _sar(high: np.ndarray, low: np.ndarray, close: np.ndarray, start: float, inc: float, maximum: float) -> np.ndarray:
    n = close.shape[0]
    out = np.full(n, np.nan)
    result = np.nan
    max_min = np.nan
    acc = np.nan
    below = False
    for i in range(1, n):
        first = False
        if i == 1:
            if close[1] > close[0]:
                below = True
                max_min = high[1]
                result = low[0]
            else:
                below = False
                max_min = low[1]
                result = high[0]
            first = True
            acc = start
        result = result + acc * (max_min - result)
        if below:
            if result > low[i]:
                first = True
                below = False
                result = max(high[i], max_min)
                max_min = low[i]
                acc = start
        else:
            if result < high[i]:
                first = True
                below = True
                result = min(low[i], max_min)
                max_min = high[i]
                acc = start
        if not first:
            if below:
                if high[i] > max_min:
                    max_min = high[i]
                    acc = min(acc + inc, maximum)
            else:
                if low[i] < max_min:
                    max_min = low[i]
                    acc = min(acc + inc, maximum)
        if below:
            result = min(result, low[i - 1])
            if i > 1:
                result = min(result, low[i - 2])
        else:
            result = max(result, high[i - 1])
            if i > 1:
                result = max(result, high[i - 2])
        out[i] = result
    return out


def sar(df: pd.DataFrame, start: float = 0.02, inc: float = 0.02, maximum: float = 0.2) -> pd.Series:
    """Pine ``ta.sar`` (reference implementation from the Pine manual)."""
    return pd.Series(
        _sar(df["high"].to_numpy(np.float64), df["low"].to_numpy(np.float64), df["close"].to_numpy(np.float64),
             start, inc, maximum),
        index=df.index,
    )


@numba.njit(cache=True)
def _fisher(raw: np.ndarray) -> np.ndarray:
    n = raw.shape[0]
    out = np.full(n, np.nan)
    value_prev = 0.0
    fish_prev = 0.0
    for i in range(n):
        v = 0.66 * (raw[i] - 0.5) + 0.67 * value_prev
        if np.isnan(v):
            value_prev = 0.0
            fish_prev = 0.0
            continue
        v = 0.999 if v > 0.99 else (-0.999 if v < -0.99 else v)
        f = 0.5 * np.log((1 + v) / (1 - v)) + 0.5 * fish_prev
        out[i] = f
        value_prev = v
        fish_prev = f
    return out


def fisher(df: pd.DataFrame, length: int = 10) -> pd.Series:
    """Ehlers Fisher transform of hl2 (TradingView built-in script)."""
    hl2 = (df["high"] + df["low"]) / 2
    hi, lo = highest(hl2, length), lowest(hl2, length)
    raw = (hl2 - lo) / (hi - lo).replace(0, np.nan)
    return pd.Series(_fisher(raw.to_numpy(np.float64)), index=df.index)


def tsi(x: pd.Series, short: int = 13, long: int = 25) -> pd.Series:
    """Pine ``ta.tsi(x, short, long)`` in [-1, 1]."""
    change = x.diff()
    num = ema(ema(change, long), short)
    den = ema(ema(change.abs(), long), short)
    return num / den.replace(0, np.nan)


def cmo(x: pd.Series, length: int) -> pd.Series:
    """Pine ``ta.cmo``."""
    mom = x.diff()
    up = mom.clip(lower=0).rolling(length, min_periods=length).sum()
    down = (-mom).clip(lower=0).rolling(length, min_periods=length).sum()
    return 100 * (up - down) / (up + down).replace(0, np.nan)


def ultimate(df: pd.DataFrame, a: int = 7, b: int = 14, c: int = 28) -> pd.Series:
    prev_close = df["close"].shift(1)
    hi = np.maximum(df["high"], prev_close)
    lo = np.minimum(df["low"], prev_close)
    bp, tr = df["close"] - lo, hi - lo

    def avg(n: int) -> pd.Series:
        return bp.rolling(n, min_periods=n).sum() / tr.rolling(n, min_periods=n).sum().replace(0, np.nan)

    return 100 * (4 * avg(a) + 2 * avg(b) + avg(c)) / 7


def vortex(df: pd.DataFrame, length: int = 14) -> pd.Series:
    """VI+ - VI-."""
    vmp = (df["high"] - df["low"].shift(1)).abs().rolling(length, min_periods=length).sum()
    vmm = (df["low"] - df["high"].shift(1)).abs().rolling(length, min_periods=length).sum()
    tr = atr(df, 1).rolling(length, min_periods=length).sum()
    return (vmp - vmm) / tr.replace(0, np.nan)


def trix(x: pd.Series, length: int = 15) -> pd.Series:
    """10000 x one-bar change of the triple EMA of log(close)."""
    triple = ema(ema(ema(np.log(x), length), length), length)
    return 10000 * triple.diff()


def donchian_mid(df: pd.DataFrame, length: int) -> pd.Series:
    return (highest(df["high"], length) + lowest(df["low"], length)) / 2


@numba.njit(cache=True)
def _streak(close: np.ndarray) -> np.ndarray:
    out = np.zeros(close.shape[0])
    s = 0.0
    for i in range(1, close.shape[0]):
        if close[i] > close[i - 1]:
            s = s + 1 if s > 0 else 1.0
        elif close[i] < close[i - 1]:
            s = s - 1 if s < 0 else -1.0
        else:
            s = 0.0
        out[i] = s
    return out


def streak(x: pd.Series) -> pd.Series:
    """Consecutive up (+) or down (-) closes."""
    return pd.Series(_streak(x.to_numpy(np.float64)), index=x.index)
