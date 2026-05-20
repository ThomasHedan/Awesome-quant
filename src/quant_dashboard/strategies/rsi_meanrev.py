"""Reference strategy #2: RSI mean reversion.

Long when RSI crosses below ``lower`` (oversold); exit when RSI crosses back
above ``upper``. Wilder-smoothed RSI uses an EWMA recurrence, so values at
time t depend only on bars up through t — no look-ahead.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant_dashboard.data import OHLCV
from quant_dashboard.strategies.base import ParamSpec, Signals, Strategy, boolify
from quant_dashboard.strategies.registry import register


def wilder_rsi(close: pd.Series, period: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = -delta.clip(upper=0.0)
    # Wilder smoothing = EWMA with alpha = 1/period; using adjust=False makes
    # the recurrence strictly causal.
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0.0, np.nan)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    # When avg_loss is 0 the asset only went up: RSI is 100.
    rsi = rsi.where(~avg_loss.eq(0.0), 100.0)
    return rsi


@register
class RSIMeanReversion(Strategy):
    name = "rsi_meanrev"
    description = "Long when RSI crosses below `lower`; exit when RSI crosses above `upper`."

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="period",
                kind="int",
                default=14,
                low=2,
                high=60,
                step=1,
                description="RSI lookback period.",
            ),
            ParamSpec(
                name="lower",
                kind="float",
                default=30.0,
                low=5.0,
                high=45.0,
                step=5.0,
                description="Oversold threshold; long entry when RSI crosses below.",
            ),
            ParamSpec(
                name="upper",
                kind="float",
                default=70.0,
                low=55.0,
                high=95.0,
                step=5.0,
                description="Exit threshold; close long when RSI crosses above.",
            ),
        ]

    def _compute_signals(
        self,
        data: OHLCV,
        *,
        period: int,
        lower: float,
        upper: float,
    ) -> Signals:
        if lower >= upper:
            empty = pd.Series(False, index=data.df.index, dtype=bool)
            return Signals(entries=empty, exits=empty)

        rsi = wilder_rsi(data.df["close"], period=period)
        prev = rsi.shift(1)
        entries = (rsi < lower) & (prev >= lower)
        exits = (rsi > upper) & (prev <= upper)
        return Signals(entries=boolify(entries), exits=boolify(exits))
