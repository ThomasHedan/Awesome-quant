"""Reference strategy #1: SMA crossover.

Long on golden cross (fast SMA crosses above slow), exit on death cross. Both
SMAs use only close values up to and including the decision bar, so the
no-look-ahead test in ``tests/test_strategies.py`` should pass without any
shifting inside the strategy.
"""

from __future__ import annotations

from typing import Any

import pandas as pd

from quant_dashboard.data import OHLCV
from quant_dashboard.strategies.base import ParamSpec, Signals, Strategy, boolify
from quant_dashboard.strategies.registry import register


@register
class SMACross(Strategy):
    name = "sma_cross"
    description = "Long when fast SMA crosses above slow SMA; exit on the inverse cross."

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="fast",
                kind="int",
                default=10,
                low=2,
                high=100,
                step=1,
                description="Fast SMA window (bars).",
            ),
            ParamSpec(
                name="slow",
                kind="int",
                default=30,
                low=5,
                high=300,
                step=1,
                description="Slow SMA window (bars).",
            ),
        ]

    def _compute_signals(self, data: OHLCV, *, fast: int, slow: int) -> Signals:
        if fast >= slow:
            # Degenerate parameterizations should produce no trades rather than
            # silently bleed cash through churn. The optimizer can sweep this
            # region and we want it to look uniformly flat.
            empty = pd.Series(False, index=data.df.index, dtype=bool)
            return Signals(entries=empty, exits=empty)

        close = data.df["close"]
        ma_fast = close.rolling(fast, min_periods=fast).mean()
        ma_slow = close.rolling(slow, min_periods=slow).mean()

        # "Cross" = sign of (fast - slow) changes from <=0 to >0 at this bar.
        spread = ma_fast - ma_slow
        prev = spread.shift(1)
        cross_up = (spread > 0) & (prev <= 0)
        cross_dn = (spread < 0) & (prev >= 0)

        return Signals(entries=boolify(cross_up), exits=boolify(cross_dn))
