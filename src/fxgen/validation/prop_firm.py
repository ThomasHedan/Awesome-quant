"""Prop-firm constraints: daily loss limit and simultaneous exposure across pairs.

Realised daily losses attribute each trade to its exit day, so floating losses
of positions still open are not included; ``max_concurrent`` gives the
worst-case exposure (every open position stopped out at -1R on the same day).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import pandas as pd

from fxgen.backtest.metrics import daily_r, max_concurrent_positions
from fxgen.config import PropFirmConfig
from fxgen.data.splits import Period


@dataclass(frozen=True)
class PropFirmResult:
    worst_day_r: float
    max_concurrent: int
    max_risk_pct_daily: float
    max_risk_pct_all_stopped: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def prop_firm_limits(trades: pd.DataFrame, period: Period, prop: PropFirmConfig) -> PropFirmResult:
    """Risk per trade that keeps the worst realised day under the daily loss limit."""
    worst = float(min(0.0, daily_r(trades, period).min())) if not trades.empty else 0.0
    concurrent = max_concurrent_positions(trades)
    return PropFirmResult(
        worst_day_r=worst,
        max_concurrent=concurrent,
        max_risk_pct_daily=prop.max_daily_loss_pct / -worst if worst < 0 else float("inf"),
        max_risk_pct_all_stopped=prop.max_daily_loss_pct / concurrent if concurrent else float("inf"),
    )
