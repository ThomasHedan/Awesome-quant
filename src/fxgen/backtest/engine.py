"""Path-dependent trade simulation.

Timing contract: the tree is evaluated at the close of bar ``t``; any order
resulting from it is filled at the open of bar ``t + 1``. Stops and targets are
set from the ATR known at the decision bar and checked against each bar's
high/low (stop first when both are touched, fills at the open on gaps).

After a stop, target or time exit the same branch cannot re-enter until the
tree leaves it, so one signal episode produces at most one such trade.

Entries and leaf assignments are computed vectorised upstream; only the
position state machine runs bar by bar, compiled with numba, because exits
depend on the path.
"""

from __future__ import annotations

from dataclasses import dataclass

import numba
import numpy as np
import pandas as pd

from fxgen.config import BacktestConfig, CostsConfig
from fxgen.market import PairSplit
from fxgen.timeframes import pip_size

EXIT_REASONS = ("stop", "target", "time", "signal", "end")
_STOP, _TARGET, _TIME, _SIGNAL, _END = range(5)


@dataclass(frozen=True)
class ExitRules:
    stop_atr: float
    take_profit_atr: float | None
    max_holding_bars: int | None
    exit_on_leaf_change: bool

    @classmethod
    def from_config(cls, cfg: BacktestConfig) -> ExitRules:
        return cls(cfg.stop_atr, cfg.take_profit_atr, cfg.max_holding_bars, cfg.exit_on_leaf_change)

    def scaled(self, factor: float) -> ExitRules:
        return ExitRules(
            self.stop_atr * factor,
            None if self.take_profit_atr is None else self.take_profit_atr * factor,
            None if self.max_holding_bars is None else max(1, round(self.max_holding_bars * factor)),
            self.exit_on_leaf_change,
        )


@numba.njit(cache=True)
def _simulate(
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    close: np.ndarray,
    atr: np.ndarray,
    action: np.ndarray,
    leaf: np.ndarray,
    stop_atr: float,
    tp_atr: float,
    max_hold: int,
    exit_on_leaf_change: bool,
) -> np.ndarray:
    n = open_.shape[0]
    # columns: entry_idx, exit_idx, direction, entry_price, exit_price, stop_dist, reason, leaf
    out = np.empty((n, 8))
    k = 0
    pos = 0
    entry_idx = 0
    entry_price = 0.0
    stop = 0.0
    target = 0.0
    stop_dist = 0.0
    entry_leaf = 0
    blocked = -1
    for t in range(n - 1):
        j = t + 1
        if blocked >= 0 and leaf[t] != blocked:
            blocked = -1
        if pos != 0:
            exit_price = np.nan
            reason = -1
            if action[t] == -pos or (exit_on_leaf_change and leaf[t] != entry_leaf):
                exit_price, reason = open_[j], _SIGNAL
            elif max_hold > 0 and j - entry_idx >= max_hold:
                exit_price, reason = open_[j], _TIME
            if reason >= 0:
                out[k, 0] = entry_idx
                out[k, 1] = j
                out[k, 2] = pos
                out[k, 3] = entry_price
                out[k, 4] = exit_price
                out[k, 5] = stop_dist
                out[k, 6] = reason
                out[k, 7] = entry_leaf
                k += 1
                pos = 0
                if reason == _TIME:
                    blocked = entry_leaf
        if pos == 0 and action[t] != 0 and atr[t] > 0 and blocked < 0:
            pos = int(action[t])
            entry_idx = j
            entry_price = open_[j]
            stop_dist = stop_atr * atr[t]
            stop = entry_price - pos * stop_dist
            target = entry_price + pos * tp_atr * atr[t] if tp_atr > 0 else np.nan
            entry_leaf = leaf[t]
        if pos != 0:
            exit_price = np.nan
            reason = -1
            if pos == 1:
                if open_[j] <= stop and j > entry_idx:
                    exit_price, reason = open_[j], _STOP
                elif low[j] <= stop:
                    exit_price, reason = stop, _STOP
                elif tp_atr > 0 and open_[j] >= target and j > entry_idx:
                    exit_price, reason = open_[j], _TARGET
                elif tp_atr > 0 and high[j] >= target:
                    exit_price, reason = target, _TARGET
            else:
                if open_[j] >= stop and j > entry_idx:
                    exit_price, reason = open_[j], _STOP
                elif high[j] >= stop:
                    exit_price, reason = stop, _STOP
                elif tp_atr > 0 and open_[j] <= target and j > entry_idx:
                    exit_price, reason = open_[j], _TARGET
                elif tp_atr > 0 and low[j] <= target:
                    exit_price, reason = target, _TARGET
            if reason >= 0:
                out[k, 0] = entry_idx
                out[k, 1] = j
                out[k, 2] = pos
                out[k, 3] = entry_price
                out[k, 4] = exit_price
                out[k, 5] = stop_dist
                out[k, 6] = reason
                out[k, 7] = entry_leaf
                k += 1
                pos = 0
                blocked = entry_leaf
    if pos != 0:
        out[k, 0] = entry_idx
        out[k, 1] = n - 1
        out[k, 2] = pos
        out[k, 3] = entry_price
        out[k, 4] = close[n - 1]
        out[k, 5] = stop_dist
        out[k, 6] = _END
        out[k, 7] = entry_leaf
        k += 1
    return out[:k]


def rollover_units(index: pd.DatetimeIndex) -> np.ndarray:
    """Cumulative swap units at each bar open (22:00 UTC rollover, triple on Wednesday)."""
    if len(index) == 0:
        return np.zeros(0)
    days = pd.date_range(index[0].normalize() - pd.Timedelta(days=1), index[-1].normalize() + pd.Timedelta(days=1), freq="D")
    instants = days + pd.Timedelta(hours=22)
    weights = np.where(instants.day_of_week == 2, 3.0, 1.0)
    weights[instants.day_of_week >= 5] = 0.0
    cum = np.concatenate([[0.0], np.cumsum(weights)])
    return cum[np.searchsorted(instants.asi8, index.asi8, side="right")]


TRADE_COLUMNS = [
    "pair", "entry_time", "exit_time", "direction", "entry_price", "exit_price", "bars_held",
    "gross_pips", "cost_pips", "net_pips", "r_multiple", "exit_reason", "leaf",
]


def simulate_pair(
    data: PairSplit,
    action: np.ndarray,
    leaf: np.ndarray,
    rules: ExitRules,
    costs: CostsConfig,
) -> pd.DataFrame:
    """Simulate one pair and return its trades (one row per trade, net of costs)."""
    raw = _simulate(
        data.open, data.high, data.low, data.close, data.atr, action.astype(np.int8), leaf.astype(np.int16),
        rules.stop_atr, rules.take_profit_atr or 0.0, rules.max_holding_bars or 0, rules.exit_on_leaf_change,
    )
    if raw.shape[0] == 0:
        return pd.DataFrame(columns=TRADE_COLUMNS)
    pip = pip_size(data.pair)
    entry_idx = raw[:, 0].astype(np.int64)
    exit_idx = raw[:, 1].astype(np.int64)
    direction = raw[:, 2]
    gross = direction * (raw[:, 4] - raw[:, 3]) / pip
    units = rollover_units(data.index)
    nights = units[exit_idx] - units[entry_idx]
    swap = np.where(direction > 0, costs.swap_pips_long, costs.swap_pips_short) * nights
    cost = costs.spread_for(data.pair) + costs.commission_pips_round_turn - swap
    net = gross - cost
    stop_pips = raw[:, 5] / pip
    return pd.DataFrame(
        {
            "pair": data.pair,
            "entry_time": data.index[entry_idx],
            "exit_time": data.index[exit_idx],
            "direction": direction.astype(np.int8),
            "entry_price": raw[:, 3],
            "exit_price": raw[:, 4],
            "bars_held": exit_idx - entry_idx,
            "gross_pips": gross,
            "cost_pips": cost,
            "net_pips": net,
            "r_multiple": net / stop_pips,
            "exit_reason": [EXIT_REASONS[int(r)] for r in raw[:, 6]],
            "leaf": raw[:, 7].astype(np.int16),
        }
    )
