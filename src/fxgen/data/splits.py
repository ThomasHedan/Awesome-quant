"""Temporal splits (train / validation / holdout) and walk-forward windows."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd

from fxgen.config import SplitsConfig, WalkForwardConfig
from fxgen.errors import HoldoutLockedError

Split = Literal["train", "validation", "holdout", "train+validation"]


@dataclass(frozen=True)
class Period:
    """Half-open interval ``[start, end)`` in UTC."""

    start: pd.Timestamp
    end: pd.Timestamp

    def mask(self, index: pd.DatetimeIndex) -> pd.Series:
        return pd.Series((index >= self.start) & (index < self.end), index=index)

    def label(self) -> str:
        return f"{self.start.date()}..{self.end.date()}"


def _ts(value: str) -> pd.Timestamp:
    return pd.Timestamp(value, tz="UTC")


def split_period(splits: SplitsConfig, split: Split, allow_holdout: bool = False) -> Period:
    """Return the period for a named split. The holdout requires explicit opt-in."""
    train, val, hold = _ts(splits.train_start), _ts(splits.validation_start), _ts(splits.holdout_start)
    end = _ts(splits.end) + pd.Timedelta(days=1)
    if split == "holdout":
        if not allow_holdout:
            raise HoldoutLockedError("the holdout split is locked; it can only be read by the final holdout stage")
        return Period(hold, end)
    periods = {
        "train": Period(train, val),
        "validation": Period(val, hold),
        "train+validation": Period(train, hold),
    }
    return periods[split]


def walkforward_windows(splits: SplitsConfig, wf: WalkForwardConfig) -> list[tuple[Period, Period]]:
    """Rolling (in-sample, out-of-sample) windows ending at the holdout start.

    The last OOS window ends exactly at the holdout start; earlier windows step
    back by ``test_years``. Windows starting before ``train_start`` are dropped.
    """
    start, end = _ts(splits.train_start), _ts(splits.holdout_start)
    train_len = pd.Timedelta(days=round(365.25 * wf.train_years))
    test_len = pd.Timedelta(days=round(365.25 * wf.test_years))
    windows: list[tuple[Period, Period]] = []
    oos_end = end
    for _ in range(wf.n_windows):
        oos_start = oos_end - test_len
        is_start = oos_start - train_len
        if is_start < start:
            break
        windows.append((Period(is_start, oos_start), Period(oos_start, oos_end)))
        oos_end = oos_start
    return list(reversed(windows))


def subperiods(period: Period, n: int) -> list[Period]:
    """Split a period into ``n`` equal consecutive sub-periods."""
    edges = pd.date_range(period.start, period.end, periods=n + 1)
    return [Period(a, b) for a, b in zip(edges[:-1], edges[1:])]
