"""Calendar conditions on the execution timeframe (UTC)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class CalendarRule:
    """A boolean calendar condition evaluated on the decision bar's close time."""

    name: str
    label: str
    pine: str
    hours: tuple[int, ...] = ()
    weekdays: tuple[int, ...] = ()

    def evaluate(self, decision_times: pd.DatetimeIndex) -> np.ndarray:
        if self.hours:
            return np.isin(decision_times.hour, self.hours)
        return np.isin(decision_times.day_of_week, self.weekdays)


def _hours(start: int, end: int) -> tuple[int, ...]:
    return tuple(h % 24 for h in range(start, end if end > start else end + 24))


_SESSIONS = {"asia": (22, 7), "london": (7, 16), "newyork": (12, 21), "overlap": (12, 16)}
_DAYS = ["monday", "tuesday", "wednesday", "thursday", "friday"]
_PINE_DAYS = ["dayofweek.monday", "dayofweek.tuesday", "dayofweek.wednesday", "dayofweek.thursday", "dayofweek.friday"]

CALENDAR_RULES: list[CalendarRule] = [
    CalendarRule(
        name=f"session_{name}",
        label=f"decision time in {name} session ({a:02d}-{b:02d} UTC)",
        pine=f"sessionHour >= {a} or sessionHour < {b}" if a > b else f"sessionHour >= {a} and sessionHour < {b}",
        hours=_hours(a, b),
    )
    for name, (a, b) in _SESSIONS.items()
] + [
    CalendarRule(
        name=f"day_{day}",
        label=f"decision day is {day.capitalize()} (UTC)",
        pine=f"sessionDay == {pine}",
        weekdays=(i,),
    )
    for i, (day, pine) in enumerate(zip(_DAYS, _PINE_DAYS))
]
