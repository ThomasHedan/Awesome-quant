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
    month_days: tuple[int, ...] = ()

    def evaluate(self, decision_times: pd.DatetimeIndex) -> np.ndarray:
        out = np.ones(len(decision_times), dtype=bool)
        if self.hours:
            out &= np.isin(decision_times.hour, self.hours)
        if self.weekdays:
            out &= np.isin(decision_times.day_of_week, self.weekdays)
        if self.month_days:
            out &= np.isin(decision_times.day, self.month_days)
        return out


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


def _hour_rule(name: str, start: int, end: int, label: str) -> CalendarRule:
    return CalendarRule(
        name=f"hour_{name}",
        label=f"decision time {label} ({start:02d}-{end:02d} UTC)",
        pine=f"sessionHour >= {start} or sessionHour < {end}" if start > end else f"sessionHour >= {start} and sessionHour < {end}",
        hours=_hours(start, end),
    )


CALENDAR_RULES += [
    _hour_rule("asia_early", 22, 2, "early Asia"),
    _hour_rule("tokyo", 0, 6, "Tokyo"),
    _hour_rule("london_open", 7, 10, "London open"),
    _hour_rule("london_midday", 10, 13, "London midday"),
    _hour_rule("ny_open", 13, 16, "New York open"),
    _hour_rule("ny_afternoon", 16, 19, "New York afternoon"),
    _hour_rule("late_ny", 19, 22, "late New York"),
    CalendarRule(name="day_monday_am", label="Monday before 10 UTC", pine="sessionDay == dayofweek.monday and sessionHour < 10",
                 hours=tuple(range(0, 10)), weekdays=(0,)),
    CalendarRule(name="day_friday_pm", label="Friday from 14 UTC", pine="sessionDay == dayofweek.friday and sessionHour >= 14",
                 hours=tuple(range(14, 24)), weekdays=(4,)),
    CalendarRule(name="month_start", label="first 3 days of the month", pine="sessionDom <= 3", month_days=(1, 2, 3)),
    CalendarRule(name="month_mid", label="days 13-17 of the month", pine="sessionDom >= 13 and sessionDom <= 17",
                 month_days=(13, 14, 15, 16, 17)),
    CalendarRule(name="month_end", label="from the 27th of the month", pine="sessionDom >= 27", month_days=(27, 28, 29, 30, 31)),
]
