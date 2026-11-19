"""Binary conditions (tree nodes) built from features.

Condition ids are a small grammar so that hand-written trees can reference
any condition, even one that was not kept after redundancy filtering:

    ``H4.rsi14>p75``        feature above its 75th train percentile
    ``H4.rsi14>70``         feature above an absolute level
    ``D1.dist_ema200>0``    signed feature above zero (position vs another series)
    ``H1.macd_hist:rising`` feature higher than ``RISING_LAG`` bars ago
    ``H4.ema20_50:cross_up`` zero cross up within the last ``CROSS_WINDOW`` bars
    ``H1.cal:session_london`` calendar rule

A condition is evaluated at the close of the decision bar. Rising and cross
conditions are computed on the feature's own timeframe, then aligned.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, replace
from typing import Literal

import numpy as np
import pandas as pd

from fxgen.errors import TreeError
from fxgen.indicators.base import Feature
from fxgen.indicators.time import CALENDAR_RULES

Kind = Literal["above", "rising", "cross_up", "cross_down", "calendar"]
RISING_LAG = 3
CROSS_WINDOW = 5
_ID_RE = re.compile(
    r"^(?P<tf>H1|H4|D1)\.(?P<feature>\w+)"
    r"(?:>(?P<level>p\d{1,2}|-?\d+(?:\.\d+)?)|:(?P<kind>rising|cross_up|cross_down|session_\w+|day_\w+))$"
)


@dataclass(frozen=True)
class Condition:
    """One binary node. ``threshold`` is filled for ``above`` once fitted."""

    id: str
    family: str
    timeframe: str
    feature: str
    kind: Kind
    percentile: int | None = None
    threshold: float | None = None
    label: str = ""

    @property
    def column(self) -> str:
        return f"{self.timeframe}.{self.feature}"

    def describe(self) -> str:
        if self.kind == "above":
            level = f"{self.threshold:.4g}" if self.threshold is not None else "?"
            suffix = f" (p{self.percentile})" if self.percentile is not None else ""
            return f"{self.timeframe} {self.label} > {level}{suffix}"
        if self.kind == "rising":
            return f"{self.timeframe} {self.label} rising over {RISING_LAG} bars"
        if self.kind in ("cross_up", "cross_down"):
            way = "above" if self.kind == "cross_up" else "below"
            return f"{self.timeframe} {self.label} crossed {way} 0 within {CROSS_WINDOW} bars"
        return self.label

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, object]) -> Condition:
        return cls(**data)  # type: ignore[arg-type]


def _feature_index(features: list[Feature]) -> dict[str, Feature]:
    return {f.name: f for f in features}


def parse_condition_id(cid: str, features: list[Feature]) -> Condition:
    """Build an (unfitted) Condition from its id."""
    match = _ID_RE.match(cid)
    if not match:
        raise TreeError(f"invalid condition id {cid!r}; expected e.g. 'H4.rsi14>p75' or 'D1.dist_ema200:rising'")
    tf, name, level, kind = match["tf"], match["feature"], match["level"], match["kind"]
    if name == "cal":
        rules = {r.name: r for r in CALENDAR_RULES}
        if kind not in rules:
            raise TreeError(f"unknown calendar rule {kind!r}; available: {sorted(rules)}")
        return Condition(id=cid, family="time", timeframe=tf, feature="cal", kind="calendar", label=rules[kind].label)
    index = _feature_index(features)
    if name not in index:
        raise TreeError(f"unknown feature {name!r} in condition {cid!r}")
    feature = index[name]
    if level is not None:
        if level.startswith("p"):
            return Condition(cid, feature.family, tf, name, "above", percentile=int(level[1:]), label=feature.label)
        return Condition(cid, feature.family, tf, name, "above", threshold=float(level), label=feature.label)
    if kind not in ("rising", "cross_up", "cross_down"):
        raise TreeError(f"condition kind {kind!r} only applies to calendar rules")
    return Condition(cid, feature.family, tf, name, kind, label=feature.label)  # type: ignore[arg-type]


def calendar_rule_name(cond: Condition) -> str:
    return cond.id.split(":", 1)[1]


def generate_candidates(features: list[Feature], timeframes: list[str], percentiles: list[int]) -> list[Condition]:
    """All candidate conditions before threshold fitting and redundancy filtering.

    Only ``above`` conditions are generated: ``below`` is the false branch of
    the same node, so it would be a pure duplicate.
    """
    out: list[Condition] = []
    exec_tf = timeframes[0]
    for rule in CALENDAR_RULES:
        cid = f"{exec_tf}.cal:{rule.name}"
        out.append(Condition(cid, "time", exec_tf, "cal", "calendar", label=rule.label))
    for tf in timeframes:
        for f in features:
            base = f"{tf}.{f.name}"
            for p in percentiles:
                out.append(Condition(f"{base}>p{p}", f.family, tf, f.name, "above", percentile=p, label=f.label))
            if f.signed:
                out.append(Condition(f"{base}>0", f.family, tf, f.name, "above", threshold=0.0, label=f.label))
            out.append(Condition(f"{base}:rising", f.family, tf, f.name, "rising", label=f.label))
            if f.crosses:
                out.append(Condition(f"{base}:cross_up", f.family, tf, f.name, "cross_up", label=f.label))
                out.append(Condition(f"{base}:cross_down", f.family, tf, f.name, "cross_down", label=f.label))
    return out


def native_state(values: pd.Series, kind: Kind) -> pd.Series:
    """Rising / cross state computed on the feature's own timeframe (float 0/1, NaN in warmup)."""
    if kind == "rising":
        prev = values.shift(RISING_LAG)
        state = (values > prev).astype("float32")
        return state.where(values.notna() & prev.notna())
    prev = values.shift(1)
    if kind == "cross_up":
        event = (values > 0) & (prev <= 0)
    else:
        event = (values < 0) & (prev >= 0)
    return event.astype("float32").rolling(CROSS_WINDOW, min_periods=1).max().where(values.notna())


def with_threshold(cond: Condition, threshold: float) -> Condition:
    return replace(cond, threshold=float(threshold))


def evaluate_above(values: np.ndarray, threshold: float) -> np.ndarray:
    """``values > threshold`` with NaN mapped to False."""
    with np.errstate(invalid="ignore"):
        return np.nan_to_num(values, nan=-np.inf) > threshold
