"""Percentile thresholds fitted on the training period only, pooled over pairs."""

from __future__ import annotations

import numpy as np

from fxgen.conditions.builder import Condition, with_threshold
from fxgen.data.splits import Period
from fxgen.market import PairFrame, is_available


def pooled_train_values(frames: dict[str, PairFrame], column: str, period: Period) -> np.ndarray:
    chunks = [f.features[column].to_numpy()[period.mask(f.features.index).to_numpy()] for f in frames.values()]
    values = np.concatenate(chunks) if chunks else np.array([])
    return values[~np.isnan(values)]


def fit_thresholds(frames: dict[str, PairFrame], conditions: list[Condition], train: Period) -> list[Condition]:
    """Fill the threshold of percentile conditions from pooled train values.

    Conditions whose feature has no train values (e.g. volume missing) are dropped.
    """
    cache: dict[str, np.ndarray] = {}
    fitted: list[Condition] = []
    for cond in conditions:
        if not all(is_available(f, cond) for f in frames.values()):
            continue
        if cond.kind != "above" or cond.percentile is None:
            fitted.append(cond)
            continue
        if cond.column not in cache:
            cache[cond.column] = pooled_train_values(frames, cond.column, train)
        values = cache[cond.column]
        if values.size == 0:
            continue
        fitted.append(with_threshold(cond, float(np.percentile(values, cond.percentile))))
    return fitted


def shifted_percentile(cond: Condition, frames: dict[str, PairFrame], train: Period, shift: int) -> Condition | None:
    """Same condition with its percentile moved by ``shift`` points (sensitivity tests)."""
    if cond.kind != "above" or cond.percentile is None:
        return None
    p = min(99, max(1, cond.percentile + shift))
    values = pooled_train_values(frames, cond.column, train)
    return with_threshold(cond, float(np.percentile(values, p)))
