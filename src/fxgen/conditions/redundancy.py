"""Redundancy analysis: correlation of conditions and removal of near duplicates."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

from fxgen.conditions.builder import Condition
from fxgen.data.splits import Period
from fxgen.market import PairFrame, condition_column

logger = logging.getLogger(__name__)


@dataclass
class RedundancyReport:
    kept: list[Condition]
    dropped_inactive: list[str] = field(default_factory=list)
    dropped_redundant: dict[str, tuple[str, float]] = field(default_factory=dict)
    activation: dict[str, float] = field(default_factory=dict)


def sample_matrix(
    frames: dict[str, PairFrame], conditions: list[Condition], period: Period, max_rows: int, seed: int
) -> np.ndarray:
    """Boolean condition matrix over ``period``, pooled over pairs, subsampled."""
    rng = np.random.default_rng(seed)
    per_pair = max(1, max_rows // max(1, len(frames)))
    blocks = []
    for frame in frames.values():
        mask = period.mask(frame.ohlc.index).to_numpy() & frame.atr.notna().to_numpy()
        rows = np.flatnonzero(mask)
        if rows.size > per_pair:
            rows = np.sort(rng.choice(rows, per_pair, replace=False))
        sel = np.zeros(len(mask), dtype=bool)
        sel[rows] = True
        blocks.append(np.column_stack([condition_column(frame, c, sel) for c in conditions]))
    return np.vstack(blocks)


def correlation_matrix(X: np.ndarray, chunk: int = 4096) -> np.ndarray:
    """Phi coefficient (Pearson on 0/1) between columns; constant columns get 0.

    Accumulated over row chunks so thousands of conditions fit in memory.
    """
    n, m = X.shape
    cross = np.zeros((m, m), dtype=np.float32)
    for start in range(0, n, chunk):
        block = X[start : start + chunk].astype(np.float32)
        cross += block.T @ block
    p = X.mean(axis=0, dtype=np.float64).astype(np.float32)
    std = np.sqrt(np.clip(p * (1 - p), 0, None))
    std[std == 0] = np.inf
    cross /= n
    cross -= np.outer(p, p)
    cross /= std[:, None]
    cross /= std[None, :]
    return cross

def filter_conditions(
    frames: dict[str, PairFrame],
    conditions: list[Condition],
    train: Period,
    threshold: float,
    min_rate: float,
    max_rate: float,
    max_rows: int = 40_000,
    seed: int = 0,
) -> RedundancyReport:
    """Drop rarely/always-true conditions, then greedy-remove near duplicates.

    Conditions are visited from the most balanced (activation near 50%) to the
    least; a condition is kept if its |phi| with every kept one is <= threshold.
    """
    X = sample_matrix(frames, conditions, train, max_rows, seed)
    rates = X.mean(axis=0)
    report = RedundancyReport(kept=[], activation={c.id: float(r) for c, r in zip(conditions, rates)})
    active = [i for i, r in enumerate(rates) if min_rate <= r <= max_rate]
    report.dropped_inactive = [conditions[i].id for i in range(len(conditions)) if i not in set(active)]
    order = sorted(active, key=lambda i: (abs(rates[i] - 0.5), conditions[i].id))
    corr = np.abs(correlation_matrix(X[:, order]))
    kept_local: list[int] = []
    for local, i in enumerate(order):
        if kept_local:
            row = corr[local, kept_local]
            worst = int(np.argmax(row))
            if row[worst] > threshold:
                report.dropped_redundant[conditions[i].id] = (conditions[order[kept_local[worst]]].id, float(row[worst]))
                continue
        kept_local.append(local)
    kept_ids = {order[k] for k in kept_local}
    report.kept = [c for i, c in enumerate(conditions) if i in kept_ids]
    logger.info(
        "conditions: %d candidates, %d inactive, %d redundant, %d kept",
        len(conditions), len(report.dropped_inactive), len(report.dropped_redundant), len(report.kept),
    )
    return report
