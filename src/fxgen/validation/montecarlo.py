"""Monte Carlo on the order of trades: drawdown distribution and prop-firm breach odds.

``max_risk_pct_for_limit`` is the risk per trade at which the drawdown at the
chosen confidence equals the prop-firm maximum drawdown.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

from fxgen.config import MonteCarloConfig, PropFirmConfig


@dataclass(frozen=True)
class MonteCarloResult:
    n_sims: int
    n_trades: int
    max_dd_r_median: float
    max_dd_r_at_conf: float
    max_dd_pct_at_conf: float
    total_r_p05: float
    total_r_p50: float
    prob_breach_total_dd: float
    max_risk_pct_for_limit: float
    confidence: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def _max_dd(equity_paths: np.ndarray) -> np.ndarray:
    zero = np.zeros((equity_paths.shape[0], 1))
    curves = np.hstack([zero, np.cumsum(equity_paths, axis=1)])
    return np.max(np.maximum.accumulate(curves, axis=1) - curves, axis=1)


def run_monte_carlo(
    r_multiples: np.ndarray, risk_pct: float, mc: MonteCarloConfig, prop: PropFirmConfig, seed: int
) -> MonteCarloResult:
    """Shuffle the trade order (drawdowns) and bootstrap trades (total R spread)."""
    r = np.asarray(r_multiples, dtype=float)
    rng = np.random.default_rng(seed)
    if r.size == 0:
        return MonteCarloResult(mc.n_sims, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, mc.confidence)
    shuffled = np.stack([rng.permutation(r) for _ in range(mc.n_sims)])
    dd = _max_dd(shuffled)
    boot = rng.choice(r, size=(mc.n_sims, r.size), replace=True)
    totals = boot.sum(axis=1)
    dd_conf = float(np.quantile(dd, mc.confidence))
    return MonteCarloResult(
        n_sims=mc.n_sims,
        n_trades=int(r.size),
        max_dd_r_median=float(np.median(dd)),
        max_dd_r_at_conf=dd_conf,
        max_dd_pct_at_conf=dd_conf * risk_pct,
        total_r_p05=float(np.quantile(totals, 0.05)),
        total_r_p50=float(np.median(totals)),
        prob_breach_total_dd=float(np.mean(dd * risk_pct >= prop.max_total_drawdown_pct)),
        max_risk_pct_for_limit=float(prop.max_total_drawdown_pct / dd_conf) if dd_conf > 0 else float("inf"),
        confidence=mc.confidence,
    )
