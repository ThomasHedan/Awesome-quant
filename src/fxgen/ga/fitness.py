"""Robust fitness of a tree on the training dataset.

    fitness = sharpe x pair_stability x pair_concentration x period_stability
              - complexity_penalty x internal_nodes

* ``sharpe``: annualised Sharpe of the equal-risk portfolio of all pairs.
* ``pair_stability``: share of pairs with a positive total R.
* ``pair_concentration``: 1 when no pair carries more than
  ``max_pair_pnl_share`` of the gross positive R, scaled down otherwise.
* ``period_stability``: 1 when at least ``min_positive_subperiods`` of the
  ``n_subperiods`` train slices are positive, scaled down otherwise.

Too few trades in total or on any pair, or more than
``max_trades_per_pair_year`` on average (not swing trading), gives exactly 0.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fxgen.backtest.metrics import daily_r, sharpe
from fxgen.backtest.runner import BacktestResult
from fxgen.config import FitnessConfig
from fxgen.data.splits import Period, subperiods
from fxgen.tree.node import Tree, n_internal


@dataclass
class FitnessResult:
    value: float
    sharpe: float
    valid: bool
    components: dict[str, float] = field(default_factory=dict)
    reason: str = ""


def pair_totals(result: BacktestResult) -> dict[str, float]:
    return {pair: float(m["total_r"]) for pair, m in result.per_pair.items()}


def concentration(totals: dict[str, float]) -> float:
    """Largest share of the gross positive R carried by a single pair."""
    positive = np.array([v for v in totals.values() if v > 0])
    return float(positive.max() / positive.sum()) if positive.size and positive.sum() > 0 else 1.0


def positive_subperiods(trades: pd.DataFrame, period: Period, n: int) -> int:
    if trades.empty:
        return 0
    count = 0
    for sub in subperiods(period, n):
        in_sub = (trades["exit_time"] >= sub.start) & (trades["exit_time"] < sub.end)
        count += int(trades.loc[in_sub, "r_multiple"].sum() > 0)
    return count


def compute_fitness(tree: Tree, result: BacktestResult, period: Period, cfg: FitnessConfig) -> FitnessResult:
    """Fitness of ``tree`` given its backtest on the training period."""
    trades = result.trades
    portfolio_sharpe = sharpe(daily_r(trades, period))
    counts = [m["n_trades"] for m in result.per_pair.values()]
    if len(trades) < cfg.min_trades_total:
        return FitnessResult(0.0, portfolio_sharpe, False, reason=f"{len(trades)} trades < {cfg.min_trades_total}")
    if min(counts, default=0) < cfg.min_trades_per_pair:
        return FitnessResult(0.0, portfolio_sharpe, False, reason=f"a pair has {min(counts)} trades < {cfg.min_trades_per_pair}")

    if cfg.max_trades_per_pair_year is not None and counts:
        years = max((period.end - period.start).days / 365.25, 1e-9)
        per_year = len(trades) / len(counts) / years
        if per_year > cfg.max_trades_per_pair_year:
            return FitnessResult(0.0, portfolio_sharpe, False, reason=f"{per_year:.0f} trades/pair/year > {cfg.max_trades_per_pair_year:g}")

    totals = pair_totals(result)
    pair_stability = float(np.mean([v > 0 for v in totals.values()]))
    share = concentration(totals)
    pair_concentration = min(1.0, cfg.max_pair_pnl_share / share) if share > 0 else 1.0
    n_pos = positive_subperiods(trades, period, cfg.n_subperiods)
    period_stability = min(1.0, n_pos / cfg.min_positive_subperiods) if cfg.min_positive_subperiods else 1.0
    complexity = cfg.complexity_penalty * n_internal(tree)

    if portfolio_sharpe > 0:
        value = portfolio_sharpe * pair_stability * pair_concentration * period_stability - complexity
    else:
        value = portfolio_sharpe - complexity
    return FitnessResult(
        value=float(value),
        sharpe=portfolio_sharpe,
        valid=True,
        components={
            "sharpe": portfolio_sharpe,
            "pair_stability": pair_stability,
            "max_pair_share": share,
            "pair_concentration": pair_concentration,
            "positive_subperiods": float(n_pos),
            "period_stability": period_stability,
            "complexity": complexity,
            "n_trades": float(len(trades)),
        },
    )
