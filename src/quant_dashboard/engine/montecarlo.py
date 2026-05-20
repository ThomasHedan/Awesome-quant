"""Monte Carlo edge analysis.

Two methods, both seeded for reproducibility:

* ``shuffle`` — randomly reorder the realized per-trade returns. Asks: would
  this equity curve still look like an edge if the trades had happened in a
  different sequence?
* ``bootstrap`` — sample per-bar returns with replacement to build synthetic
  equity curves of the same length. Asks: how often does a random arrangement
  of these returns produce a result this good?

For each method we report the distribution of terminal returns, Sharpe, and
max drawdown; probability of ruin (max drawdown <= -``ruin_threshold``);
probability of clearing a return goal; and the percentile rank of the
realized Sharpe and total return inside the simulated distribution — the
single best "is the edge real?" number.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import pandas as pd

from quant_dashboard.engine.runner import BacktestResult, periods_per_year


class MonteCarloError(RuntimeError):
    """User-actionable Monte Carlo configuration errors."""


Method = Literal["shuffle", "bootstrap"]


@dataclass
class MonteCarloResult:
    """Distribution of simulated outcomes + edge-vs-luck summary."""

    method: Method
    n_simulations: int
    seed: int

    # Realized values (from the actual backtest).
    realized_total_return: float
    realized_sharpe: float
    realized_max_drawdown: float

    # Per-simulation aggregates.
    terminal_returns: np.ndarray
    sharpes: np.ndarray
    max_drawdowns: np.ndarray

    # Confidence-interval equity fan (each column is a percentile equity
    # curve indexed by bar position 0..N).
    equity_curves_ci: pd.DataFrame
    realized_equity_normalized: pd.Series  # starts at 1.0

    # Edge-vs-luck summary.
    prob_ruin: float
    prob_goal: float
    sharpe_percentile: float       # 0..100; what % of sims this Sharpe beats
    return_percentile: float       # same, for total return

    goal_return: float
    ruin_threshold: float

    meta: dict = field(default_factory=dict)

    def to_summary(self) -> dict:
        return {
            "method": self.method,
            "n_simulations": self.n_simulations,
            "realized_total_return": self.realized_total_return,
            "realized_sharpe": self.realized_sharpe,
            "realized_max_drawdown": self.realized_max_drawdown,
            "prob_ruin": self.prob_ruin,
            "prob_goal": self.prob_goal,
            "sharpe_percentile": self.sharpe_percentile,
            "return_percentile": self.return_percentile,
            "goal_return": self.goal_return,
            "ruin_threshold": self.ruin_threshold,
        }


# ---------------------------------------------------------------------------


def _per_trade_returns(result: BacktestResult) -> np.ndarray:
    """Per-trade percentage returns (PnL / entry value). Empty if no trades."""
    trades = result.portfolio.trades
    if trades.count() == 0:
        return np.array([], dtype=float)
    rec = trades.records_readable
    # vectorbt's "Return" column already expresses PnL as a fraction of the
    # position's entry value, which is exactly what we want to recompound.
    if "Return" in rec.columns:
        return rec["Return"].to_numpy(dtype=float)
    # Fallback: PnL / (Size * Avg Entry Price).
    pnl = rec["PnL"].to_numpy(dtype=float)
    size = rec["Size"].to_numpy(dtype=float)
    entry = rec["Avg Entry Price"].to_numpy(dtype=float)
    notional = np.where((size * entry) != 0, size * entry, np.nan)
    return pnl / notional


def _equity_from_per_bar_returns(rets: np.ndarray) -> np.ndarray:
    return np.cumprod(1.0 + rets)


def _max_drawdown_from_equity(equity: np.ndarray) -> float:
    if equity.size == 0:
        return 0.0
    running_max = np.maximum.accumulate(equity)
    drawdown = equity / running_max - 1.0
    return float(drawdown.min())


def _sharpe_from_per_bar_returns(rets: np.ndarray, ppy: float) -> float:
    if rets.size < 2:
        return float("nan")
    std = float(rets.std(ddof=1))
    if std == 0:
        return float("nan")
    return float(rets.mean() / std) * float(np.sqrt(ppy))


def _percentile_rank(value: float, distribution: np.ndarray) -> float:
    """Percentage of the distribution that ``value`` strictly beats. 50 = median."""
    if distribution.size == 0 or not np.isfinite(value):
        return float("nan")
    finite = distribution[np.isfinite(distribution)]
    if finite.size == 0:
        return float("nan")
    return 100.0 * float((finite < value).mean())


# ---------------------------------------------------------------------------


def run_monte_carlo(
    result: BacktestResult,
    *,
    method: Method = "bootstrap",
    n_simulations: int = 1000,
    seed: int = 42,
    goal_return: float = 0.20,
    ruin_threshold: float = 0.50,
) -> MonteCarloResult:
    """Run a Monte Carlo analysis on ``result``.

    Parameters
    ----------
    method:
        ``"bootstrap"`` resamples per-bar returns with replacement. Best for
        answering "could this Sharpe come from this distribution by chance?".
        ``"shuffle"`` reorders the realized per-trade returns. Best for
        answering "would a different trade sequence have looked the same?".
    n_simulations:
        Number of synthetic paths.
    seed:
        Seed for the RNG; the same seed produces identical results.
    goal_return:
        Total-return target used by ``prob_goal`` (e.g. 0.20 = +20%).
    ruin_threshold:
        Drawdown depth used by ``prob_ruin`` (e.g. 0.50 = 50% max drawdown).
    """
    if n_simulations < 10:
        raise MonteCarloError("n_simulations must be >= 10")
    if ruin_threshold <= 0 or ruin_threshold >= 1:
        raise MonteCarloError("ruin_threshold must be in (0, 1)")

    rng = np.random.default_rng(seed)
    ppy = periods_per_year(result.data.df.index)

    if method == "bootstrap":
        per_bar = result.returns.to_numpy(dtype=float)
        if per_bar.size < 2:
            raise MonteCarloError("not enough return observations for bootstrap")
        n_bars = per_bar.size
        # n_simulations x n_bars matrix of resampled returns.
        idx = rng.integers(0, n_bars, size=(n_simulations, n_bars))
        synthetic = per_bar[idx]
    elif method == "shuffle":
        per_trade = _per_trade_returns(result)
        if per_trade.size < 2:
            raise MonteCarloError(
                "not enough trades for shuffle method (need >= 2). "
                "Try the bootstrap method instead."
            )
        n_steps = per_trade.size
        synthetic = np.empty((n_simulations, n_steps), dtype=float)
        for i in range(n_simulations):
            rng.shuffle(per_trade)  # in-place; deterministic given the seeded rng
            synthetic[i] = per_trade
    else:  # pragma: no cover - exhausted by Literal
        raise MonteCarloError(f"unknown method '{method}'")

    # Equity per simulation, starting at 1.0.
    equity = np.cumprod(1.0 + synthetic, axis=1)
    terminal = equity[:, -1] - 1.0

    if method == "bootstrap":
        sharpes = np.array(
            [_sharpe_from_per_bar_returns(synthetic[i], ppy) for i in range(n_simulations)]
        )
    else:
        # For trade-level shuffles a per-bar Sharpe is meaningless; use a
        # per-trade Sharpe (mean/std of trade returns, unannualized). Still
        # comparable across simulations.
        means = synthetic.mean(axis=1)
        stds = synthetic.std(axis=1, ddof=1)
        with np.errstate(divide="ignore", invalid="ignore"):
            sharpes = np.where(stds > 0, means / stds, np.nan)

    max_dds = np.array([_max_drawdown_from_equity(equity[i]) for i in range(n_simulations)])

    # Percentile equity fan for the chart.
    pct_labels = ["p05", "p25", "p50", "p75", "p95"]
    pct_vals = np.percentile(equity, [5, 25, 50, 75, 95], axis=0)
    equity_ci = pd.DataFrame(pct_vals.T, columns=pct_labels)

    # Realized values for the dashboard's "you are here" overlay.
    realized_returns = result.returns.to_numpy(dtype=float)
    realized_equity = _equity_from_per_bar_returns(realized_returns)
    realized_total_return = float(realized_equity[-1] - 1.0) if realized_equity.size else 0.0
    realized_max_dd = _max_drawdown_from_equity(realized_equity)
    realized_sharpe = _sharpe_from_per_bar_returns(realized_returns, ppy)

    realized_equity_norm = pd.Series(realized_equity, index=result.returns.index, name="realized")

    # Edge probabilities + percentile ranks.
    prob_ruin = float((max_dds <= -ruin_threshold).mean())
    prob_goal = float((terminal >= goal_return).mean())
    return_pct = _percentile_rank(realized_total_return, terminal)
    sharpe_pct = _percentile_rank(realized_sharpe, sharpes)

    return MonteCarloResult(
        method=method,
        n_simulations=n_simulations,
        seed=seed,
        realized_total_return=realized_total_return,
        realized_sharpe=realized_sharpe,
        realized_max_drawdown=realized_max_dd,
        terminal_returns=terminal,
        sharpes=sharpes,
        max_drawdowns=max_dds,
        equity_curves_ci=equity_ci,
        realized_equity_normalized=realized_equity_norm,
        prob_ruin=prob_ruin,
        prob_goal=prob_goal,
        sharpe_percentile=sharpe_pct,
        return_percentile=return_pct,
        goal_return=goal_return,
        ruin_threshold=ruin_threshold,
    )
