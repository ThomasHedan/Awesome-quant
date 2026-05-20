"""Benchmark comparison: strategy vs. buy-and-hold (or any reference series).

Default benchmark is buy-and-hold on the same instrument the strategy was
backtested on — asset-agnostic and always available. Pass an explicit
returns series to compare against an external reference (SPY for an equity
strategy, BTC for a crypto strategy, etc.).

All metrics are computed against the inner-joined index, so misaligned
benchmarks (different calendar, missing days) don't silently corrupt the
regression.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from quant_dashboard.data import OHLCV
from quant_dashboard.engine.runner import BacktestResult, periods_per_year


class BenchmarkError(RuntimeError):
    """User-actionable benchmark comparison errors."""


@dataclass
class BenchmarkComparison:
    """Side-by-side analysis vs. a benchmark returns series."""

    benchmark_name: str
    strategy_returns: pd.Series
    benchmark_returns: pd.Series

    # Annualized
    alpha: float
    beta: float
    information_ratio: float

    correlation: float
    r_squared: float

    # Totals over the joined window
    strategy_total_return: float
    benchmark_total_return: float
    excess_total_return: float

    excess_returns: pd.Series
    relative_equity: pd.Series  # strategy_equity / benchmark_equity, base 1.0

    def to_dict(self) -> dict:
        return {
            "benchmark_name": self.benchmark_name,
            "alpha": self.alpha,
            "beta": self.beta,
            "information_ratio": self.information_ratio,
            "correlation": self.correlation,
            "r_squared": self.r_squared,
            "strategy_total_return": self.strategy_total_return,
            "benchmark_total_return": self.benchmark_total_return,
            "excess_total_return": self.excess_total_return,
        }


# ---------------------------------------------------------------------------


def build_buy_and_hold_returns(data: OHLCV) -> pd.Series:
    """Per-bar buy-and-hold returns of the instrument the strategy traded.

    Computed from close-to-close so it lines up exactly with the engine's
    returns series. tz-naive to match what the engine emits.
    """
    close = data.df["close"]
    rets = close.pct_change().fillna(0.0)
    if hasattr(rets.index, "tz") and rets.index.tz is not None:
        rets = rets.tz_localize(None)
    rets.name = f"bh_{data.spec.symbol}"
    return rets


def _align(a: pd.Series, b: pd.Series) -> tuple[pd.Series, pd.Series]:
    """Inner-join two return series on their indices."""
    df = pd.concat([a.rename("a"), b.rename("b")], axis=1, join="inner")
    df = df.dropna(how="any")
    if len(df) < 3:
        raise BenchmarkError(
            "fewer than 3 overlapping observations between strategy and "
            "benchmark - check date ranges and calendars."
        )
    return df["a"], df["b"]


def _ols_alpha_beta(
    strat: np.ndarray,
    bench: np.ndarray,
    rf_per_period: float = 0.0,
) -> tuple[float, float, float]:
    """OLS regression of strategy excess returns on benchmark excess returns.

    Returns (alpha_per_period, beta, r_squared). Alpha is per-period; the
    caller annualizes.
    """
    y = strat - rf_per_period
    x = bench - rf_per_period
    x_mean = x.mean()
    y_mean = y.mean()
    var_x = ((x - x_mean) ** 2).mean()
    if var_x <= 0:
        raise BenchmarkError("benchmark has zero variance - cannot regress")
    cov_xy = ((x - x_mean) * (y - y_mean)).mean()
    beta = cov_xy / var_x
    alpha = y_mean - beta * x_mean
    # R^2 = 1 - SSE/SST; use corrcoef for numerical stability.
    if y.std() == 0:
        r2 = 0.0
    else:
        corr = float(np.corrcoef(x, y)[0, 1])
        r2 = corr * corr
    return float(alpha), float(beta), float(r2)


def compute_benchmark_comparison(
    result: BacktestResult,
    benchmark_returns: pd.Series | None = None,
    *,
    benchmark_name: str | None = None,
    risk_free_annual: float = 0.0,
) -> BenchmarkComparison:
    """Compare ``result`` to ``benchmark_returns`` (or buy-and-hold).

    Parameters
    ----------
    benchmark_returns:
        Per-period returns aligned to a timestamp index. If ``None``, uses
        buy-and-hold on the same instrument.
    risk_free_annual:
        Annualized risk-free rate (e.g. 0.04 for 4%). Subtracted per-period
        from both legs before the OLS to compute Jensen's alpha.
    """
    strat = result.returns
    if benchmark_returns is None:
        bench = build_buy_and_hold_returns(result.data)
        bench_name = benchmark_name or f"B&H {result.data.spec.symbol}"
    else:
        bench = benchmark_returns.copy()
        if hasattr(bench.index, "tz") and bench.index.tz is not None:
            bench = bench.tz_localize(None)
        bench_name = benchmark_name or "benchmark"

    strat_a, bench_a = _align(strat, bench)

    ppy = periods_per_year(result.data.df.index)
    rf_per_period = (1.0 + risk_free_annual) ** (1.0 / ppy) - 1.0

    alpha_p, beta, r2 = _ols_alpha_beta(strat_a.to_numpy(), bench_a.to_numpy(), rf_per_period)
    alpha_annual = (1.0 + alpha_p) ** ppy - 1.0

    excess = strat_a - bench_a
    excess_std = float(excess.std(ddof=1))
    if excess_std == 0:
        info_ratio = float("nan")
    else:
        info_ratio = float(excess.mean() / excess_std) * float(np.sqrt(ppy))

    corr = float(strat_a.corr(bench_a))

    strat_total = float((1.0 + strat_a).prod() - 1.0)
    bench_total = float((1.0 + bench_a).prod() - 1.0)

    rel_eq = (1.0 + strat_a).cumprod() / (1.0 + bench_a).cumprod()
    rel_eq.name = "relative_equity"

    return BenchmarkComparison(
        benchmark_name=bench_name,
        strategy_returns=strat_a,
        benchmark_returns=bench_a,
        alpha=alpha_annual,
        beta=beta,
        information_ratio=info_ratio,
        correlation=corr,
        r_squared=r2,
        strategy_total_return=strat_total,
        benchmark_total_return=bench_total,
        excess_total_return=strat_total - bench_total,
        excess_returns=excess,
        relative_equity=rel_eq,
    )
