"""Deflated Sharpe Ratio (Bailey & Lopez de Prado, 2014).

The expected maximum Sharpe among ``N`` trials with Sharpe variance ``V`` is

    SR0 = sqrt(V) * ((1 - g) * Z(1 - 1/N) + g * Z(1 - 1/(N e)))

with ``g`` the Euler-Mascheroni constant and ``Z`` the standard normal
quantile. The DSR is the probability that the observed Sharpe exceeds SR0,
adjusted for sample length, skewness and kurtosis of the returns:

    DSR = Phi((SR - SR0) * sqrt(T - 1) / sqrt(1 - skew * SR + (kurt - 1) / 4 * SR^2))

All Sharpe ratios inside the formula are per observation (daily), not annualised.

``V`` is the variance of the Sharpe estimator under the null of no skill,
``1 / (T - 1)``. The empirical cross-trial variance of a GA population is
dominated by deterministic cost-driven losers, not by estimation noise, and
would inflate SR0 to meaningless levels.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from statistics import NormalDist

import numpy as np
import pandas as pd

from fxgen.backtest.metrics import TRADING_DAYS

EULER_GAMMA = 0.5772156649015329
_N = NormalDist()


@dataclass(frozen=True)
class DSRResult:
    dsr: float
    sharpe_annual: float
    sr0_annual: float
    n_trials: int
    n_obs: int
    skew: float
    kurtosis: float

    def to_dict(self) -> dict[str, float | int]:
        return asdict(self)


def expected_max_sharpe(n_trials: int, variance: float) -> float:
    """Expected maximum of ``n_trials`` Sharpe estimates with the given variance (same units)."""
    if n_trials <= 1 or variance <= 0:
        return 0.0
    a = _N.inv_cdf(1 - 1 / n_trials)
    b = _N.inv_cdf(1 - 1 / (n_trials * math.e))
    return math.sqrt(variance) * ((1 - EULER_GAMMA) * a + EULER_GAMMA * b)


def deflated_sharpe(returns: pd.Series | np.ndarray, n_trials: int) -> DSRResult:
    """DSR of a daily return series selected among ``n_trials`` trials."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    t = r.size
    trial_var = 1.0 / (t - 1) if t > 1 else 0.0
    sd = r.std(ddof=1) if t > 1 else 0.0
    if t < 3 or sd == 0:
        return DSRResult(0.0, 0.0, expected_max_sharpe(n_trials, trial_var) * math.sqrt(TRADING_DAYS), n_trials, t, 0.0, 3.0)
    sr = r.mean() / sd
    z = (r - r.mean()) / r.std(ddof=0)
    skew = float(np.mean(z**3))
    kurt = float(np.mean(z**4))
    sr0 = expected_max_sharpe(n_trials, trial_var)
    denom = 1 - skew * sr + (kurt - 1) / 4 * sr**2
    stat = (sr - sr0) * math.sqrt(t - 1) / math.sqrt(max(denom, 1e-12))
    return DSRResult(
        dsr=float(_N.cdf(stat)),
        sharpe_annual=float(sr * math.sqrt(TRADING_DAYS)),
        sr0_annual=float(sr0 * math.sqrt(TRADING_DAYS)),
        n_trials=int(n_trials),
        n_obs=int(t),
        skew=skew,
        kurtosis=kurt,
    )
