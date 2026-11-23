"""Prop-firm challenge simulator.

Daily returns of a strategy (or a portfolio) over train + validation are
resampled by blocks of days (keeping losing streaks) into thousands of
possible futures. Each future runs the challenge phases with fresh balances:
a phase is passed when the profit target is reached with enough trading days,
and failed on the daily loss limit, the maximum drawdown (static from the
initial balance or trailing from the equity peak) or the time limit. Passing
accounts then run ``funded_days`` under the same loss rules.

Returns are measured in R per day and scaled by the risk per trade, so every
risk level of ``risk_grid`` is compared on the same futures. ``edge_haircut``
removes part of the backtest's mean daily return, because a selected backtest
is optimistic. Daily losses are realised at trade exit: floating intraday
losses are not modelled, so the real daily-loss risk is somewhat higher.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from fxgen.backtest.metrics import daily_r
from fxgen.config import ChallengeConfig, PropFirmConfig
from fxgen.data.splits import Period
from fxgen.errors import FxgenError

PASS, FAIL_DAILY, FAIL_DD, TIMEOUT = 0, 1, 2, 3


def trading_days(trades: pd.DataFrame, period: Period) -> pd.Series:
    """Business-day flag: at least one position opened that day (weekend entries roll to Monday)."""
    days = pd.bdate_range(period.start.normalize(), period.end.normalize(), inclusive="left", tz="UTC")
    if trades.empty:
        return pd.Series(False, index=days)
    entry = trades["entry_time"].dt.normalize()
    dow = entry.dt.day_of_week
    entry = entry + pd.to_timedelta(np.select([dow == 5, dow == 6], [2, 1], 0), unit="D")
    return pd.Series(days.isin(entry.unique()), index=days)


def daily_series(trades: pd.DataFrame, period: Period) -> tuple[pd.Series, pd.Series]:
    """(R per business day, traded flag) of a trade list."""
    return daily_r(trades, period), trading_days(trades, period)


def _bootstrap(n_days: int, n_sims: int, length: int, block: int, rng: np.random.Generator) -> np.ndarray:
    n_blocks = math.ceil(length / block)
    starts = rng.integers(0, n_days, size=(n_sims, n_blocks))
    idx = (starts[:, :, None] + np.arange(block)) % n_days
    return idx.reshape(n_sims, -1)[:, :length]


def _first(mask: np.ndarray) -> np.ndarray:
    """Index of the first True per row, or the row length when none."""
    any_true = mask.any(axis=1)
    return np.where(any_true, mask.argmax(axis=1), mask.shape[1])


def run_phase(pct: np.ndarray, traded: np.ndarray, target: float | None, daily_limit: float, dd_limit: float,
              trailing: bool, min_trading_days: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Outcome code, days used and final equity (%) of every simulated path of one phase."""
    equity = np.cumsum(pct, axis=1)
    if trailing:
        peak = np.maximum.accumulate(np.maximum(equity, 0.0), axis=1)
        dd_breach = equity <= peak - dd_limit
    else:
        dd_breach = equity <= -dd_limit
    day_breach = pct <= -daily_limit
    first_fail = _first(dd_breach | day_breach)
    horizon = pct.shape[1]
    if target is None:
        first_win = np.full(pct.shape[0], horizon)
    else:
        first_win = _first((equity >= target) & (np.cumsum(traded, axis=1) >= min_trading_days))
    outcome = np.full(pct.shape[0], TIMEOUT)
    won = first_win < first_fail
    outcome[won] = PASS
    failed = (~won) & (first_fail < horizon)
    rows = np.flatnonzero(failed)
    outcome[rows] = np.where(day_breach[rows, first_fail[rows]], FAIL_DAILY, FAIL_DD)
    days = np.minimum(np.minimum(first_win, first_fail) + 1, horizon)
    final = equity[np.arange(pct.shape[0]), days - 1]
    return outcome, days, final


def simulate_challenge(daily: pd.Series, traded: pd.Series, prop: PropFirmConfig, seed: int,
                       challenge: ChallengeConfig | None = None) -> dict[str, Any]:
    """Pass / fail probabilities of the challenge at every risk level of ``risk_grid``."""
    ch = challenge or prop.challenge
    r = daily.to_numpy(dtype=float)
    flags = traded.reindex(daily.index, fill_value=False).to_numpy(dtype=bool)
    if r.size < 20 or not flags.any():
        raise FxgenError("not enough trading history to simulate a challenge (need at least 20 days with trades)")
    mean = r.mean()
    if mean > 0:
        r = r - ch.edge_haircut * mean
    rng = np.random.default_rng(seed)
    horizons = [p.max_days or ch.no_limit_cap_days for p in ch.phases]
    paths = [_bootstrap(r.size, ch.n_sims, h, ch.block_days, rng) for h in horizons]
    funded_idx = _bootstrap(r.size, ch.n_sims, ch.funded_days, ch.block_days, rng) if ch.funded_days else None
    trailing = ch.drawdown_mode == "trailing"

    rows: list[dict[str, Any]] = []
    for risk in ch.risk_grid:
        alive = np.ones(ch.n_sims, dtype=bool)
        total_days = np.zeros(ch.n_sims)
        row: dict[str, Any] = {"risk_per_trade_pct": risk}
        fail_daily = fail_dd = timeout = np.zeros(ch.n_sims, dtype=bool)
        for phase, idx in zip(ch.phases, paths):
            outcome, days, _ = run_phase(r[idx] * risk, flags[idx], phase.profit_target_pct, prop.max_daily_loss_pct,
                                         prop.max_total_drawdown_pct, trailing, phase.min_trading_days)
            fail_daily = fail_daily | (alive & (outcome == FAIL_DAILY))
            fail_dd = fail_dd | (alive & (outcome == FAIL_DD))
            timeout = timeout | (alive & (outcome == TIMEOUT))
            alive &= outcome == PASS
            total_days += days
            row[f"p_pass_{phase.name}"] = float(alive.mean())
        p_pass = float(alive.mean())
        row.update({
            "p_pass_challenge": p_pass,
            "p_fail_daily_loss": float(fail_daily.mean()),
            "p_fail_drawdown": float(fail_dd.mean()),
            "p_timeout": float(timeout.mean()),
            "median_days_to_pass": float(np.median(total_days[alive])) if alive.any() else None,
            "expected_attempts": 1 / p_pass if p_pass > 0 else None,
            "expected_fee_cost": ch.fee / p_pass if p_pass > 0 and ch.fee else None,
        })
        if funded_idx is not None:
            outcome, _, final = run_phase(r[funded_idx] * risk, flags[funded_idx], None, prop.max_daily_loss_pct,
                                          prop.max_total_drawdown_pct, trailing, 0)
            survive = outcome == TIMEOUT
            row["p_keep_funded"] = float(survive.mean())
            row["p_pass_and_keep"] = p_pass * float(survive.mean())
            row["median_funded_return_pct"] = float(np.median(final[survive])) if survive.any() else None
        rows.append(row)

    best = max(rows, key=lambda x: (x["p_pass_challenge"], -x["risk_per_trade_pct"]))
    return {
        "rules": {"phases": [p.model_dump() for p in ch.phases], "max_daily_loss_pct": prop.max_daily_loss_pct,
                  "max_total_drawdown_pct": prop.max_total_drawdown_pct, "drawdown_mode": ch.drawdown_mode,
                  "funded_days": ch.funded_days},
        "history_days": int(r.size),
        "edge_haircut": ch.edge_haircut,
        "mean_daily_r_after_haircut": float(r.mean()),
        "n_sims": ch.n_sims,
        "by_risk": rows,
        "best_risk_per_trade_pct": best["risk_per_trade_pct"],
        "best_p_pass": best["p_pass_challenge"],
    }
