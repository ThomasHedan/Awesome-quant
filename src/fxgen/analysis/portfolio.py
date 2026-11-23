"""Aggregator: a portfolio of the best strategies that are not correlated with each other.

Selection only uses train and validation results stored in the database:

1. pool = candidate trees of the chosen runs (by default only those that
   survived the robustness battery) with a validation Sharpe above a floor;
2. sort by the chosen score (validation DSR by default);
3. greedily add a tree when the |correlation| of its daily R with every member
   is below ``max_correlation`` and its strategy type is under ``max_per_type``.

Each member trades at ``risk_per_trade_pct x weight``. Portfolio metrics, a
Monte Carlo on daily returns and the prop-firm risk budget follow. Holdout
figures are shown only when every member already has stored holdout results:
building a portfolio never reads the holdout.
"""

from __future__ import annotations

import json
import math
import uuid
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from fxgen.backtest.metrics import TRADING_DAYS, daily_r
from fxgen.config import Config, PortfolioConfig
from fxgen.data.splits import Period, split_period
from fxgen.errors import FxgenError
from fxgen.storage.db import Store

SPLITS = ("train", "validation", "holdout")


def _config(store: Store, run_id: str) -> Config:
    return Config.model_validate(json.loads(store.get_run(run_id)["config_json"]))


def _pool(store: Store, run_ids: list[str], settings: PortfolioConfig) -> pd.DataFrame:
    placeholders = ",".join("?" for _ in run_ids)
    rows = store.query(
        f"SELECT rt.run_id, rt.tree_id, rt.train_fitness, rt.survives, rt.strategy_type, t.rules, t.n_internal, "
        f"m.sharpe AS validation_sharpe, m.dsr AS validation_dsr, r.created_at "
        f"FROM run_trees rt JOIN trees t USING (tree_id) JOIN runs r USING (run_id) "
        f"JOIN metrics m ON m.run_id = rt.run_id AND m.tree_id = rt.tree_id AND m.split = 'validation' AND m.pair = 'ALL' "
        f"WHERE rt.is_candidate = 1 AND rt.run_id IN ({placeholders})",
        tuple(run_ids),
    )
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    if settings.survivors_only:
        df = df[df["survives"] == 1]
    df = df[df["validation_sharpe"].fillna(-np.inf) > settings.min_validation_sharpe]
    return df.sort_values("created_at", ascending=False).drop_duplicates("tree_id")


def _series(store: Store, run_id: str, tree_id: str, periods: dict[str, Period]) -> dict[str, pd.Series]:
    trades = store.get_trades(run_id, tree_id)
    return {split: daily_r(trades[trades["split"] == split], period) for split, period in periods.items()
            if split != "holdout" or (trades["split"] == "holdout").any()}


def _stats(daily_pct: pd.Series) -> dict[str, float]:
    if daily_pct.empty:
        return {}
    equity = daily_pct.cumsum()
    dd = float((equity.cummax().clip(lower=0) - equity).max())
    std = float(daily_pct.std(ddof=1))
    return {
        "sharpe": float(daily_pct.mean() / std * math.sqrt(TRADING_DAYS)) if std > 0 else 0.0,
        "return_pct": float(daily_pct.sum()),
        "annual_return_pct": float(daily_pct.mean() * TRADING_DAYS),
        "max_dd_pct": dd,
        "worst_day_pct": float(min(0.0, daily_pct.min())),
        "positive_months_share": float((daily_pct.groupby([daily_pct.index.year, daily_pct.index.month]).sum() > 0).mean()),
    }


def _monte_carlo(daily_pct: pd.Series, n_sims: int, confidence: float, seed: int) -> tuple[float, float]:
    """(max drawdown %, at ``confidence``) over shuffled days, and its median."""
    values = daily_pct.to_numpy()
    if values.size == 0:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    paths = np.cumsum(np.stack([rng.permutation(values) for _ in range(n_sims)]), axis=1)
    paths = np.hstack([np.zeros((n_sims, 1)), paths])
    dd = np.max(np.maximum.accumulate(paths, axis=1) - paths, axis=1)
    return float(np.quantile(dd, confidence)), float(np.median(dd))


def build_portfolio(store: Store, run_ids: list[str] | None = None, settings: PortfolioConfig | None = None,
                    save: bool = True) -> dict[str, Any]:
    """Select decorrelated strategies and evaluate the resulting portfolio."""
    runs = run_ids or [r["run_id"] for r in store.query("SELECT run_id FROM runs WHERE status = 'done'")]
    if not runs:
        raise FxgenError("no finished run to build a portfolio from")
    configs = {rid: _config(store, rid) for rid in runs}
    base = configs[runs[0]]
    settings = settings or base.portfolio
    if any(c.splits != base.splits for c in configs.values()):
        raise FxgenError("the selected runs use different splits; build one portfolio per split definition")
    periods = {s: split_period(base.splits, s, allow_holdout=True) for s in SPLITS}  # type: ignore[arg-type]

    pool = _pool(store, runs, settings)
    if pool.empty:
        raise FxgenError("no eligible strategy: no survivor with a positive validation Sharpe in these runs "
                         "(set survivors_only=false or lower min_validation_sharpe to widen the pool)")
    score_col = {"validation_dsr": "validation_dsr", "validation_sharpe": "validation_sharpe",
                 "train_fitness": "train_fitness"}[settings.select_by]
    pool = pool.sort_values([score_col, "validation_sharpe"], ascending=False, na_position="last", kind="stable")

    selected: list[dict[str, Any]] = []
    selected_series: list[pd.Series] = []
    rejected: list[dict[str, Any]] = []
    type_count: dict[str, int] = {}
    for row in pool.itertuples():
        series = _series(store, row.run_id, row.tree_id, periods)
        insample = pd.concat([series["train"], series["validation"]])
        stype = row.strategy_type or "unclassified"
        if settings.max_per_type is not None and type_count.get(stype, 0) >= settings.max_per_type:
            rejected.append({"tree_id": row.tree_id, "reason": f"type quota reached ({stype})"})
            continue
        corrs = [float(insample.corr(other)) for other in selected_series]
        worst = max((abs(c) for c in corrs if not math.isnan(c)), default=0.0)
        if worst > settings.max_correlation:
            rejected.append({"tree_id": row.tree_id, "reason": f"correlation {worst:.2f} > {settings.max_correlation}"})
            continue
        selected.append({"run_id": row.run_id, "tree_id": row.tree_id, "strategy_type": stype,
                         "validation_sharpe": row.validation_sharpe, "validation_dsr": row.validation_dsr,
                         "train_fitness": row.train_fitness, "rules": row.rules, "series": series})
        selected_series.append(insample)
        type_count[stype] = type_count.get(stype, 0) + 1
        if len(selected) == settings.max_strategies:
            break

    n = len(selected)
    if settings.weighting == "inverse_vol":
        vols = np.array([s.std(ddof=1) or 1.0 for s in selected_series])
        weights = (1 / vols) / (1 / vols).sum() * n
    else:
        weights = np.ones(n)
    risk = base.backtest.risk_per_trade_pct
    members = []
    for m, w in zip(selected, weights):
        members.append({k: v for k, v in m.items() if k != "series"} | {"weight": float(w), "risk_per_trade_pct": float(risk * w)})

    result: dict[str, Any] = {"splits": {}, "member_splits": {}}
    for split in SPLITS:
        if not all(split in m["series"] for m in selected):
            continue
        frame = pd.DataFrame({m["tree_id"]: m["series"][split] for m in selected}).fillna(0.0)
        pct = frame.mul(weights * risk, axis=1)
        portfolio = pct.sum(axis=1)
        stats = _stats(portfolio)
        member_sharpes = [_stats(pct[c])["sharpe"] for c in pct.columns]
        stats["mean_member_sharpe"] = float(np.mean(member_sharpes))
        stats["diversification_ratio"] = stats["sharpe"] / stats["mean_member_sharpe"] if stats["mean_member_sharpe"] > 0 else float("nan")
        result["splits"][split] = stats
        result["member_splits"][split] = {c: _stats(pct[c]) for c in pct.columns}

    insample_frame = pd.DataFrame({m["tree_id"]: pd.concat([m["series"]["train"], m["series"]["validation"]]) for m in selected}).fillna(0.0)
    corr = insample_frame.corr()
    off = corr.to_numpy()[~np.eye(n, dtype=bool)] if n > 1 else np.array([])
    result["correlation"] = {a: {b: float(corr.loc[a, b]) for b in corr.columns} for a in corr.index}
    result["mean_abs_correlation"] = float(np.nanmean(np.abs(off))) if off.size else 0.0
    daily_pct = insample_frame.mul(weights * risk, axis=1).sum(axis=1)
    dd95, dd50 = _monte_carlo(daily_pct, base.validation.montecarlo.n_sims, base.validation.montecarlo.confidence, base.seed)
    worst_day = float(min(0.0, daily_pct.min()))
    prop = base.prop_firm
    scale_dd = prop.max_total_drawdown_pct / dd95 if dd95 > 0 else float("inf")
    scale_day = prop.max_daily_loss_pct / -worst_day if worst_day < 0 else float("inf")
    scale = min(scale_dd, scale_day)
    result["prop_firm"] = {
        "max_dd_pct_at_conf": dd95, "max_dd_pct_median": dd50, "worst_day_pct": worst_day,
        "risk_scale_limit": scale,
        "max_risk_per_trade_pct_per_member": {m["tree_id"]: m["risk_per_trade_pct"] * scale for m in members},
        "note": "drawdown from shuffled daily returns of train + validation at the configured risk per trade",
    }
    result["types"] = type_count
    result["holdout_available"] = "holdout" in result["splits"]
    result["rejected"] = rejected[:50]
    result["pool_size"] = int(len(pool))

    portfolio_id = f"pf_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:4]}"
    out = {"portfolio_id": portfolio_id, "runs": runs, "settings": settings.model_dump(), "members": members, "result": result}
    out["_series"] = {m["tree_id"]: {s: v for s, v in m["series"].items()} for m in selected}
    out["_weights"] = weights
    if save:
        store.save_portfolio(portfolio_id, {"runs": runs, **settings.model_dump()}, members, result)
    return out
