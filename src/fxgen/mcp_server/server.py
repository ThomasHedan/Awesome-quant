"""fxgen MCP server (FastMCP, stdio transport).

Read tools query the SQLite database; long runs are launched as background
jobs. Holdout rules: no tool can optimise on the holdout or move the splits;
``backtest_tree`` on the holdout is recorded and increases the trial count used
by the deflated Sharpe ratio.

Configuration: ``FXGEN_CONFIG`` (path to the YAML config, default
``config/default.yaml``) selects the data, the database and the output folder.
"""

from __future__ import annotations

import functools
import json
import logging
import math
import os
from pathlib import Path
from typing import Annotated, Any, Literal

import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import Field

from fxgen import jobs, pipeline
from fxgen.config import DEFAULT_CONFIG_PATH, Config, load_config
from fxgen.data.loader import discover_datasets
from fxgen.errors import ConfigError, FxgenError, HoldoutLockedError, TreeError
from fxgen.logging_setup import setup_logging
from fxgen.reporting.ranking import ranking_frame
from fxgen.storage.db import Store
from fxgen.tree.node import depth, n_internal, n_nodes
from fxgen.tree.serialize import from_dict

logger = logging.getLogger(__name__)

Stage = Literal["train", "validation", "holdout"]
Split = Literal["train", "validation", "holdout"]
TreeId = Annotated[str, Field(min_length=4, max_length=32, pattern=r"^[0-9a-f]+$", description="tree id (hash)")]
RunId = Annotated[str | None, Field(default=None, description="run id; latest finished run when omitted")]
LOCKED_OVERRIDES = ("splits", "storage")
SUMMARY_KEYS = ("n_trades", "trades_per_year", "sharpe", "profit_factor", "win_rate", "expectancy_r", "total_r",
                "return_pct", "max_dd_pct", "worst_day_pct", "max_concurrent", "avg_bars_held")

mcp = FastMCP(
    "fxgen",
    instructions=(
        "Genetic decision-tree generator for FX swing trading. Use get_rankings to browse results, "
        "get_tree / get_metrics / get_robustness to inspect a tree, run_evolution for new runs. "
        "The holdout is locked: holdout backtests are logged and penalise the deflated Sharpe."
    ),
)


def config_path() -> Path:
    return Path(os.environ.get("FXGEN_CONFIG", str(DEFAULT_CONFIG_PATH)))


def base_config() -> Config:
    return load_config(config_path())


def store() -> Store:
    cfg = base_config()
    return Store(cfg.resolve(cfg.storage.database))


def _round(value: Any, digits: int = 3) -> Any:
    if isinstance(value, float):
        return None if not math.isfinite(value) else round(value, digits)
    if isinstance(value, dict):
        return {k: _round(v, digits) for k, v in value.items()}
    if isinstance(value, list):
        return [_round(v, digits) for v in value]
    return value


def _summary(metrics: dict[str, Any]) -> dict[str, Any]:
    return _round({k: metrics.get(k) for k in SUMMARY_KEYS})


def _run(st: Store, run_id: str | None) -> str:
    run_id = run_id or st.latest_run()
    if run_id is None:
        raise FxgenError("no finished run yet; start one with run_evolution")
    st.get_run(run_id)
    return run_id


def _tree_run(st: Store, tree_id: str, run_id: str | None) -> str:
    st.get_tree(tree_id)
    rid = run_id or st.run_of_tree(tree_id)
    if rid is None:
        raise FxgenError(f"tree {tree_id} is not attached to any run")
    return rid


def _errors(fn):  # type: ignore[no-untyped-def]
    """Turn domain errors into clear tool errors (the MCP client sees the message)."""
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            return fn(*args, **kwargs)
        except FxgenError as exc:
            raise ValueError(f"{type(exc).__name__}: {exc}") from exc

    return wrapper


# read tools -----------------------------------------------------------------


@mcp.tool()
@_errors
def list_datasets() -> dict[str, Any]:
    """Pairs, timeframes, date ranges and data quality of the CSV files in the data directory."""
    cfg = base_config()
    rows = discover_datasets(cfg)
    return {
        "directory": str(cfg.resolve(cfg.data.directory)),
        "configured_pairs": cfg.data.pairs,
        "execution_timeframe": cfg.data.execution_timeframe,
        "splits": cfg.splits.model_dump(),
        "files": [{k: r.get(k) for k in ("pair", "timeframe", "start", "end", "n_bars", "n_gaps", "max_gap_hours",
                                          "duplicates_removed", "ohlc_repaired", "error") if k in r} for r in rows],
    }


@mcp.tool()
@_errors
def list_conditions(
    family: Literal["trend", "momentum", "volatility", "structure", "volume", "time"] | None = None,
    run_id: RunId = None,
    limit: Annotated[int, Field(ge=1, le=500)] = 100,
    offset: Annotated[int, Field(ge=0)] = 0,
) -> dict[str, Any]:
    """Binary conditions kept after redundancy filtering (thresholds fitted on train), paginated."""
    st = store()
    rid = _run(st, run_id)
    rows, total = st.get_conditions(rid, family, limit, offset)
    return {"run_id": rid, "total": total, "offset": offset, "conditions": _round(rows)}


@mcp.tool()
@_errors
def get_rankings(
    stage: Stage = "holdout",
    top_n: Annotated[int, Field(ge=1, le=200)] = 20,
    sort_by: Literal["dsr", "sharpe", "fitness", "profit_factor"] = "dsr",
    run_id: RunId = None,
    survivors_only: bool = False,
) -> dict[str, Any]:
    """Ranking of a run's candidate trees on one stage. Holdout values exist only for finalists."""
    st = store()
    rid = _run(st, run_id)
    column = "train_fitness" if sort_by == "fitness" else f"{stage}_{sort_by}"
    df = ranking_frame(st.ranking_rows(rid), sort_by=column)
    if stage == "holdout":
        df = df[df["holdout_sharpe"].notna()]
    if survivors_only:
        df = df[df["survives"] == True]  # noqa: E712
    cols = ["tree_id", "survives", "finalist", "strategy_type", "n_internal", "train_fitness", f"{stage}_sharpe", f"{stage}_dsr",
            f"{stage}_profit_factor", f"{stage}_max_dd_pct", "wf_positive_share", "sensitivity_worst_ratio", "mc_max_risk_pct"]
    cols = list(dict.fromkeys(c for c in cols if c in df.columns))
    rows = df[cols].head(top_n).astype(object).where(df[cols].head(top_n).notna(), None).to_dict("records")
    run = st.get_run(rid)
    note = None
    if stage == "holdout" and not run["holdout_done"]:
        note = "holdout stage not executed for this run yet (CLI: fxgen holdout RUN_ID)"
    return {"run_id": rid, "stage": stage, "sorted_by": column, "total_trials": st.total_trials(),
            "rows": _round(rows), "note": note}


@mcp.tool()
@_errors
def get_tree(tree_id: TreeId) -> dict[str, Any]:
    """Readable rules, JSON, depth and node counts of a tree."""
    row = store().get_tree(tree_id)
    tree = from_dict(row["tree_json"])
    return {"tree_id": tree_id, "rules": row["rules"], "tree_json": row["tree_json"], "depth": depth(tree),
            "internal_nodes": n_internal(tree), "total_nodes": n_nodes(tree)}


@mcp.tool()
@_errors
def get_metrics(
    tree_id: TreeId,
    split: Annotated[str | None, Field(default=None, pattern=r"^(manual_)?(train|validation|holdout)$")] = None,
    pair: Annotated[str | None, Field(default=None, pattern=r"^[A-Z]{6}$")] = None,
    run_id: RunId = None,
) -> dict[str, Any]:
    """Metrics of a tree by split (portfolio and per pair) and per year."""
    st = store()
    rid = _tree_run(st, tree_id, run_id)
    rows = st.get_metrics(rid, tree_id, split, pair)
    out: dict[str, Any] = {"run_id": rid, "tree_id": tree_id, "splits": {}}
    for r in rows:
        entry = out["splits"].setdefault(r["split"], {"portfolio": None, "pairs": {}})
        if r["pair"] == "ALL":
            entry["portfolio"] = {**_summary(r["metrics"]), "dsr": _round(r["dsr"])}
        else:
            entry["pairs"][r["pair"]] = _round({k: r["metrics"].get(k) for k in ("n_trades", "sharpe", "total_r", "win_rate")})
    out["per_year"] = _round(st.get_yearly(rid, tree_id, split))
    return out


@mcp.tool()
@_errors
def get_trades(
    tree_id: TreeId,
    pair: Annotated[str | None, Field(default=None, pattern=r"^[A-Z]{6}$")] = None,
    split: Annotated[str | None, Field(default=None, pattern=r"^(manual_)?(train|validation|holdout)$")] = None,
    limit: Annotated[int, Field(ge=1, le=1000)] = 100,
    offset: Annotated[int, Field(ge=0)] = 0,
    run_id: RunId = None,
) -> dict[str, Any]:
    """Trades of a tree (paginated, oldest first)."""
    st = store()
    rid = _tree_run(st, tree_id, run_id)
    df = st.get_trades(rid, tree_id, split, pair, limit, offset)
    cols = ["split", "pair", "entry_time", "exit_time", "direction", "bars_held", "net_pips", "r_multiple", "exit_reason", "branch"]
    df = df[cols].copy()
    for c in ("entry_time", "exit_time"):
        df[c] = df[c].dt.strftime("%Y-%m-%d %H:%M")
    return {"run_id": rid, "total": st.count_trades(rid, tree_id, split, pair), "offset": offset,
            "trades": _round(df.to_dict("records"))}


@mcp.tool()
@_errors
def get_robustness(tree_id: TreeId, run_id: RunId = None) -> dict[str, Any]:
    """Walk-forward, permutation, deflated Sharpe, sensitivity and Monte Carlo results of a tree."""
    st = store()
    rid = _tree_run(st, tree_id, run_id)
    report = st.get_robustness(rid, tree_id)
    if report is None:
        return {"run_id": rid, "tree_id": tree_id, "report": None, "note": "no robustness report; call run_robustness"}
    perm = report.get("permutation") or {}
    return {
        "run_id": rid, "tree_id": tree_id, "survives": report["survives"], "checks": report["checks"],
        "dsr": _round(report["dsr"]),
        "walkforward": _round(report["walkforward"]),
        "sensitivity": _round({k: report["sensitivity"][k] for k in ("base_sharpe", "worst_sharpe", "median_ratio", "worst_ratio", "passed")}),
        "montecarlo": _round(report["montecarlo"]),
        "prop_firm": _round(report.get("prop_firm") or {}),
        "permutation": _round({k: perm.get(k) for k in ("real_best", "permuted_best", "p_value", "tree_p_value", "z_score")}) if perm else None,
        "run_walkforward_process": _round(_json(st.get_run(rid).get("walkforward_json"))),
    }


def _json(text: str | None) -> Any:
    return json.loads(text) if text else None


@mcp.tool()
@_errors
def compare_trees(
    tree_ids: Annotated[list[TreeId], Field(min_length=2, max_length=10)],
    split: Split = "validation",
    run_id: RunId = None,
) -> dict[str, Any]:
    """Side-by-side metrics and correlation of daily equity changes (to build decorrelated portfolios)."""
    st = store()
    rid = _run(st, run_id) if run_id else _tree_run(st, tree_ids[0], None)
    table = []
    for tid in tree_ids:
        rows = st.get_metrics(rid, tid, split, "ALL")
        if not rows:
            raise TreeError(f"tree {tid} has no stored {split} results in run {rid}")
        table.append({"tree_id": tid, **_summary(rows[0]["metrics"]), "dsr": _round(rows[0]["dsr"])})
    corr = pipeline.equity_correlation(st, rid, list(tree_ids), split)
    return {"run_id": rid, "split": split, "metrics": table,
            "daily_r_correlation": _round(corr.where(pd.notna(corr), None).to_dict())}


# action tools ---------------------------------------------------------------


@mcp.tool()
@_errors
def run_evolution(config_overrides: dict[str, Any] | None = None) -> dict[str, Any]:
    """Start a GA run + validation in the background. Overrides are nested YAML keys,
    e.g. {"ga": {"generations": 20}, "tree": {"max_depth": 4}}. Splits and storage are locked."""
    overrides = config_overrides or {}
    locked = [k for k in overrides if k in LOCKED_OVERRIDES]
    if locked:
        raise HoldoutLockedError(f"overriding {locked} is not allowed from MCP (it could expose the holdout)")
    try:
        load_config(config_path(), overrides)
    except Exception as exc:
        raise ConfigError(f"invalid config_overrides: {exc}") from exc
    cfg = base_config()
    st = store()
    job_id = jobs.launch_evolution(st, config_path(), overrides, cfg.resolve(cfg.storage.output_dir) / "jobs")
    return {"job_id": job_id, "status": "running", "hint": "poll get_job_status(job_id)"}


@mcp.tool()
@_errors
def get_job_status(job_id: Annotated[str, Field(pattern=r"^job_[0-9a-f]+$")]) -> dict[str, Any]:
    """Status, progress and run id of a background job."""
    job = jobs.job_status(store(), job_id)
    return {k: job.get(k) for k in ("job_id", "kind", "status", "run_id", "progress", "error", "created_at", "updated_at")}


@mcp.tool()
@_errors
def cancel_job(job_id: Annotated[str, Field(pattern=r"^job_[0-9a-f]+$")]) -> dict[str, Any]:
    """Request cancellation; the worker stops at the next generation or stage boundary."""
    job = jobs.cancel(store(), job_id)
    return {"job_id": job_id, "status": job["status"]}


@mcp.tool()
@_errors
def backtest_tree(
    tree_json: dict[str, Any],
    pairs: Annotated[list[Annotated[str, Field(pattern=r"^[A-Z]{6}$")]] | None, Field(default=None, max_length=30)] = None,
    timeframe: Literal["H1", "H4", "D1"] = "H1",
    split: Split = "validation",
    run_id: RunId = None,
) -> dict[str, Any]:
    """Backtest a hand-written or modified tree, e.g. {"if": "D1.dist_ema200>0", "then": {"leaf": "LONG"},
    "else": {"leaf": "SHORT"}}. Condition ids: 'H4.rsi14>p75', 'H4.rsi14>70', 'D1.ema20_50:cross_up',
    'H1.macd_hist:rising', 'H1.cal:session_london'. Every call counts as a trial for the deflated Sharpe;
    holdout calls are recorded as holdout accesses."""
    st = store()
    tree = from_dict(tree_json)
    result = pipeline.manual_backtest(st, tree, split, pairs, timeframe, run_id)
    out = {
        "tree_id": result["tree_id"], "run_id": result["run_id"], "split": split,
        "rules": st.get_tree(result["tree_id"])["rules"],
        "portfolio": _summary(result["metrics"]), "dsr": _round(result["dsr"]),
        "per_pair": {p: _round({k: m.get(k) for k in ("n_trades", "sharpe", "total_r")}) for p, m in result["per_pair"].items()},
        "n_trials_total": result["n_trials_total"],
    }
    if split == "holdout":
        out["warning"] = (f"holdout accessed and logged ({result['holdout_accesses_total']} holdout accesses in total). "
                          "Every look at the holdout weakens it as an unbiased test.")
    return out


@mcp.tool()
@_errors
def run_robustness(tree_id: TreeId, run_id: RunId = None) -> dict[str, Any]:
    """Re-run walk-forward, deflated Sharpe, sensitivity and Monte Carlo on a stored tree (no holdout)."""
    st = store()
    rid = _tree_run(st, tree_id, run_id)
    report = pipeline.rerun_robustness(st, tree_id, rid)
    return {"run_id": rid, "tree_id": tree_id, "survives": report["survives"], "checks": report["checks"],
            "dsr": _round(report["dsr"])}


@mcp.tool()
@_errors
def export_pine(tree_id: TreeId, run_id: RunId = None) -> dict[str, Any]:
    """Pine Script v6 indicator showing the tree's signals and active branch."""
    st = store()
    rid = _tree_run(st, tree_id, run_id)
    return {"tree_id": tree_id, "pine": pipeline.pine_source(st, tree_id, rid)}


@mcp.tool()
@_errors
def generate_report(tree_id: TreeId, run_id: RunId = None) -> dict[str, Any]:
    """Render the HTML report of a tree and return its path."""
    st = store()
    rid = _tree_run(st, tree_id, run_id)
    return {"tree_id": tree_id, "path": str(pipeline.generate_report(st, tree_id, rid))}


@mcp.tool()
@_errors
def classify_strategy(tree_id: TreeId, run_id: RunId = None) -> dict[str, Any]:
    """Strategy type of a tree (trend following, momentum, breakout, mean reversion or mixed), from its
    rules (which indicators, with or against the move) and its trades (entries vs the prior 1-day / 5-day move)."""
    st = store()
    rid = _tree_run(st, tree_id, run_id)
    result = pipeline.classify_stored(st, tree_id, rid)
    st.save_classification(rid, tree_id, result)
    return {"run_id": rid, "tree_id": tree_id, **_round(result)}


@mcp.tool()
@_errors
def build_portfolio(
    run_ids: Annotated[list[str] | None, Field(default=None, max_length=50)] = None,
    max_strategies: Annotated[int | None, Field(default=None, ge=1, le=30)] = None,
    max_correlation: Annotated[float | None, Field(default=None, gt=0, le=1)] = None,
    select_by: Literal["validation_dsr", "validation_sharpe", "train_fitness"] | None = None,
    max_per_type: Annotated[int | None, Field(default=None, ge=1)] = None,
    weighting: Literal["equal", "inverse_vol"] | None = None,
    survivors_only: bool | None = None,
) -> dict[str, Any]:
    """Aggregate the best strategies that are not correlated with each other (daily R, train + validation).
    Omitted settings come from the `portfolio` section of the config. Never reads the holdout."""
    st = store()
    overrides = {k: v for k, v in {"max_strategies": max_strategies, "max_correlation": max_correlation,
                                   "select_by": select_by, "max_per_type": max_per_type, "weighting": weighting,
                                   "survivors_only": survivors_only}.items() if v is not None}
    settings = base_config().portfolio.model_copy(update=overrides)
    out = pipeline.build_and_report_portfolio(st, run_ids, type(settings).model_validate(settings.model_dump()))
    r = out["result"]
    return {
        "portfolio_id": out["portfolio_id"], "report": out["report"],
        "members": _round([{k: m[k] for k in ("tree_id", "strategy_type", "weight", "risk_per_trade_pct",
                                              "validation_sharpe", "validation_dsr")} for m in out["members"]]),
        "splits": _round(r["splits"]), "mean_abs_correlation": _round(r["mean_abs_correlation"]),
        "prop_firm": _round({k: v for k, v in r["prop_firm"].items() if k != "note"}),
        "types": r["types"], "pool_size": r["pool_size"], "n_rejected": len(r["rejected"]),
    }


@mcp.tool()
@_errors
def get_portfolio(portfolio_id: Annotated[str, Field(pattern=r"^pf_\w+$")]) -> dict[str, Any]:
    """A stored portfolio: settings, members, metrics by split, correlation matrix and prop-firm budget."""
    p = store().get_portfolio(portfolio_id)
    return _round(p)


@mcp.tool()
@_errors
def simulate_challenge(
    tree_id: Annotated[str | None, Field(default=None, pattern=r"^[0-9a-f]{4,32}$")] = None,
    portfolio_id: Annotated[str | None, Field(default=None, pattern=r"^pf_\w+$")] = None,
    profit_targets_pct: Annotated[list[float] | None, Field(default=None, min_length=1, max_length=4)] = None,
    max_days: Annotated[int | None, Field(default=None, ge=5)] = None,
    drawdown_mode: Literal["static", "trailing"] | None = None,
    risk_grid: Annotated[list[float] | None, Field(default=None, min_length=1, max_length=12)] = None,
    edge_haircut: Annotated[float | None, Field(default=None, ge=0, le=1)] = None,
    fee: Annotated[float | None, Field(default=None, ge=0)] = None,
) -> dict[str, Any]:
    """Prop-firm challenge simulator for a tree or a portfolio: probability to pass each phase and the whole
    challenge, failure causes, days to pass and funded-account survival, for each risk per trade. Uses train +
    validation days only. Omitted rules come from prop_firm.challenge in the config."""
    if (tree_id is None) == (portfolio_id is None):
        raise ConfigError("give exactly one of tree_id or portfolio_id")
    base = base_config().prop_firm.challenge
    update: dict[str, Any] = {k: v for k, v in {"drawdown_mode": drawdown_mode, "risk_grid": risk_grid,
                                               "edge_haircut": edge_haircut, "fee": fee}.items() if v is not None}
    if profit_targets_pct is not None or max_days is not None:
        targets = profit_targets_pct or [p.profit_target_pct for p in base.phases]
        update["phases"] = [{"name": f"phase {i + 1}", "profit_target_pct": t, "max_days": max_days,
                             "min_trading_days": base.phases[0].min_trading_days if base.phases else 0}
                            for i, t in enumerate(targets)]
    challenge = type(base).model_validate({**base.model_dump(), **update})
    st = store()
    res = (pipeline.challenge_for_tree(st, tree_id, None, challenge) if tree_id
           else pipeline.challenge_for_portfolio(st, portfolio_id, challenge))  # type: ignore[arg-type]
    return _round(res)


@mcp.tool()
@_errors
def list_runs(limit: Annotated[int, Field(ge=1, le=100)] = 20) -> dict[str, Any]:
    """Recent runs with their status and trial counts."""
    st = store()
    return {"runs": st.list_runs(limit), "portfolios": st.list_portfolios(limit), "total_trials": st.total_trials(),
            "holdout_accesses": st.holdout_trial_count()}


def main() -> None:
    setup_logging(logging.WARNING)
    mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
