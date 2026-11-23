"""End-to-end orchestration: evolve, validate, select finalists, holdout, reports.

Stage 1 (``run_pipeline``) never reads the holdout. Stage 2 (``run_holdout``)
reads it once per run, on that run's finalists only.
"""

from __future__ import annotations

import gc
import json
import logging
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from fxgen.backtest.metrics import daily_r, per_year
from fxgen.backtest.runner import BacktestResult, run_backtest
from fxgen.config import Config, Timeframe
from fxgen.data.loader import PairData, load_universe
from fxgen.data.splits import split_period
from fxgen.errors import FxgenError, HoldoutLockedError, TreeError
from fxgen.ga.diversity import similarity_matrix
from fxgen.ga.evolve import Evaluation, EvolutionResult, GenerationStats, evolve
from fxgen.reporting.html_report import ReportData, render_report
from fxgen.reporting.pine import export_pine
from fxgen.reporting.ranking import ranking_frame, write_ranking
from fxgen.storage.db import Store
from fxgen.tree.node import Tree, depth, n_internal
from fxgen.tree.serialize import from_dict, to_dict, to_rules, tree_hash
from fxgen.validation.battery import TrialContext, run_robustness
from fxgen.validation.dsr import deflated_sharpe
from fxgen.validation.holdout import backtest_holdout, run_holdout_stage
from fxgen.validation.permutation import PermutationResult, run_permutation_test
from fxgen.validation.walkforward import walkforward_process
from fxgen.workspace import Workspace, build_workspace

logger = logging.getLogger(__name__)

Progress = Callable[[str, dict[str, Any]], None]
_WORKSPACES: dict[str, Workspace] = {}


class Cancelled(FxgenError):
    """The run was cancelled by the user."""


def get_workspace(cfg: Config, universe: dict[str, PairData] | None = None) -> Workspace:
    """Workspace for a config, cached in-process by config hash (the 2 most recent)."""
    key = cfg.config_hash()
    if key not in _WORKSPACES:
        if len(_WORKSPACES) >= 2:
            _WORKSPACES.pop(next(iter(_WORKSPACES)))
        _WORKSPACES[key] = build_workspace(cfg, universe)
    return _WORKSPACES[key]


def run_config(store: Store, run_id: str) -> Config:
    return Config.model_validate(json.loads(store.get_run(run_id)["config_json"]))


def output_dir(cfg: Config, run_id: str) -> Path:
    return cfg.resolve(cfg.storage.output_dir) / run_id


def save_tree(store: Store, ws: Workspace, tree: Tree) -> str:
    tid = tree_hash(tree)
    store.save_tree(tid, to_dict(tree), to_rules(tree, ws.describe), depth(tree), n_internal(tree))
    return tid


def store_backtest(store: Store, run_id: str, tree_id: str, split: str, result: BacktestResult, risk_pct: float,
                   dsr: float | None = None) -> None:
    store.save_backtest(run_id, tree_id, split, result.metrics, result.per_pair, result.trades,
                        per_year(result.trades, risk_pct), dsr=dsr)


def select_candidates(evo: EvolutionResult, n: int, radius: float) -> list[Evaluation]:
    """Best trees by train fitness, skipping near-clones of already selected ones."""
    picked: list[Evaluation] = []
    for e in evo.best(max(10 * n, 50)):
        if picked:
            stack = np.stack([e.phenotype, *[p.phenotype for p in picked]])
            if similarity_matrix(stack)[0, 1:].max() >= radius:
                continue
        picked.append(e)
        if len(picked) == n:
            break
    return picked


def run_pipeline(
    cfg: Config,
    store: Store,
    progress: Progress | None = None,
    should_stop: Callable[[], bool] | None = None,
    universe: dict[str, PairData] | None = None,
    run_id: str | None = None,
) -> str:
    """Evolve on train, validate candidates, run robustness tests, pick finalists. No holdout access."""
    notify = progress or (lambda stage, info: None)
    stop = should_stop or (lambda: False)
    run_id = store.create_run(cfg.model_dump(mode="json"), cfg.config_hash(), cfg.seed, run_id=run_id)
    try:
        _run_stages(cfg, store, run_id, notify, stop, universe)
    except Cancelled:
        store.update_run(run_id, status="cancelled")
        raise
    except Exception as exc:
        store.update_run(run_id, status="failed", error=str(exc))
        raise
    store.update_run(run_id, status="done")
    return run_id


def _run_stages(cfg: Config, store: Store, run_id: str, notify: Progress, stop: Callable[[], bool],
                universe: dict[str, PairData] | None) -> None:
    risk = cfg.backtest.risk_per_trade_pct
    notify("workspace", {"run_id": run_id})
    ws = get_workspace(cfg, universe)
    store.save_conditions(run_id, [
        {"id": c.id, "family": c.family, "timeframe": c.timeframe, "kind": c.kind, "description": c.describe(),
         "activation": ws.redundancy.activation.get(c.id), "condition": c.to_dict()}
        for c in ws.conditions
    ])

    def on_gen(stats: GenerationStats) -> None:
        notify("evolution", vars(stats))

    evo = evolve(ws.split_dataset("train"), ws.conditions, cfg, ws.rules, on_generation=on_gen, should_stop=stop)
    if evo.cancelled:
        raise Cancelled(f"run {run_id} cancelled during evolution")
    store.log_trials(run_id, "ga", evo.n_trials)
    store.update_run(run_id, n_trials=evo.n_trials, history_json=evo.history_frame().to_dict("records"))

    candidates = select_candidates(evo, cfg.validation.n_candidates, cfg.ga.sharing_radius)
    notify("candidates", {"n_candidates": len(candidates)})
    for e in candidates:
        tid = save_tree(store, ws, e.tree)
        store.save_run_tree(run_id, tid, e.fitness.value, {**e.fitness.components, "reason": e.fitness.reason}, is_candidate=1)
        for split in ("train", "validation"):
            result = run_backtest(e.tree, ws.tree_dataset(e.tree, ws.period(split)), ws.rules, cfg.costs, risk)
            store_backtest(store, run_id, tid, split, result, risk)
        if stop():
            raise Cancelled(f"run {run_id} cancelled during validation")

    permutation: PermutationResult | None = None
    if cfg.validation.permutation.n_runs > 0 and candidates:
        notify("permutation", {"n_runs": cfg.validation.permutation.n_runs})
        # Free the real workspace while permuted copies are built: it is rebuilt afterwards.
        del ws
        _WORKSPACES.pop(cfg.config_hash(), None)
        gc.collect()
        permutation = run_permutation_test(universe if universe is not None else load_universe(cfg), cfg,
                                           candidates[0].fitness.value, seed=cfg.seed + 7)
        store.update_run(run_id, permutation_json=permutation.to_dict())
        ws = get_workspace(cfg, universe)
    if stop():
        raise Cancelled(f"run {run_id} cancelled before walk-forward")
    if cfg.validation.walkforward.n_windows > 0:
        notify("walkforward", {"n_windows": cfg.validation.walkforward.n_windows})
        store.update_run(run_id, walkforward_json=walkforward_process(ws, cfg, seed=cfg.seed + 3).to_dict())

    context = TrialContext(n_trials=store.total_trials(), permutation=permutation)
    survivors: list[tuple[float, float, str]] = []
    for i, e in enumerate(candidates):
        notify("robustness", {"tree": i + 1, "of": len(candidates)})
        tid = tree_hash(e.tree)
        report = run_robustness(e.tree, ws, cfg, context, e.fitness.value).to_dict()
        store.save_robustness(run_id, tid, report)
        for split, dsr in report["dsr"].items():
            store.set_dsr(run_id, tid, split, dsr["dsr"])
        if report["survives"]:
            survivors.append((report["dsr"]["validation"]["dsr"], report["dsr"]["validation"]["sharpe_annual"], tid))
        if stop():
            raise Cancelled(f"run {run_id} cancelled during robustness")

    finalists = [tid for _, _, tid in sorted(survivors, reverse=True)[: cfg.validation.n_finalists]]
    for tid in finalists:
        store.set_flags(run_id, tid, is_finalist=1)
    write_rankings(store, cfg, run_id)
    notify("done", {"run_id": run_id, "n_candidates": len(candidates), "n_finalists": len(finalists)})
    logger.info("run %s: %d candidates, %d survive, %d finalists", run_id, len(candidates), len(survivors), len(finalists))


def write_rankings(store: Store, cfg: Config, run_id: str) -> tuple[Path, Path]:
    df = ranking_frame(store.ranking_rows(run_id), sort_by="validation_dsr")
    return write_ranking(df, output_dir(cfg, run_id))


def finalists(store: Store, run_id: str) -> list[str]:
    return [r["tree_id"] for r in store.query(
        "SELECT tree_id FROM run_trees WHERE run_id = ? AND is_finalist = 1", (run_id,))]


def run_holdout(store: Store, run_id: str, make_reports: bool = True) -> dict[str, dict[str, Any]]:
    """Final, single holdout evaluation of a run's finalists, then reports and Pine exports."""
    cfg = run_config(store, run_id)
    ws = get_workspace(cfg)
    ids = finalists(store, run_id)
    if not ids:
        raise HoldoutLockedError(f"run {run_id} has no finalist: nothing survived validation, the holdout stays untouched")
    trees = {tid: from_dict(store.get_tree(tid)["tree_json"]) for tid in ids}
    results = run_holdout_stage(ws, run_id, list(trees.values()), store, prior_trials=store.selection_trials())
    store.update_run(run_id, holdout_done=1)
    risk = cfg.backtest.risk_per_trade_pct
    summary: dict[str, dict[str, Any]] = {}
    for tid, entry in results.items():
        store_backtest(store, run_id, tid, "holdout", entry["result"], risk, dsr=entry["dsr"]["dsr"])
        summary[tid] = {"metrics": entry["metrics"], "dsr": entry["dsr"]}
    write_rankings(store, cfg, run_id)
    if make_reports:
        for tid in ids:
            generate_report(store, tid, run_id)
            write_pine(store, tid, run_id)
    return summary


def generate_report(store: Store, tree_id: str, run_id: str | None = None) -> Path:
    """Render the HTML report of a tree from stored results."""
    run_id = run_id or store.run_of_tree(tree_id)
    if run_id is None:
        raise TreeError(f"tree {tree_id} is not attached to any run")
    cfg = run_config(store, run_id)
    tree_row = store.get_tree(tree_id)
    trades = store.get_trades(run_id, tree_id)
    trades = trades[trades["split"].isin(["train", "validation", "holdout"])]
    metrics: dict[str, dict[str, Any]] = {}
    per_pair: dict[str, dict[str, dict[str, Any]]] = {}
    for row in store.get_metrics(run_id, tree_id):
        if row["split"] not in ("train", "validation", "holdout"):
            continue
        if row["pair"] == "ALL":
            metrics[row["split"]] = row["metrics"]
        else:
            per_pair.setdefault(row["split"], {})[row["pair"]] = row["metrics"]
    if not metrics:
        raise TreeError(f"no stored backtest for tree {tree_id} in run {run_id}")
    data = ReportData(
        tree_id=tree_id, rules=tree_row["rules"], tree_json=tree_row["tree_json"], depth=tree_row["depth"],
        n_internal=tree_row["n_internal"], trades=trades, metrics=metrics, per_pair=per_pair,
        risk_pct=cfg.backtest.risk_per_trade_pct, robustness=store.get_robustness(run_id, tree_id), run_id=run_id,
        notes=[] if "holdout" in metrics else ["Holdout not evaluated yet for this tree."],
    )
    return render_report(data, output_dir(cfg, run_id) / "reports" / f"{tree_id}.html")


def pine_source(store: Store, tree_id: str, run_id: str | None = None) -> str:
    run_id = run_id or store.run_of_tree(tree_id)
    if run_id is None:
        raise TreeError(f"tree {tree_id} is not attached to any run")
    cfg = run_config(store, run_id)
    ws = get_workspace(cfg)
    tree = from_dict(store.get_tree(tree_id)["tree_json"])
    conditions = {c.id: c for c in ws.tree_conditions(tree)}
    return export_pine(tree, conditions, ws.features, tree_id, cfg.data.execution_timeframe)


def write_pine(store: Store, tree_id: str, run_id: str | None = None) -> Path:
    run_id = run_id or store.run_of_tree(tree_id)
    cfg = run_config(store, run_id)  # type: ignore[arg-type]
    path = output_dir(cfg, run_id) / "pine" / f"{tree_id}.pine"  # type: ignore[arg-type]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(pine_source(store, tree_id, run_id))
    return path


def rerun_robustness(store: Store, tree_id: str, run_id: str | None = None) -> dict[str, Any]:
    """Re-run the robustness battery on a stored tree (never touches the holdout)."""
    run_id = run_id or store.run_of_tree(tree_id)
    if run_id is None:
        raise TreeError(f"tree {tree_id} is not attached to any run")
    cfg = run_config(store, run_id)
    ws = get_workspace(cfg)
    tree = from_dict(store.get_tree(tree_id)["tree_json"])
    run = store.get_run(run_id)
    perm = PermutationResult(**json.loads(run["permutation_json"])) if run.get("permutation_json") else None
    fitness_rows = store.query("SELECT train_fitness FROM run_trees WHERE run_id = ? AND tree_id = ?", (run_id, tree_id))
    train_fitness = fitness_rows[0]["train_fitness"] if fitness_rows and fitness_rows[0]["train_fitness"] is not None else 0.0
    report = run_robustness(tree, ws, cfg, TrialContext(store.total_trials(), perm), train_fitness).to_dict()
    store.save_robustness(run_id, tree_id, report)
    return report


def manual_backtest(
    store: Store,
    tree: Tree,
    split: str,
    pairs: list[str] | None = None,
    timeframe: Timeframe | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """Backtest a hand-written tree. Every call is logged as a trial; holdout calls are also
    recorded as holdout accesses and count in the deflated-Sharpe trial number."""
    run_id = run_id or store.latest_run()
    if run_id is None:
        raise FxgenError("no finished run yet: run an evolution first so thresholds are fitted on its train split")
    cfg = run_config(store, run_id)
    if timeframe and timeframe != cfg.data.execution_timeframe:
        order = ["H1", "H4", "D1"]
        context = [tf for tf in order if order.index(tf) > order.index(timeframe)]
        cfg = Config.model_validate({**cfg.model_dump(mode="json"),
                                     "data": {**cfg.data.model_dump(mode="json"), "execution_timeframe": timeframe,
                                              "context_timeframes": context}})
    ws = get_workspace(cfg)
    unknown = [p for p in pairs or [] if p not in ws.frames]
    if unknown:
        raise FxgenError(f"pairs {unknown} are not loaded; available: {sorted(ws.frames)}")
    ws.tree_conditions(tree)
    tid = save_tree(store, ws, tree)
    if split == "holdout":
        full = backtest_holdout(ws, tree, store, run_id, "manual")
        result = full if not pairs else run_backtest(tree, ws.tree_dataset(tree, ws.period("holdout", allow_holdout=True)),
                                                     ws.rules, cfg.costs, cfg.backtest.risk_per_trade_pct, pairs)
        period = ws.period("holdout", allow_holdout=True)
    else:
        store.log_trials(run_id, "manual", 1, note=f"{tid} on {split}")
        period = ws.period(split)  # type: ignore[arg-type]
        result = run_backtest(tree, ws.tree_dataset(tree, period), ws.rules, cfg.costs, cfg.backtest.risk_per_trade_pct, pairs)
    n_trials = store.total_trials()
    dsr = deflated_sharpe(daily_r(result.trades, period), n_trials).to_dict()
    store_backtest(store, run_id, tid, f"manual_{split}", result, cfg.backtest.risk_per_trade_pct, dsr=dsr["dsr"])
    return {"tree_id": tid, "run_id": run_id, "split": split, "metrics": result.metrics, "per_pair": result.per_pair,
            "dsr": dsr, "n_trials_total": n_trials, "holdout_accesses_total": store.holdout_trial_count(),
            "trades": len(result.trades)}


def equity_correlation(store: Store, run_id: str, tree_ids: list[str], split: str) -> pd.DataFrame:
    """Correlation of daily R series of several trees on one split."""
    cfg = run_config(store, run_id)
    period = split_period(cfg.splits, split, allow_holdout=True)  # type: ignore[arg-type]
    series = {tid: daily_r(store.get_trades(run_id, tid, split), period) for tid in tree_ids}
    return pd.DataFrame(series).corr()
