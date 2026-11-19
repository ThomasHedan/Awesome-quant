"""Walk-forward analysis on train + validation.

* ``walkforward_tree``: a fixed tree evaluated on each out-of-sample window.
* ``walkforward_process``: the whole GA re-run on each in-sample window and
  its best tree scored on the following out-of-sample window. This measures
  whether the *search process* generalises (walk-forward efficiency).
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np

from fxgen.backtest.runner import run_backtest
from fxgen.config import Config
from fxgen.data.splits import walkforward_windows
from fxgen.ga.evolve import evolve
from fxgen.tree.node import Tree
from fxgen.tree.serialize import to_dict
from fxgen.workspace import Workspace

logger = logging.getLogger(__name__)


@dataclass
class WalkForwardTreeResult:
    windows: list[dict[str, object]] = field(default_factory=list)
    positive_share: float = 0.0
    mean_sharpe: float = 0.0
    passed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass
class WalkForwardProcessResult:
    windows: list[dict[str, object]] = field(default_factory=list)
    mean_is_sharpe: float = 0.0
    mean_oos_sharpe: float = 0.0
    efficiency: float = 0.0
    positive_share: float = 0.0
    n_trials: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def walkforward_tree(tree: Tree, ws: Workspace, cfg: Config) -> WalkForwardTreeResult:
    """Sharpe of a fixed tree on every OOS window; passes if most windows are positive."""
    windows = walkforward_windows(cfg.splits, cfg.validation.walkforward)
    rows: list[dict[str, object]] = []
    for _, oos in windows:
        result = run_backtest(tree, ws.tree_dataset(tree, oos), ws.rules, cfg.costs, cfg.backtest.risk_per_trade_pct)
        rows.append({"oos": oos.label(), "sharpe": result.metrics["sharpe"], "n_trades": result.metrics["n_trades"],
                     "total_r": result.metrics["total_r"]})
    sharpes = np.array([r["sharpe"] for r in rows], dtype=float)
    positive = float((sharpes > 0).mean()) if sharpes.size else 0.0
    return WalkForwardTreeResult(
        windows=rows,
        positive_share=positive,
        mean_sharpe=float(sharpes.mean()) if sharpes.size else 0.0,
        passed=bool(sharpes.size and positive > 0.5 and sharpes.mean() > 0),
    )


def walkforward_process(ws: Workspace, cfg: Config, seed: int) -> WalkForwardProcessResult:
    """Re-run the GA on each IS window and score its best tree OOS."""
    wf = cfg.validation.walkforward
    rows: list[dict[str, object]] = []
    n_trials = 0
    for k, (is_period, oos_period) in enumerate(walkforward_windows(cfg.splits, wf)):
        logger.info("walk-forward window %d: IS %s OOS %s", k, is_period.label(), oos_period.label())
        evo = evolve(ws.dataset(is_period), ws.conditions, cfg, ws.rules, seed=seed + k,
                     population_size=wf.population_size, generations=wf.generations)
        n_trials += evo.n_trials
        best = evo.best(1)
        if not best:
            rows.append({"is": is_period.label(), "oos": oos_period.label(), "is_sharpe": 0.0, "oos_sharpe": 0.0, "tree": None})
            continue
        tree = best[0].tree
        oos = run_backtest(tree, ws.tree_dataset(tree, oos_period), ws.rules, cfg.costs, cfg.backtest.risk_per_trade_pct)
        rows.append({"is": is_period.label(), "oos": oos_period.label(), "is_sharpe": best[0].fitness.sharpe,
                     "oos_sharpe": oos.metrics["sharpe"], "tree": to_dict(tree)})
    is_s = np.array([r["is_sharpe"] for r in rows], dtype=float)
    oos_s = np.array([r["oos_sharpe"] for r in rows], dtype=float)
    mean_is = float(is_s.mean()) if is_s.size else 0.0
    mean_oos = float(oos_s.mean()) if oos_s.size else 0.0
    return WalkForwardProcessResult(
        windows=rows,
        mean_is_sharpe=mean_is,
        mean_oos_sharpe=mean_oos,
        efficiency=mean_oos / mean_is if mean_is > 0 else 0.0,
        positive_share=float((oos_s > 0).mean()) if oos_s.size else 0.0,
        n_trials=n_trials,
    )
