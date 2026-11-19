"""The locked holdout.

The holdout is read in exactly two situations, both recorded in a ledger:

* the final stage of a run, once, on that run's finalists;
* an explicit manual backtest (e.g. ``backtest_tree`` from the MCP server),
  which is logged and adds one trial to the deflated-Sharpe trial count.

No optimisation can ever receive holdout data (see ``ga.evolve.check_not_holdout``).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Protocol

from fxgen.backtest.metrics import daily_r
from fxgen.backtest.runner import BacktestResult, run_backtest
from fxgen.errors import HoldoutLockedError
from fxgen.tree.node import Tree
from fxgen.tree.serialize import tree_hash
from fxgen.validation.dsr import deflated_sharpe
from fxgen.workspace import Workspace

logger = logging.getLogger(__name__)


class HoldoutLedger(Protocol):
    def holdout_stage_done(self, run_id: str) -> bool: ...

    def record_holdout_access(self, run_id: str | None, tree_hash: str, source: str) -> None: ...

    def holdout_trial_count(self) -> int: ...


@dataclass
class MemoryLedger:
    """In-memory ledger (tests and dry runs)."""

    accesses: list[tuple[str | None, str, str]] = field(default_factory=list)

    def holdout_stage_done(self, run_id: str) -> bool:
        return any(r == run_id and s == "final_stage" for r, _, s in self.accesses)

    def record_holdout_access(self, run_id: str | None, tree_hash: str, source: str) -> None:
        self.accesses.append((run_id, tree_hash, source))

    def holdout_trial_count(self) -> int:
        return len(self.accesses)


def backtest_holdout(ws: Workspace, tree: Tree, ledger: HoldoutLedger, run_id: str | None, source: str) -> BacktestResult:
    """Backtest one tree on the holdout and record the access first."""
    ledger.record_holdout_access(run_id, tree_hash(tree), source)
    period = ws.period("holdout", allow_holdout=True)
    logger.warning("holdout accessed (%s) for tree %s", source, tree_hash(tree))
    return run_backtest(tree, ws.tree_dataset(tree, period), ws.rules, ws.cfg.costs, ws.cfg.backtest.risk_per_trade_pct)


def run_holdout_stage(
    ws: Workspace, run_id: str, finalists: list[Tree], ledger: HoldoutLedger, prior_trials: int
) -> dict[str, dict[str, Any]]:
    """Evaluate a run's finalists on the holdout, once per run.

    The deflated Sharpe uses ``prior_trials`` (every tree evaluated so far)
    plus every holdout access recorded in the ledger, this stage included.
    """
    if ledger.holdout_stage_done(run_id):
        raise HoldoutLockedError(f"the holdout stage of run {run_id} has already been executed; it cannot be repeated")
    results = {tree_hash(t): backtest_holdout(ws, t, ledger, run_id, "final_stage") for t in finalists}
    n_trials = prior_trials + ledger.holdout_trial_count()
    period = ws.period("holdout", allow_holdout=True)
    out: dict[str, dict[str, Any]] = {}
    for h, result in results.items():
        out[h] = {
            "metrics": result.metrics,
            "per_pair": result.per_pair,
            "dsr": deflated_sharpe(daily_r(result.trades, period), n_trials).to_dict(),
            "result": result,
        }
    return out
