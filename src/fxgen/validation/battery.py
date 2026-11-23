"""Full robustness battery for one tree (never touches the holdout)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np

from fxgen.backtest.metrics import daily_r
from fxgen.backtest.runner import run_backtest
from fxgen.config import Config
from fxgen.tree.node import Tree
from fxgen.validation.dsr import deflated_sharpe
from fxgen.validation.montecarlo import run_monte_carlo
from fxgen.validation.permutation import PermutationResult
from fxgen.validation.prop_firm import prop_firm_limits
from fxgen.validation.sensitivity import run_sensitivity
from fxgen.validation.walkforward import walkforward_tree
from fxgen.workspace import Workspace


@dataclass
class TrialContext:
    """Selection context needed by the deflated Sharpe and the permutation comparison."""

    n_trials: int
    permutation: PermutationResult | None = None


@dataclass
class RobustnessReport:
    train_fitness: float
    dsr: dict[str, dict[str, Any]] = field(default_factory=dict)
    walkforward: dict[str, Any] = field(default_factory=dict)
    sensitivity: dict[str, Any] = field(default_factory=dict)
    montecarlo: dict[str, Any] = field(default_factory=dict)
    prop_firm: dict[str, Any] = field(default_factory=dict)
    permutation: dict[str, Any] = field(default_factory=dict)
    checks: dict[str, bool] = field(default_factory=dict)

    @property
    def survives(self) -> bool:
        return bool(self.checks) and all(self.checks.values())

    def to_dict(self) -> dict[str, Any]:
        return {**asdict(self), "survives": self.survives}


def run_robustness(tree: Tree, ws: Workspace, cfg: Config, context: TrialContext, train_fitness: float) -> RobustnessReport:
    """Walk-forward, deflated Sharpe, sensitivity, Monte Carlo and permutation for ``tree``."""
    risk = cfg.backtest.risk_per_trade_pct
    report = RobustnessReport(train_fitness=float(train_fitness))

    for split in ("train", "validation"):
        period = ws.period(split)
        result = run_backtest(tree, ws.tree_dataset(tree, period), ws.rules, cfg.costs, risk)
        report.dsr[split] = deflated_sharpe(daily_r(result.trades, period), context.n_trials).to_dict()

    report.walkforward = walkforward_tree(tree, ws, cfg).to_dict()
    report.sensitivity = run_sensitivity(tree, ws, ws.period("validation"), cfg).to_dict()

    full = ws.period("train+validation")
    trades = run_backtest(tree, ws.tree_dataset(tree, full), ws.rules, cfg.costs, risk).trades
    mc = run_monte_carlo(trades["r_multiple"].to_numpy() if len(trades) else np.array([]), risk,
                         cfg.validation.montecarlo, cfg.prop_firm, seed=cfg.seed)
    report.montecarlo = mc.to_dict()
    prop = prop_firm_limits(trades, full, cfg.prop_firm)
    report.prop_firm = prop.to_dict()

    if context.permutation is not None:
        perm = context.permutation
        exceed = sum(b >= train_fitness for b in perm.permuted_best)
        p_tree = (1 + exceed) / (1 + len(perm.permuted_best))
        report.permutation = {**perm.to_dict(), "tree_p_value": float(p_tree)}
        report.checks["permutation"] = bool(p_tree <= cfg.validation.permutation.significance)

    report.checks["validation_positive"] = report.dsr["validation"]["sharpe_annual"] > 0
    report.checks["dsr_train"] = report.dsr["train"]["dsr"] >= cfg.validation.dsr_min
    report.checks["walkforward"] = bool(report.walkforward["passed"])
    report.checks["sensitivity"] = bool(report.sensitivity["passed"])
    report.checks["montecarlo"] = mc.max_risk_pct_for_limit >= cfg.prop_firm.min_risk_per_trade_pct
    report.checks["daily_loss"] = prop.max_risk_pct_daily >= cfg.prop_firm.min_risk_per_trade_pct
    return report
