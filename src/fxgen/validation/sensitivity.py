"""Sensitivity of a tree to small changes of its thresholds and exit parameters."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

from fxgen.backtest.runner import run_backtest
from fxgen.conditions.thresholds import shifted_percentile
from fxgen.config import Config
from fxgen.data.splits import Period
from fxgen.tree.node import Tree
from fxgen.workspace import Workspace


@dataclass
class SensitivityResult:
    base_sharpe: float
    variants: dict[str, float] = field(default_factory=dict)
    worst_sharpe: float = 0.0
    median_ratio: float = 0.0
    worst_ratio: float = 0.0
    passed: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def run_sensitivity(tree: Tree, ws: Workspace, period: Period, cfg: Config) -> SensitivityResult:
    """Re-run the backtest with each percentile node shifted +-k points and exits scaled.

    Passes when every variant keeps at least ``1 - max_degradation`` of the base Sharpe.
    """
    sens = cfg.validation.sensitivity
    risk = cfg.backtest.risk_per_trade_pct
    base_conditions = ws.tree_conditions(tree)
    base_ds = ws.dataset(period, base_conditions)
    base = run_backtest(tree, base_ds, ws.rules, cfg.costs, risk).metrics["sharpe"]
    variants: dict[str, float] = {}

    for shift in (-sens.percentile_shift, sens.percentile_shift):
        shifted_all = []
        for i, cond in enumerate(base_conditions):
            moved = shifted_percentile(cond, ws.frames, ws.train, shift)
            shifted_all.append(moved or cond)
            if moved is None:
                continue
            conds = [*base_conditions[:i], moved, *base_conditions[i + 1 :]]
            ds = ws.dataset(period, conds)
            variants[f"{cond.id}{shift:+d}"] = run_backtest(tree, ds, ws.rules, cfg.costs, risk).metrics["sharpe"]
        if any(a is not b for a, b in zip(shifted_all, base_conditions)):
            ds = ws.dataset(period, shifted_all)
            variants[f"all{shift:+d}"] = run_backtest(tree, ds, ws.rules, cfg.costs, risk).metrics["sharpe"]

    for factor in sens.exit_scale:
        variants[f"exits_x{factor:g}"] = run_backtest(tree, base_ds, ws.rules.scaled(factor), cfg.costs, risk).metrics["sharpe"]

    values = list(variants.values())
    worst = min(values) if values else base
    ratios = sorted(v / base for v in values) if base > 0 else [0.0]
    return SensitivityResult(
        base_sharpe=float(base),
        variants={k: float(v) for k, v in variants.items()},
        worst_sharpe=float(worst),
        median_ratio=float(ratios[len(ratios) // 2]),
        worst_ratio=float(ratios[0]),
        passed=bool(base > 0 and ratios[0] >= 1 - sens.max_degradation),
    )
