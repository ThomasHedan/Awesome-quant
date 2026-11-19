"""Backtest one tree on every pair of a dataset and aggregate the results."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import pandas as pd

from fxgen.backtest.engine import TRADE_COLUMNS, ExitRules, simulate_pair
from fxgen.backtest.metrics import compute_metrics, daily_r, per_year
from fxgen.config import CostsConfig
from fxgen.market import Dataset
from fxgen.tree.evaluate import evaluate, leaf_paths
from fxgen.tree.node import Tree


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    metrics: dict[str, Any]
    per_pair: dict[str, dict[str, Any]]
    leaf_paths: list[str] = field(default_factory=list)

    def per_year(self, risk_pct: float) -> pd.DataFrame:
        return per_year(self.trades, risk_pct)

    def daily(self, dataset: Dataset) -> pd.Series:
        return daily_r(self.trades, dataset.period)


def run_backtest(
    tree: Tree,
    dataset: Dataset,
    rules: ExitRules,
    costs: CostsConfig,
    risk_pct: float,
    pairs: list[str] | None = None,
) -> BacktestResult:
    """Evaluate ``tree`` on each pair, simulate trades, aggregate metrics."""
    columns = dataset.column_of()
    frames: list[pd.DataFrame] = []
    per_pair: dict[str, dict[str, Any]] = {}
    for pair, data in dataset.pairs.items():
        if pairs is not None and pair not in pairs:
            continue
        action, leaf = evaluate(tree, data.X, columns)
        trades = simulate_pair(data, action, leaf, rules, costs)
        per_pair[pair] = compute_metrics(trades, dataset.period, risk_pct)
        if not trades.empty:
            frames.append(trades)
    all_trades = (
        pd.concat(frames, ignore_index=True).sort_values(["exit_time", "pair"], ignore_index=True, kind="stable")
        if frames else pd.DataFrame(columns=TRADE_COLUMNS)
    )
    paths = leaf_paths(tree)
    if not all_trades.empty:
        all_trades["branch"] = [paths[i] for i in all_trades["leaf"]]
    return BacktestResult(
        trades=all_trades,
        metrics=compute_metrics(all_trades, dataset.period, risk_pct),
        per_pair=per_pair,
        leaf_paths=paths,
    )
