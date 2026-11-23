"""Ranking tables of trees with in-sample, out-of-sample and holdout metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

RANK_COLUMNS = [
    "tree_id", "survives", "finalist", "strategy_type", "depth", "n_internal",
    "train_fitness", "train_sharpe", "train_dsr", "train_n_trades",
    "validation_sharpe", "validation_dsr", "validation_profit_factor", "validation_max_dd_pct",
    "holdout_sharpe", "holdout_dsr", "holdout_profit_factor", "holdout_max_dd_pct",
    "wf_positive_share", "sensitivity_worst_ratio", "mc_max_dd_pct_95", "mc_max_risk_pct", "rules",
]


def ranking_frame(rows: list[dict[str, Any]], sort_by: str = "validation_dsr") -> pd.DataFrame:
    """Rows -> DataFrame with a stable column order, best first."""
    df = pd.DataFrame(rows)
    for col in RANK_COLUMNS:
        if col not in df:
            df[col] = None
    df = df[RANK_COLUMNS]
    if sort_by in df and df[sort_by].notna().any():
        df = df.sort_values(sort_by, ascending=False, na_position="last", kind="stable")
    return df.reset_index(drop=True)


def write_ranking(df: pd.DataFrame, directory: Path, name: str = "ranking") -> tuple[Path, Path]:
    """Write ``<name>.csv`` and ``<name>.json`` in ``directory``."""
    directory.mkdir(parents=True, exist_ok=True)
    csv_path, json_path = directory / f"{name}.csv", directory / f"{name}.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(json.loads(df.to_json(orient="records")), indent=2))
    return csv_path, json_path
