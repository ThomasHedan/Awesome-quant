"""Exhaustive parameter-grid search with overfitting guardrails.

Per the architecture decision, this is brute-force only - no Optuna, no
Bayesian search. We rely on vectorbt's speed plus a large-grid guardrail to
keep us out of trouble.

The optimizer:

* Builds the Cartesian product of per-parameter grids (defaults come from
  each :class:`ParamSpec`).
* Estimates combination count + RAM and refuses to run if either crosses
  user-configurable thresholds (``force=True`` to override).
* Runs every combo through :func:`run_backtest` + :func:`compute_metrics`
  and collects a ranking table.
* Exposes :func:`heatmap` to pivot two parameters against the chosen metric.
* Exposes :func:`parameter_stability` which scores each combo by how well
  its neighbours perform — sharp isolated peaks are likely overfit.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product
from typing import Any, Iterable, Type

import numpy as np
import pandas as pd

from quant_dashboard.data import OHLCV
from quant_dashboard.engine.metrics import Metrics, compute_metrics
from quant_dashboard.engine.runner import BacktestResult, CostsConfig, run_backtest
from quant_dashboard.strategies.base import Strategy


class GridSearchError(RuntimeError):
    """User-actionable optimizer errors (e.g. too-large grid)."""


@dataclass
class OptimizationResult:
    """Output of :func:`run_grid_search`.

    ``rankings`` has one row per parameter combo with columns ``param_<name>``
    for each parameter and one column per metric in :class:`Metrics`. Sorted
    descending by the chosen ``metric``.
    """

    strategy_name: str
    param_names: tuple[str, ...]
    metric: str
    higher_is_better: bool
    rankings: pd.DataFrame
    best_params: dict[str, Any]
    best_metric_value: float
    failed_runs: list[dict] = field(default_factory=list)

    @property
    def n_combos(self) -> int:
        return len(self.rankings)


# ---------------------------------------------------------------------------


_METRIC_DIRECTION: dict[str, bool] = {
    # True = higher (more positive) is better.
    "total_return": True,
    "cagr": True,
    "sharpe": True,
    "sortino": True,
    "calmar": True,
    "win_rate": True,
    "profit_factor": True,
    "expectancy": True,
    # These are returned as *negative* numbers (loss tail / drawdown depth),
    # so "best" = closest to zero = largest raw value -> higher is better.
    "max_drawdown": True,
    "var_95": True,
    "cvar_95": True,
    # Strictly non-negative; smaller is better.
    "max_drawdown_duration_days": False,
    "annualized_volatility": False,
    "total_fees": False,
}


def _metric_is_higher_better(metric: str) -> bool:
    if metric not in _METRIC_DIRECTION:
        raise GridSearchError(
            f"unknown metric '{metric}'. Known: {sorted(_METRIC_DIRECTION)}"
        )
    return _METRIC_DIRECTION[metric]


# ---------------------------------------------------------------------------


def _resolve_grids(
    strategy_cls: Type[Strategy],
    grids: dict[str, Iterable[Any]] | None,
) -> dict[str, list[Any]]:
    """Build the effective per-param grid: caller overrides, otherwise the
    ParamSpec's ``grid()``."""
    specs = {p.name: p for p in strategy_cls.param_specs()}
    out: dict[str, list[Any]] = {}
    overrides = dict(grids or {})

    for name, spec in specs.items():
        if name in overrides:
            values = list(overrides.pop(name))
            if not values:
                raise GridSearchError(f"grid for '{name}' is empty")
            for v in values:
                spec.validate(v)  # surface bad values up front
            out[name] = values
        else:
            out[name] = spec.grid()

    if overrides:
        raise GridSearchError(
            f"grids for unknown params: {sorted(overrides)}"
        )
    return out


def estimate_combo_count(grids: dict[str, Iterable[Any]]) -> int:
    n = 1
    for values in grids.values():
        n *= len(list(values))
    return n


def estimate_ram_bytes(n_combos: int, n_bars: int, bytes_per_curve: int = 64) -> int:
    """Rough RAM estimate. Per combo we hold an equity curve + a returns
    series + some scalars; 64 bytes/bar is a comfortable upper bound."""
    return n_combos * n_bars * bytes_per_curve


# ---------------------------------------------------------------------------


def run_grid_search(
    data: OHLCV,
    strategy_cls: Type[Strategy],
    grids: dict[str, Iterable[Any]] | None = None,
    *,
    metric: str = "sharpe",
    costs: CostsConfig | None = None,
    max_combos: int = 50_000,
    max_ram_bytes: int = 4 * 1024**3,
    force: bool = False,
) -> OptimizationResult:
    """Exhaustive grid search.

    Parameters
    ----------
    grids:
        Per-parameter override. Missing keys use the strategy's ParamSpec
        default grid.
    metric:
        Ranking metric; one of the keys in :class:`Metrics`. See
        ``_METRIC_DIRECTION`` for direction.
    max_combos / max_ram_bytes:
        Guardrails. If the configured grid exceeds either, raise
        :class:`GridSearchError` unless ``force=True``.
    """
    higher_is_better = _metric_is_higher_better(metric)
    effective = _resolve_grids(strategy_cls, grids)
    n_combos = estimate_combo_count(effective)
    n_bars = len(data.df)
    est_ram = estimate_ram_bytes(n_combos, n_bars)

    if not force:
        if n_combos > max_combos:
            raise GridSearchError(
                f"{n_combos:,} combinations exceeds max_combos={max_combos:,}. "
                "Tighten the grid, raise max_combos, or pass force=True."
            )
        if est_ram > max_ram_bytes:
            raise GridSearchError(
                f"estimated RAM {est_ram / 1024**3:.1f} GiB exceeds "
                f"max_ram_bytes={max_ram_bytes / 1024**3:.1f} GiB. "
                "Tighten the grid, raise max_ram_bytes, or pass force=True."
            )

    strategy = strategy_cls()
    rows: list[dict[str, Any]] = []
    failed: list[dict] = []

    names = list(effective.keys())
    for combo in product(*[effective[n] for n in names]):
        params = dict(zip(names, combo, strict=True))
        try:
            result = run_backtest(data, strategy, params, costs=costs)
            m: Metrics = compute_metrics(result)
        except Exception as exc:
            failed.append({"params": params, "error": str(exc)})
            continue

        row = {f"param_{k}": v for k, v in params.items()}
        row.update(m.to_dict())
        rows.append(row)

    if not rows:
        raise GridSearchError(
            f"all {n_combos} backtests failed. First error: "
            f"{failed[0]['error'] if failed else 'unknown'}"
        )

    df = pd.DataFrame(rows)
    df = df.sort_values(metric, ascending=not higher_is_better, na_position="last")
    df = df.reset_index(drop=True)

    best_row = df.iloc[0]
    best_params = {k.removeprefix("param_"): best_row[k] for k in df.columns if k.startswith("param_")}

    return OptimizationResult(
        strategy_name=strategy.name,
        param_names=tuple(names),
        metric=metric,
        higher_is_better=higher_is_better,
        rankings=df,
        best_params=best_params,
        best_metric_value=float(best_row[metric]),
        failed_runs=failed,
    )


# ---------------------------------------------------------------------------


def heatmap(
    opt: OptimizationResult,
    x_param: str,
    y_param: str,
    *,
    metric: str | None = None,
    aggfunc: str = "mean",
) -> pd.DataFrame:
    """Pivot ``rankings`` to a 2-D ``metric`` surface over (x_param, y_param).

    If the strategy has more than two parameters, other axes are aggregated
    by ``aggfunc`` (default ``"mean"``). Returns a DataFrame with ``y_param``
    on the rows and ``x_param`` on the columns - ready for plotly's heatmap.
    """
    metric = metric or opt.metric
    if f"param_{x_param}" not in opt.rankings.columns:
        raise GridSearchError(f"x_param '{x_param}' not in optimization results")
    if f"param_{y_param}" not in opt.rankings.columns:
        raise GridSearchError(f"y_param '{y_param}' not in optimization results")
    if metric not in opt.rankings.columns:
        raise GridSearchError(f"metric '{metric}' not in optimization results")

    pivot = opt.rankings.pivot_table(
        index=f"param_{y_param}",
        columns=f"param_{x_param}",
        values=metric,
        aggfunc=aggfunc,
    )
    pivot.index.name = y_param
    pivot.columns.name = x_param
    return pivot.sort_index().sort_index(axis=1)


# ---------------------------------------------------------------------------


def parameter_stability(
    opt: OptimizationResult,
    *,
    metric: str | None = None,
    neighborhood: int = 1,
) -> pd.DataFrame:
    """Annotate the ranking table with a neighborhood-stability score.

    For each combo, look at the ``(2*neighborhood + 1)^k`` grid of combos
    within ``neighborhood`` grid steps in every dimension and compute the
    mean metric across them. ``stability_score`` = neighborhood_mean /
    combo_metric (or the additive analogue for non-positive metrics).

    Stable combos have a score near 1.0; sharp isolated peaks have a score
    well below 1.0 (the peak is much higher than its neighbours) and are
    flagged via the ``is_isolated_peak`` boolean.
    """
    metric = metric or opt.metric
    df = opt.rankings.copy()
    param_cols = [f"param_{n}" for n in opt.param_names]

    # Map each unique value of each param to its rank along its axis, so
    # "neighbour" is defined by grid position rather than absolute value.
    ranks = {}
    for col in param_cols:
        uniq = sorted(df[col].unique())
        idx_map = {v: i for i, v in enumerate(uniq)}
        ranks[col] = df[col].map(idx_map)
        df[f"_rank_{col}"] = ranks[col]

    metric_arr = df[metric].to_numpy(dtype=float)
    rank_matrix = df[[f"_rank_{c}" for c in param_cols]].to_numpy(dtype=int)

    n = len(df)
    neighborhood_means = np.full(n, np.nan)

    for i in range(n):
        diffs = np.abs(rank_matrix - rank_matrix[i])
        mask = (diffs <= neighborhood).all(axis=1)
        # Exclude the combo itself from its own neighborhood mean.
        mask[i] = False
        values = metric_arr[mask]
        values = values[np.isfinite(values)]
        if values.size > 0:
            neighborhood_means[i] = values.mean()

    df["neighborhood_mean"] = neighborhood_means
    own = metric_arr
    # Score: ratio when both positive; additive fallback when own metric is
    # non-positive (Sharpe < 0, for example) to keep the score meaningful.
    score = np.where(
        np.isfinite(neighborhood_means) & (own > 0),
        neighborhood_means / np.where(own != 0, own, np.nan),
        1.0 + (neighborhood_means - own),  # additive: 1.0 == perfectly stable
    )
    df["stability_score"] = score

    # An "isolated peak" is a combo whose neighbours are noticeably worse.
    # Threshold: neighbours are at least 30% lower (for positive metrics) or
    # the additive gap exceeds 0.3 (for non-positive metrics).
    isolated = np.where(
        np.isfinite(neighborhood_means) & (own > 0),
        neighborhood_means / np.where(own != 0, own, np.nan) < 0.7,
        (own - neighborhood_means) > 0.3,
    )
    df["is_isolated_peak"] = isolated

    df = df.drop(columns=[f"_rank_{c}" for c in param_cols])
    return df
