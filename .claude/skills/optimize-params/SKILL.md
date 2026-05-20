---
name: optimize-params
description: Run an exhaustive grid search on a registered strategy, then surface the top-N table, a 2-D heatmap (if applicable), and the parameter-stability flagging. Use when the user says "find the best parameters", "sweep fast/slow", "optimize my strategy", or asks how robust a configuration is.
---

# Optimize parameters

Goal: brute-force the parameter space (no Bayesian / Optuna in this repo —
locked by architecture) and report a robustness-aware ranking, not just the
peak.

## Step 1 — agree on the grid

Defaults from each `ParamSpec` are usually too broad for a sensible sweep.
Get from the user:

1. **Strategy + data** (same as `run-backtest`).
2. **Per-parameter ranges**. E.g. `fast: 5..30 step 5`, `slow: 30..150 step 10`.
3. **Ranking metric.** Default to `sharpe`. Known options: see
   `quant_dashboard.engine.optimizer._METRIC_DIRECTION`.

## Step 2 — preview the grid size

Before launching, compute the combination count and warn if it's big:

```python
from quant_dashboard.engine import estimate_combo_count, estimate_ram_bytes
grids = {"fast": list(range(5, 31, 5)), "slow": list(range(30, 151, 10))}
print(estimate_combo_count(grids),
      estimate_ram_bytes(estimate_combo_count(grids), len(data.df)) / 1024**3, "GiB")
```

The optimizer refuses grids above `max_combos=50_000` or
`max_ram_bytes=4 GiB` without `force=True`. That refusal is intentional —
don't bypass it. If the user genuinely needs a huge grid, surface the
warning and ask before adding `force=True`.

## Step 3 — run

```python
from quant_dashboard.engine import run_grid_search, parameter_stability, heatmap

opt = run_grid_search(
    data, strategy_cls, grids,
    metric="sharpe",                # or total_return, sortino, calmar, ...
    costs=CostsConfig(fees=0.0005, slippage=0.0005),
)
```

## Step 4 — analyze, not just report the peak

Always show three things:

1. **Top 10** of `opt.rankings`. Print the param columns + the ranking
   metric. Don't dump all 22 metrics columns.
2. **Parameter stability.** Run:

   ```python
   stab = parameter_stability(opt)
   isolated = stab[stab["is_isolated_peak"]]
   ```

   Tell the user how many combos are flagged as isolated peaks, what their
   neighbourhood means look like, and explicitly warn that the top-1 result
   is suspect if it appears in the isolated list. "Isolated peak" means
   neighbours are >30% worse — typical fingerprint of overfit.
3. **Heatmap data** if the strategy has exactly two parameters (or the user
   picks two of >2). Use `heatmap(opt, x_param=..., y_param=...)` and print
   the pivoted table. For visual inspection, point the user at the
   Optimization page of the dashboard (`uv run streamlit run
   dashboard/app.py`); don't generate plots in chat.

## Step 5 — persist

```python
from quant_dashboard.storage import RunStore
RunStore().save_optimization(
    opt, source=data.source, symbol=data.spec.symbol,
    timeframe=data.timeframe,
    start=data.df.index[0], end=data.df.index[-1], grids=grids,
)
```

So the dashboard can re-load the ranking table without re-running.

## Step 6 — recommend, don't just rank

Recommend the *best stable* combo, not the peak. A reasonable rule:

> pick the highest-ranked combo that is **not** flagged as `is_isolated_peak`
> and whose `stability_score` is within [0.7, 1.3].

State this rule explicitly in the reply so the user understands why you
didn't just hand them row 0.

## Common pitfalls

- **All zeros in rankings.** Strategy emitted no signals for any combo —
  almost always degenerate parameters (e.g. `fast >= slow` everywhere).
- **Failed runs.** `opt.failed_runs` lists params + error. Common cause:
  invalid combo (e.g. RSI `lower > upper`). Report the count.
- **NaN ranking metric.** Strategy traded but the metric was undefined
  (e.g. `sharpe` is NaN when returns are flat). The optimizer pushes
  NaN to the bottom; the recommended combo should not be NaN.
- **Choosing the wrong metric direction.** For `max_drawdown`, `var_95`,
  `cvar_95` "best" = closest-to-zero = highest raw value. The optimizer
  handles this via `_METRIC_DIRECTION`; don't second-guess it.
