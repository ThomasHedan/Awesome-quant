---
name: monte-carlo-edge
description: Run a Monte Carlo edge analysis on a backtest - bootstrap per-bar returns and/or shuffle per-trade returns to test whether the realized result is a real edge or luck. Use when the user says "is this strategy real?", "test the edge", "run Monte Carlo", or asks for confidence intervals / probability of ruin / probability of hitting a return goal.
---

# Monte Carlo edge analysis

Goal: answer the single most important question after a promising backtest:
**is this edge real, or did the realized sequence get lucky?**

## Step 1 — decide which method (or run both)

Two methods, each answering a different question:

- **`bootstrap`** (default) — resample per-bar returns with replacement.
  Asks: *could this Sharpe come from this return distribution by chance?*
  Works on any backtest, including those with very few trades.
- **`shuffle`** — reorder the realized per-trade returns. Asks: *would a
  different trade sequence have produced the same equity curve?* Requires
  at least 2 trades.

When in doubt, run both. They diagnose different failure modes (e.g. a
strategy with one massive winning trade looks better under shuffle than
under bootstrap).

## Step 2 — get / build the backtest

If the user has a `run_id` in mind (e.g. "MC on run 7"), load it via
`RunStore().get_run(7)` and rebuild via `rerun_from_saved` (or replay the
config manually). Otherwise run the strategy first (see `run-backtest`
skill).

## Step 3 — choose realistic goal / ruin thresholds

- `goal_return` — what does the user consider "success"? Default 20%
  (annual) but always ask. For intraday strategies a 5% target may be more
  appropriate.
- `ruin_threshold` — drawdown depth that the user considers
  catastrophic. Default 50% but a leveraged trader may have 20% as their
  real ruin number.

Both are about the user's risk preferences, not the math — surface them
explicitly.

## Step 4 — run, seeded

```python
from quant_dashboard.engine import run_monte_carlo

mc_boot = run_monte_carlo(
    result, method="bootstrap", n_simulations=1000, seed=42,
    goal_return=0.20, ruin_threshold=0.50,
)
mc_shuf = run_monte_carlo(
    result, method="shuffle", n_simulations=1000, seed=42,
    goal_return=0.20, ruin_threshold=0.50,
)   # if num_trades >= 2
```

Use `n_simulations=1000` for interactive exploration; `n_simulations=5000`
if the user wants tighter percentile estimates. Always seed (default 42 in
the engine) so reruns match.

## Step 5 — answer the question, don't just dump stats

A useful reply has three parts:

1. **Headline verdict.** One sentence: "The realized Sharpe is at the
   75th percentile of the bootstrap distribution — modest evidence of a
   real edge, not strong." Calibrate language to the percentile:

   | Percentile | Plain-English verdict |
   |------------|-----------------------|
   | >= 95      | Strong evidence of edge. |
   | 75-95      | Modest evidence — keep validating. |
   | 50-75      | Roughly typical — could be luck. |
   | < 50       | Worse than the median random ordering. |

   Same logic for `return_percentile`.

2. **Risk numbers.** `prob_ruin` and `prob_goal` in percent, plus realized
   max drawdown vs the distribution.

3. **Caveat.** Always mention that Monte Carlo cannot detect *regime*
   shifts (it assumes returns are i.i.d.), only *sequence luck*. If the
   strategy was trained on the same data, this MC is descriptive at best —
   recommend a walk-forward / out-of-sample test as the next step.

## Step 6 — persist

```python
from quant_dashboard.storage import RunStore
RunStore().save_monte_carlo(run_id, mc_boot)
```

## Common pitfalls

- **`shuffle` raised "not enough trades"**: strategy traded <2 times. Fall
  back to `bootstrap`.
- **High `prob_goal` AND high `prob_ruin`**: the strategy is volatile.
  That's a finding, not a bug.
- **Percentile == NaN**: realized Sharpe was NaN (flat returns) or the
  distribution was empty. Report and explain.
- **Bootstrap on a strategy with autocorrelated returns**: i.i.d.
  resampling underestimates risk for trend-following / momentum strategies.
  Flag this if the user is running MC on something obviously
  serially-correlated.
