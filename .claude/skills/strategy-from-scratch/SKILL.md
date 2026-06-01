---
name: strategy-from-scratch
description: >
  Full A-to-Z workflow for developing and validating a trading strategy:
  scaffold → backtest → optimize → Monte Carlo edge test → multi-chart.
  Use when the user says "build a strategy", "develop X from scratch",
  "I want a complete strategy", or gives a signal idea and wants a full
  validated result.
---

# Strategy from scratch — A to Z

This skill orchestrates the complete development lifecycle in order.
Never skip a phase. Never report success until all phases pass.

---

## Phase 0 — Clarify (before writing any code)

Ask at most **three** things via `AskUserQuestion` if genuinely ambiguous:

1. **Signal rule.** Entry condition + exit condition in plain words.
   If the user gave a clear rule, skip this.
2. **Universe.** Default: `SPY 1d` (yfinance). Ask only if they want
   crypto, futures, or a specific list of instruments.
3. **Benchmark period.** Default: `2018-01-01` to today.

Do NOT ask about parameters — you'll sweep them in Phase 3.

---

## Phase 1 — Scaffold the strategy

Invoke the **`new-strategy`** skill.

Key requirements (enforce these even if new-strategy doesn't):
- `_compute_signals` uses only `data.df["close"]` (or other columns) at time
  `t` with no forward-looking references.
- Every indicator uses `boolify()` on the final boolean series.
- `param_specs()` returns at least two parameters with realistic bounds for
  the optimizer.

After scaffolding, run:
```bash
uv run pytest tests/test_strategies.py -x -q
```
All tests must be green before proceeding to Phase 2.
If `test_no_lookahead_for_all_strategies` fails → fix the strategy (never
bypass the test).

---

## Phase 2 — Baseline backtest

Invoke the **`run-backtest`** skill with:
- Symbol: the user's choice or `SPY`
- Timeframe: `1d`
- Date range: `2018-01-01` to today
- Params: strategy defaults
- Costs: `CostsConfig()` (5 bps fees + 5 bps slippage)

Report the 12-line summary. Flag immediately if:
- `num_trades == 0` → degenerate parameters, fix before continuing
- `Sharpe > 5` on daily bars → suspected look-ahead, run `debug-lookahead`
- `max_drawdown > 60%` → note it for the user; still continue

Persist the run with `RunStore().save_backtest(result, metrics, notes="baseline")`.

---

## Phase 3 — Parameter optimization

Invoke **`optimize-params`** skill.

Use a grid that covers the full `param_specs()` range with ~5 steps per
param. Do not exceed 2,000 total combinations without asking the user.

Report:
- Top-5 param combos by Sharpe (table)
- Stability flag: warn if the best combo is an isolated peak
- Recommended params (most stable, not necessarily highest Sharpe)

Update `strategy.py` default params to the recommended values.

---

## Phase 4 — Monte Carlo edge test

Invoke **`monte-carlo-edge`** skill on the baseline run (Phase 2 result).

Use `n_simulations=1000`, `seed=42`.

Report:
- Percentile rank of realized Sharpe vs bootstrap distribution
- Probability of ruin (drawdown > 50%)
- Probability of hitting 15% annualized return
- Verdict: **edge confirmed** (Sharpe rank ≥ 75th percentile) or
  **edge uncertain** (rank < 75th) or **likely noise** (rank < 50th)

If edge is uncertain or noise → tell the user and ask if they want to
continue to multi-chart or stop here.

---

## Phase 5 — Multi-chart validation

Invoke **`multi-chart`** skill (or directly call `run_multi_chart`) on:

```
SPY  1d
QQQ  1d
IWM  1d
GLD  1d
BTC-USD  1d  (yfinance crypto proxy)
```

Use the recommended params from Phase 3.

Report the `summary_df` table.
A strategy is **robust** if Sharpe > 0 on at least 4/5 instruments.
A strategy is **equity-specific** if it only works on SPY/QQQ/IWM.

---

## Final report to the user

Deliver a structured summary:

```
## Strategy: <StrategyName> — A to Z Report

### Signal rule
<one-sentence description>

### Recommended parameters
<param = value, param = value, ...>

### Baseline performance (SPY 1d, 2018–today)
Total return: X%   CAGR: X%   Sharpe: X.XX   Max DD: X%
Trades: N   Win rate: X%   Profit factor: X.XX

### Optimization
Best stable config: <params>
Stability: <isolated peak / stable plateau>

### Monte Carlo (1 000 sims, seed=42)
Sharpe rank: Xth percentile
P(ruin): X%   P(+15% pa): X%
Verdict: <edge confirmed / uncertain / likely noise>

### Multi-chart robustness (SPY / QQQ / IWM / GLD / BTC)
<table: symbol | sharpe | cagr% | max_dd%>
Verdict: <robust / equity-specific / fragile>

### Files changed
- src/quant_dashboard/strategies/<name>.py  (created)
- tests/test_strategies.py  (appended)
```

---

## What NOT to do

- Never skip the no-look-ahead test (Phase 1).
- Never report "looks good" if `num_trades == 0`.
- Never modify the engine shift to make a strategy pass.
- Never add `.shift(-1)` anywhere in a strategy.
- Never add a new dependency without asking.
