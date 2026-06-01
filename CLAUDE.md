# CLAUDE.md — Quant Strategy Template

This is a **local-first quantitative backtesting template** built on
`vectorbt` + `quantstats`, with a Streamlit UI and a complete CLI workflow.
The target user is an experienced trader who wants to iterate on strategies
quickly, trust the analysis, and validate ideas across multiple markets.

Read this file before every non-trivial change. It is the single source of
truth for how the AI must work in this repo.

---

## The two goals this template serves

1. **A-to-Z strategy development** — scaffold a strategy, backtest it,
   optimize parameters, run a Monte Carlo edge test, confirm there is no
   look-ahead bias. Use the `strategy-from-scratch` skill.
2. **Multi-chart validation** — take any registered strategy and run it
   across multiple symbols and timeframes to check robustness. Use the
   `multi-chart` skill.

Every AI session in this repo should start by checking which goal the user
is pursuing, then following the appropriate skill.

---

## Skill map — always use these, never reinvent

| Task | Skill | Trigger phrases |
|------|-------|----------------|
| Full strategy lifecycle A to Z | `strategy-from-scratch` | "build a strategy", "develop X from scratch", "complete strategy for Y" |
| Scaffold one strategy file | `new-strategy` | "add a strategy", "implement signal rule X" |
| Run a single backtest from chat | `run-backtest` | "backtest X on Y", "what's the Sharpe of" |
| Sweep parameters | `optimize-params` | "find best params", "sweep", "optimize" |
| Bootstrap edge test | `monte-carlo-edge` | "is this strategy real?", "test the edge", "P(ruin)" |
| Diagnose look-ahead | `debug-lookahead` | Sharpe >5 daily / >10 intraday, near-100% win rate |
| Multi-instrument test | `multi-chart` | "test on multiple symbols", "multi-chart", "robustness" |
| New data vendor | `add-data-source` | "add support for X", "connect to Y data" |
| Calibrate broker costs | `calibrate-fees` | "calibrate fees", "what fees for futures" |

**Rule:** when the user's request matches a skill trigger, invoke the skill
immediately. Do not implement the steps manually.

Each skill is in `.claude/skills/<name>/SKILL.md`.

---

## Architecture

```
src/quant_dashboard/
  data/         DataSource ABC + InstrumentSpec + adapters + parquet cache
  strategies/   Strategy ABC + ParamSpec + registry + reference strategies
  engine/       runner, metrics, benchmark, montecarlo, optimizer, multicharter
  storage/      SQLite RunStore
  smoke.py      5-line vectorbt + quantstats sanity test
dashboard/
  app.py + pages/   Streamlit UI (5 pages)
    1_Backtest.py
    2_Benchmark.py
    3_MonteCarlo.py
    4_Optimization.py
    5_MultiChart.py
tests/            pytest, no network, 100+ tests
.claude/skills/   9 domain skills
```

**Data flow:** adapter → `normalize_ohlcv()` → `OHLCV` bundle (UTC index,
`open/high/low/close/volume` float64, no NaN in OHLC) → engine.

**Strategy contract:** `_compute_signals(data, **params) -> Signals`.
Signals are boolean Series at the **decision bar** (close at `t`). The
*engine* shifts one bar forward before vectorbt. Strategies must never
shift internally.

**Engine contract:** `run_backtest(data, strategy, params, costs)` →
`BacktestResult`. All downstream analysis takes a `BacktestResult`.

**Multi-chart contract:** `run_multi_chart(sources, strategy, params, costs,
start, end)` → `MultiChartResult` with `.summary_df` and `.successful` /
`.failed` lists.

---

## Operating principles — non-negotiable

1. **`vectorbt` is a dependency, never a fork.** Wrap it behind our own
   interfaces. Never edit `.venv/`.
2. **No look-ahead bias, ever.** Strategies emit signals aligned to the
   decision bar (close at time `t`); the *engine* — not the strategy —
   shifts them one bar forward before submitting to vectorbt. The
   parametrized test `tests/test_strategies.py::test_no_lookahead_for_all_strategies`
   runs against every registered strategy automatically. Fix the strategy,
   never bypass the test.
3. **Reproducibility.** Seed every Monte Carlo (`seed=42` default). Grid
   search is deterministic by construction. Every backtest is persisted to
   SQLite with a `config_hash`.
4. **Realistic costs.** Fees and slippage are first-class. Defaults
   (5 bps each) are conservative but not calibrated — calibrate with
   `calibrate-fees` before comparing strategies seriously.
5. **Surface, don't swallow.** Raise `DataSourceError` / `EngineError` /
   `MonteCarloError` / `GridSearchError` / `StorageError` / `MultiChartError`
   with plain-English messages. Dashboard catches and shows `str(exc)`.
   Never `except Exception: pass`.

---

## Coding conventions

- **Python 3.11+**. `from __future__ import annotations` at the top of every
  module.
- **Tools.** `uv` for env and running. Never `pip install` directly.
- **Type hints throughout.** Public functions have docstrings; private
  helpers don't need them.
- **Comments are rare.** Only when the *why* is non-obvious (a hidden
  constraint, a vectorbt API gotcha, a subtle invariant). Never comment the
  *what*. Never reference tasks, PRs, or prior versions in comments.
- **Imports.** Adapters lazy-import their vendor SDK inside `_client()` /
  `_fetch_raw` so the dashboard boots even if an SDK is broken.
- **Errors.** Raise the module-local error class with a message that tells
  the user what to do.
- **No emojis** in code, commit messages, or files unless the user asks.
- **No backwards-compat shims, no feature flags.**
- **Don't pre-emptively abstract.** Three similar lines is fine.

---

## Running things

```bash
uv sync --extra dev                              # install deps
uv run pytest                                    # full suite
uv run pytest tests/test_strategies.py           # strategy tests
uv run pytest -k no_lookahead                    # the headline invariant
uv run pytest tests/test_multicharter.py         # multi-chart tests
uv run python -m quant_dashboard.smoke           # engine sanity check
uv run streamlit run dashboard/app.py            # UI (5 pages)
```

---

## Mandatory workflow for strategy development (A to Z)

When the user asks to build or develop a strategy, always follow this
sequence. Use the skills listed; do not skip phases.

```
Phase 0 — Clarify signal rule (AskUserQuestion if ambiguous)
Phase 1 — Scaffold with new-strategy skill → run tests → all green
Phase 2 — Baseline backtest with run-backtest skill → report metrics
Phase 3 — Optimize with optimize-params skill → recommend stable params
Phase 4 — Edge test with monte-carlo-edge skill → confirm real edge
Phase 5 — Validate with multi-chart skill → robustness verdict
```

Final report format (always use this structure):
```
## Strategy: <Name> — A to Z Report
### Signal rule
### Recommended parameters
### Baseline performance (symbol, date range)
### Optimization (top config + stability verdict)
### Monte Carlo (percentile rank, P(ruin), P(+15% pa), verdict)
### Multi-chart robustness (summary table + verdict)
### Files changed
```

---

## Library versions (pinned, don't bump casually)

- `vectorbt==0.26.2` — has known plotly compatibility quirk (see below).
- `quantstats==0.0.62`.
- `plotly>=5.18,<5.22` — vectorbt 0.26 references the `heatmapgl` trace
  which plotly 5.22 removed. Pin stays until vectorbt fixes upstream.
- `python>=3.11,<3.13` — quantstats has noisy warnings on 3.13.

---

## Things to never do

- **Never** push to `main` without an explicit request.
- **Never** commit `.env` or any file containing API keys / tokens.
- **Never** disable the no-look-ahead test to make a strategy pass.
- **Never** add `try / except Exception: pass`. If you genuinely need a
  broad catch, raise a module-local error with context.
- **Never** add a comment referencing a task, PR, or prior version.
- **Never** introduce a new dependency without asking.
- **Never** skip any phase of the A-to-Z workflow.
- **Never** report success when `num_trades == 0`.

## Things to do when asked to build something

- Show the proposed change in one or two sentences before doing it,
  especially if it touches the engine, the strategy contract, or the data
  contract.
- Run the relevant tests after every change. New strategies must keep the
  full `tests/test_strategies.py` green.
- If adding a new public function, add a test alongside it.
- For UI changes, run `uv run streamlit run dashboard/app.py` in the
  background, hit the page with `curl`, and confirm a 200 before claiming
  it works.
