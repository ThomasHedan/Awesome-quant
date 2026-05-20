# CLAUDE.md — Quant Backtesting Dashboard

You are working on a local-first quantitative backtesting dashboard built on
`vectorbt` + `quantstats`, with a Streamlit UI. The user is an experienced
trader migrating from QuantConnect (LEAN) who wants to iterate on strategies
quickly and trust the analysis.

This file is the durable context for working in this repo. Read it before
making non-trivial changes.

## Operating principles

1. **`vectorbt` is a dependency, never a fork.** Wrap it behind our own
   interfaces. Never edit anything under `.venv/`.
2. **No look-ahead bias, ever.** Strategies emit signals aligned to the
   decision bar (close at time `t`); the *engine* — not the strategy —
   shifts them one bar forward before submitting to vectorbt. The
   parametrized test in `tests/test_strategies.py::test_no_lookahead_for_all_strategies`
   runs against every registered strategy automatically. Don't shift inside
   a strategy. Don't bypass this in the engine.
3. **Reproducibility.** Seed every Monte Carlo. Grid search is deterministic
   by construction. Every backtest is persisted to SQLite (`runs.db`) with a
   `config_hash` so duplicates are detectable.
4. **Realistic costs.** Fees and slippage are first-class. The defaults
   (5 bps each) are conservative but not calibrated — when comparing a
   strategy seriously, set them per the table in `README.md`.
5. **Surface, don't swallow.** Errors at vendor / config boundaries become
   `DataSourceError` / `EngineError` / `MonteCarloError` / `GridSearchError`
   / `StorageError` with plain-English messages. The dashboard catches these
   and shows `str(exc)` to the user. Never `except Exception: pass`.

## Architecture

```
src/quant_dashboard/
  data/         DataSource ABC + InstrumentSpec + adapters + parquet cache
  strategies/  Strategy ABC + ParamSpec + registry + reference strategies
  engine/      runner, metrics, benchmark, montecarlo, optimizer
  storage/     SQLite RunStore
  smoke.py     5-line vectorbt + quantstats sanity test
dashboard/
  app.py + pages/   Streamlit UI (4 pages, one per analysis pillar)
tests/             pytest, no network, 98 tests
```

**Data flow:** every adapter returns a normalized `OHLCV` bundle (UTC index,
`open/high/low/close/volume` as `float64`, no NaN in OHLC, ascending,
deduped). Downstream code trusts this contract. New adapters must funnel
their raw response through `quant_dashboard.data.base.normalize_ohlcv`.

**Strategy contract:** `_compute_signals(data, **params) -> Signals` where
`Signals.entries` and `Signals.exits` are boolean Series aligned to the
OHLCV index, with **no NaN** and **decision-bar timing**. The engine adds
the one-bar execution shift.

**Engine contract:** `run_backtest(data, strategy, params, costs)` returns a
`BacktestResult` (vectorbt portfolio + reproducibility metadata). All
downstream analysis (`compute_metrics`, `compute_benchmark_comparison`,
`run_monte_carlo`) takes a `BacktestResult`, never the raw portfolio.

## Coding conventions

- **Python 3.11+**. `from __future__ import annotations` at the top of every
  module.
- **Tools.** `uv` for env (`uv sync --extra dev`, `uv run pytest`,
  `uv run python -m quant_dashboard.smoke`). Never `pip install` directly.
- **Type hints throughout.** Public functions have docstrings; private
  helpers don't need them.
- **Comments are rare.** Write a comment only when the *why* is non-obvious
  (a hidden constraint, a vectorbt API gotcha, a subtle invariant). Never
  comment the *what* — the code says that. Never reference tasks, PRs, or
  prior versions in comments.
- **Imports.** Adapters lazy-import their vendor SDK inside `_client()` /
  `_fetch_raw` so the dashboard boots even if an SDK is broken.
- **Errors.** Raise the module-local error class with a message that tells
  the user what to do (e.g. `"set ALPACA_API_KEY in your environment
  (see .env.sample)"`).
- **No emojis** in code, commit messages, or files unless the user asks.
- **No backwards-compat shims, no feature flags.** Single-user local app —
  change the code in place.
- **Don't pre-emptively abstract.** Three similar lines is fine. Abstract
  on the third real reuse, not the second hypothetical one.

## Running things

```bash
uv sync --extra dev                              # install deps
uv run pytest                                    # full suite (~17 s)
uv run pytest tests/test_strategies.py           # focused
uv run pytest -k no_lookahead                    # the headline invariant
uv run python -m quant_dashboard.smoke           # engine sanity
uv run streamlit run dashboard/app.py            # UI
```

## Library versions (pinned, don't bump casually)

- `vectorbt==0.26.2` — has known plotly compatibility quirk (see below).
- `quantstats==0.0.62`.
- `plotly>=5.18,<5.22` — vectorbt 0.26 references the `heatmapgl` trace
  which plotly 5.22 removed. Pin stays until vectorbt fixes upstream.
- `python>=3.11,<3.13` — quantstats has noisy warnings on 3.13.

## When in doubt

- **Adding a strategy?** Use the `new-strategy` skill. It enforces the
  decision-bar / no-look-ahead pattern and adds the registry hook + a unit
  test.
- **Running a backtest from chat?** Use the `run-backtest` skill.
- **Optimizing?** Use `optimize-params`. It refuses huge grids by default —
  that's a feature.
- **Edge analysis?** Use `monte-carlo-edge`.
- **Suspected look-ahead bug?** Use `debug-lookahead`.
- **New data source?** Use `add-data-source`.
- **Calibrating fees for a new instrument?** Use `calibrate-fees`.

Each skill is in `.claude/skills/<name>/SKILL.md`.

## Things to never do

- **Never** push to `main` without an explicit request.
- **Never** commit `.env` or any file containing API keys / tokens.
- **Never** disable the no-look-ahead test to make a strategy pass.
- **Never** add `try / except Exception: pass`. If you genuinely need a
  broad catch, raise a module-local error with context.
- **Never** add a comment that says "// fixes bug #123" or "// added for
  optimization page" — those belong in PR descriptions, not source.
- **Never** introduce a new dependency without asking. Especially not
  another backtest engine, another plotting library, or another data
  vendor SDK.

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
