---
name: new-strategy
description: Scaffold a new trading strategy in src/quant_dashboard/strategies/, register it, and add a unit test. Use whenever the user says "add a strategy", "implement a strategy", "create a strategy for X", or describes a signal rule (e.g. "go long on Bollinger lower-band touch"). Enforces the decision-bar / no-look-ahead pattern.
---

# Scaffold a new strategy

Goal: turn the user's verbal description into a clean, registered, tested
strategy that participates in the dashboard automatically.

## Step 1 — clarify what they want

Before writing code, confirm in one or two lines:

- **Signal rule.** Entry condition and exit condition, in plain words.
- **Parameters.** Names, default values, and reasonable bounds for the
  optimizer (e.g. `period: int, 14, 2..60`).
- **Direction.** Long-only is the default. If they want shorts too, ask
  explicitly — most reference strategies are long-only.

If anything is ambiguous (`"momentum strategy"` is not specific enough), ask
one focused question via `AskUserQuestion` before writing code.

## Step 2 — create the strategy file

Path: `src/quant_dashboard/strategies/<snake_name>.py`. Use this template
(do not deviate from the structure):

```python
"""<one-sentence description of the signal rule>."""

from __future__ import annotations

import pandas as pd

from quant_dashboard.data import OHLCV
from quant_dashboard.strategies.base import ParamSpec, Signals, Strategy, boolify
from quant_dashboard.strategies.registry import register


@register
class MyStrategy(Strategy):
    name = "my_strategy"           # snake_case, unique across the registry
    description = "<one-line description, shown in the dashboard sidebar>"

    @classmethod
    def param_specs(cls) -> list[ParamSpec]:
        return [
            ParamSpec(
                name="period", kind="int",
                default=14, low=2, high=60, step=1,
                description="<what this controls>",
            ),
            # ... one ParamSpec per parameter
        ]

    def _compute_signals(self, data: OHLCV, *, period: int) -> Signals:
        # Compute indicators on data.df. Indicators MUST be causal -
        # value at t depends only on bars [0..t]. rolling(), ewm(adjust=False),
        # cumulative-style operations are fine.
        close = data.df["close"]
        # ... your indicator here ...

        entries = boolify(<your entry condition>)
        exits   = boolify(<your exit condition>)
        return Signals(entries=entries, exits=exits)
```

Hard rules:

1. **Decision-bar timing.** Signal at index `t` reflects the close at `t`.
   Do NOT call `.shift(-1)` or look at `data.df.iloc[t+1]`. The engine
   shifts by one bar before sending to vectorbt.
2. **No NaN in signals.** Wrap the final boolean series in `boolify()`,
   which fills the leading-window NaN with `False`.
3. **Index alignment.** `entries` and `exits` must share `data.df.index`.
   `Signals.__post_init__` enforces this; the engine re-enforces it.
4. **Degenerate params.** If the parameter combination is degenerate (e.g.
   `fast >= slow` for a crossover), return all-False signals rather than
   raising — the optimizer will sweep these and we want them to look flat.

## Step 3 — register and expose

In `src/quant_dashboard/strategies/__init__.py` add the import and the name
to `__all__`. Pattern is already established for `SMACross` and
`RSIMeanReversion`.

## Step 4 — add a focused unit test

Append to `tests/test_strategies.py` (don't create a new file unless the
strategy has its own indicator helpers that deserve isolated tests). Add at
minimum:

- a test that the strategy produces some signals on the standard `_ohlcv()`
  fixture,
- a degenerate-params test (if applicable).

The parametrized `test_no_lookahead_for_all_strategies` will pick up the
new strategy automatically via the registry. If it fails, the strategy has
look-ahead — fix the strategy, do not adjust the test.

If the strategy needs a non-default parameter set for the look-ahead test
to be meaningful (i.e. the defaults emit zero signals on the test data),
extend `_strategy_params_for_no_lookahead` in `tests/test_strategies.py`.

## Step 5 — run the suite

```bash
uv run pytest tests/test_strategies.py
```

All tests must pass before reporting back. If the no-look-ahead test fails,
the strategy is peeking — common culprits:

- using `data.df["open"]` at the same bar (the open is known at t, but
  trading at the open from a close-bar decision still requires the
  decision to be from t-1; just stick with close-to-close).
- centering a rolling window (`center=True`).
- using `.shift(-k)` anywhere.
- `ewm(adjust=True, ...)` — use `adjust=False` for strict causality (Wilder
  smoothing uses this).

## Step 6 — confirm it's visible in the UI

Don't launch the dashboard just for this; the registry hook is enough. Tell
the user the strategy is now available in the **Backtest** and
**Optimization** pages of `dashboard/app.py`.

## Output to the user

Reply with:

- file paths created / modified,
- the parameter schema (so they can see what the dashboard will render),
- the test result (pass/fail count),
- a one-line example invocation:
  `uv run python -c "from quant_dashboard.data import YFinanceSource; from quant_dashboard.strategies import <Class>; from quant_dashboard.engine import run_backtest, compute_metrics; data = YFinanceSource().fetch('SPY','1d',start='2022-01-01'); print(compute_metrics(run_backtest(data, <Class>(), {<defaults>})))"`.
