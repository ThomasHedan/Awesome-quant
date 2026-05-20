---
name: debug-lookahead
description: Diagnose suspected look-ahead bias in a strategy. Use when a backtest reports an implausibly good result (Sharpe >5 on daily bars, >10 intraday, or near-100% win rate), when the parametrized no-look-ahead test fails on a registered strategy, or when the user asks "is my strategy peeking?".
---

# Debug look-ahead bias

Goal: find the future-peeking line and fix it. Look-ahead is the #1 way
new strategies look great in backtest and lose money live.

## Step 1 — confirm the symptom

Ask which case applies:

1. The parametrized test `test_no_lookahead_for_all_strategies` is failing
   for a specific strategy.
2. A backtest's metrics look too good to be true.
3. The user has a hunch and wants a check.

For (2), define "too good": Sharpe > 5 on daily bars, > 10 intraday,
win_rate near 1.0, near-zero drawdown, or terminal return that beats
buy-and-hold by 10x in a year. Any of these warrants this skill.

## Step 2 — run the smoking-gun test manually

Even if `pytest` already failed, reproduce in isolation so you can see the
diff:

```python
from quant_dashboard.strategies import get_strategy
import numpy as np
import pandas as pd
from quant_dashboard.data import OHLCV, InstrumentSpec, AssetClass

# Same synthetic data as the test.
rng = np.random.default_rng(7)
rets = rng.normal(0.0005, 0.012, 300)
close = 100 * np.exp(np.cumsum(rets))
idx = pd.date_range("2023-01-01", periods=300, freq="1D", tz="UTC", name="timestamp")
df = pd.DataFrame({"open": close, "high": close*1.005, "low": close*0.995,
                   "close": close, "volume": [1000]*300}, index=idx)
data_full = OHLCV(df=df, spec=InstrumentSpec("T", AssetClass.EQUITY),
                  timeframe="1d", source="syn")

strat = get_strategy("<name>")()
params = {<reasonable params>}
sig_full = strat.generate_signals(data_full, **params)

# Slam future bars and re-run.
df2 = df.copy()
df2.iloc[150:, df2.columns.get_loc("close")] *= 1e3
df2.iloc[150:, df2.columns.get_loc("open")] *= 1e3
df2.iloc[150:, df2.columns.get_loc("high")] *= 1e3
df2.iloc[150:, df2.columns.get_loc("low")] *= 1e3
data_mut = OHLCV(df=df2, spec=data_full.spec, timeframe="1d", source="syn")
sig_mut = strat.generate_signals(data_mut, **params)

diff_entries = (sig_full.entries.iloc[:150] != sig_mut.entries.iloc[:150])
diff_exits   = (sig_full.exits.iloc[:150]   != sig_mut.exits.iloc[:150])
print("entries diverge at:", diff_entries[diff_entries].index.tolist())
print("exits   diverge at:", diff_exits[diff_exits].index.tolist())
```

The first timestamp where the signals diverge is your peek site.

## Step 3 — common culprits

Walk the strategy code with these in mind:

| Suspect | Why it leaks | Fix |
|---------|-------------|-----|
| `.shift(-k)` anywhere | Aligns signals to future bars | Remove; use lagged inputs only |
| `rolling(..., center=True)` | Window straddles future | `center=False` (default) |
| `ewm(adjust=True)` | Uses normalization that depends on full series | `ewm(adjust=False)` for strict causality |
| `data.df["close"].max()` (no window) | Uses the full series | `rolling(N).max()` |
| `iloc[t+1]` / `loc[future_ts]` | Direct future access | Only access bars `<= t` |
| Z-score / normalization with full-sample mean | Mean computed across the entire series including future | `rolling().mean()` / `expanding().mean()` |
| Reindexing that fills forward then backward | `bfill()` brings future into past | Drop `bfill()`; use `ffill()` only |
| Use of `data.df["open"]` for execution at t | Open at t is known, but trading on it from a close-at-t decision is execution-shift territory | Stick to close-to-close; the engine adds the bar-shift |

## Step 4 — fix in place

Edit the strategy. **Do not** add a shift inside the strategy to "make
the test pass" — the engine is the only place signals get shifted. If
fixing the indicator requires shifting, you have a real look-ahead and
the shift is masking the symptom.

## Step 5 — re-run and verify

```bash
uv run pytest tests/test_strategies.py -k no_lookahead
uv run pytest tests/test_strategies.py::test_<your_strategy>_produces_signals
```

Both must pass. Then re-run the backtest from Step 2 with realistic data
and confirm the metrics dropped to plausible territory.

## Step 6 — report back

Tell the user:

- the file + line where the peek was,
- what was leaking,
- the metric before/after the fix (so they see how much "performance" was
  illusory).
