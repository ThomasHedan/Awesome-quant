---
name: run-backtest
description: Run a backtest from chat - load data via a DataSource, execute a registered Strategy, compute the full Metrics, and print a short summary. Use when the user says "backtest X on Y", "run my strategy on SPY", "what's the Sharpe of...", or asks for performance metrics without opening the dashboard.
---

# Run a backtest

Goal: turn a one-line request like "backtest SMACross fast=20 slow=100 on
SPY 2018-2024" into a real backtest with metrics, without opening the
Streamlit UI.

## Step 1 — confirm the parameters

You need five things. Ask via `AskUserQuestion` only for ones genuinely
ambiguous:

1. **Strategy name** — must be in `available_strategies()`. If unknown,
   list them: `uv run python -c "from quant_dashboard.strategies import available_strategies; print(available_strategies())"`.
2. **Parameters** — fall back to `strategy.default_params()` if unspecified
   and warn the user you did.
3. **Data source** — yfinance is the default for equities, ccxt (Binance)
   for crypto, file for local.
4. **Symbol + timeframe + date range** — be explicit. If they say "recent",
   default to last 3 years to today.
5. **Costs** — default `CostsConfig()` (5 bps fees + 5 bps slippage,
   $100k init cash). Mention this in the output so they can recalibrate.

## Step 2 — run it via uv

Use `uv run python -c "<script>"`. Keep the script self-contained:

```python
from quant_dashboard.data import YFinanceSource          # or CCXTSource, FileSource, ...
from quant_dashboard.strategies import get_strategy
from quant_dashboard.engine import CostsConfig, run_backtest, compute_metrics

data = YFinanceSource().fetch("SPY", "1d",
                              start="2018-01-01", end="2024-01-01")
cls = get_strategy("sma_cross")
result = run_backtest(data, cls(), {"fast": 20, "slow": 100},
                      costs=CostsConfig(fees=0.0005, slippage=0.0005))
m = compute_metrics(result)
for k, v in m.to_dict().items():
    print(f"{k:30s} {v}")
```

## Step 3 — save the run (optional but encouraged)

If the user wants to reload later or run benchmark / Monte Carlo against
it, persist it:

```python
from quant_dashboard.storage import RunStore
run_id = RunStore().save_backtest(result, m, notes="<short note>")
```

Then tell the user the `run_id`, since the dashboard's Benchmark and
MonteCarlo pages use it to pick saved runs.

## Step 4 — report concisely

Summarize in <12 lines: total return, CAGR, Sharpe, Sortino, max drawdown,
calmar, num_trades, win_rate, profit_factor, expectancy, time_in_market,
total_fees. Round percentages to 2 dp, ratios to 2 dp.

If `num_trades == 0`, say so explicitly — the metrics are meaningless and
the user almost certainly picked degenerate parameters.

## Common pitfalls

- **Empty result from yfinance**: usually a bad ticker or a date range with
  no trading days. `DataSourceError` will say so.
- **Wrong timeframe for the source**: ccxt accepts `1m..1w`, yfinance has a
  narrower set (`1m, 5m, 15m, 30m, 60m / 1h, 1d, 1w, 1mo`), Alpaca needs
  keys. See `_TF_MS` / `_YF_INTERVAL` / `_ALPACA_TIMEFRAME` for the exact
  lists.
- **`InstrumentSpec` defaults wrong for futures**: if the user is testing a
  futures symbol, point out that fees should be calibrated per the table in
  `README.md` and consider using the `calibrate-fees` skill.
- **Look-ahead suspicion**: if Sharpe is implausibly high (>5 for daily
  bars, >10 for intraday), flag it and run the `debug-lookahead` skill
  before reporting the result as real.
