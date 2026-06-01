---
name: multi-chart
description: >
  Run a registered strategy across multiple symbols and/or timeframes,
  then display a comparison table and equity-curve overlay.
  Use when the user says "test on multiple symbols", "how does X strategy
  perform on other markets", "run multi-chart", or asks for robustness
  across instruments.
---

# Multi-chart — run one strategy across many instruments

Goal: show whether a strategy generalizes across instruments, not just on
the symbol it was developed on.

---

## Step 1 — confirm inputs

You need:

1. **Strategy name** — must be in `available_strategies()`.
   ```bash
   uv run python -c "from quant_dashboard.strategies import available_strategies; print(available_strategies())"
   ```
2. **Parameters** — use the strategy's `default_params()` unless the user
   specifies. Warn if using defaults.
3. **Instrument list** — a list of `(symbol, timeframe)` pairs. Default if
   unspecified:
   ```
   SPY  1d
   QQQ  1d
   IWM  1d
   GLD  1d
   BTC-USD  1d
   ```
4. **Date range** — default `2020-01-01` to today.
5. **Data source** — yfinance is the default for equities and yfinance-listed
   crypto (e.g. `BTC-USD`). Use `CCXTSource` for exchange-specific crypto.
6. **Costs** — default `CostsConfig()` (5 bps fees + 5 bps slippage).

---

## Step 2 — run via uv

```python
from quant_dashboard.data import YFinanceSource
from quant_dashboard.strategies import get_strategy
from quant_dashboard.engine import CostsConfig, run_multi_chart

strategy = get_strategy("<name>")()
sources = [
    (YFinanceSource(), "SPY",     "1d"),
    (YFinanceSource(), "QQQ",     "1d"),
    (YFinanceSource(), "IWM",     "1d"),
    (YFinanceSource(), "GLD",     "1d"),
    (YFinanceSource(), "BTC-USD", "1d"),
]
result = run_multi_chart(
    sources, strategy, params={<params>},
    costs=CostsConfig(),
    start="2020-01-01",
)
print(result.summary_df.to_string(index=False))
```

---

## Step 3 — report

Print the full `summary_df` table with columns:

```
symbol | timeframe | total_return_pct | cagr_pct | sharpe | sortino
       | max_drawdown_pct | calmar | win_rate_pct | num_trades | profit_factor
```

Then deliver a **robustness verdict**:

| Result | Criterion |
|--------|-----------|
| **Robust** | Sharpe > 0 on ≥ 80% of instruments |
| **Selective** | Sharpe > 0 on 50–79% |
| **Fragile** | Sharpe > 0 on < 50% |

List instruments where `num_trades == 0` separately — they indicate
degenerate parameters for that symbol/timeframe.

---

## Step 4 — equity curve overlay (optional)

If the user wants a chart, run this snippet and save as PNG using matplotlib:

```python
import matplotlib.pyplot as plt

fig, ax = plt.subplots(figsize=(12, 5))
for ir in result.successful:
    eq = ir.result.equity_curve
    eq_norm = eq / eq.iloc[0]
    ax.plot(eq_norm.index, eq_norm.values, label=f"{ir.symbol} {ir.timeframe}")
ax.axhline(1.0, color="gray", linewidth=0.8, linestyle="--")
ax.legend()
ax.set_title(f"{result.strategy_name} — normalized equity")
ax.set_ylabel("Portfolio value (start = 1.0)")
plt.tight_layout()
plt.savefig("/tmp/multi_chart_equity.png", dpi=150)
print("Chart saved to /tmp/multi_chart_equity.png")
```

---

## Common pitfalls

- **`num_trades == 0` on some instruments.** Usually means the default
  params are too slow for that symbol's volatility. Tell the user; suggest
  optimizing per-instrument with `optimize-params`.
- **Sharpe implausibly high on all instruments.** Suspected look-ahead.
  Run `debug-lookahead` before reporting.
- **`DataSourceError` on one symbol.** yfinance may not carry it. Try
  appending the exchange suffix (e.g. `BTC-USD` vs `BTCUSDT` for CCXT).
- **Mismatched date ranges.** Some instruments have shorter history. The
  runner clips to available data — note this in the output if start dates
  differ materially.
