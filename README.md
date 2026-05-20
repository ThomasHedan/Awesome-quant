# Quant Backtesting Dashboard

A local-first quantitative backtesting dashboard built on
[`vectorbt`](https://github.com/polakowo/vectorbt) and
[`quantstats`](https://github.com/ranaroussi/quantstats), wrapped behind clean
data and analysis interfaces and exposed through a multipage Streamlit UI.

The four pillars:

1. **Full backtest** with pro metrics (Sharpe, Sortino, Calmar, CAGR, max
   drawdown, profit factor, expectancy, win rate, exposure, fees, tail ratio,
   VaR/CVaR).
2. **Benchmark comparison** — strategy vs. buy-and-hold (alpha, beta, IR,
   correlation, relative equity).
3. **Monte Carlo** edge analysis — trade-order shuffles + returns bootstrap,
   probability of ruin / hitting a return goal, confidence-interval fan
   chart, and a percentile rank of the realized result.
4. **Grid-search optimization** — exhaustive, vectorized parameter sweeps
   with heatmaps and a parameter-stability view to expose overfitting.

> **Architecture note.** `vectorbt` is a dependency, not a fork. We wrap it
> behind our own interfaces (data adapters, analysis modules, dashboard) so
> upstream releases continue to work and we keep our code small.

## Status

- [x] Step 1 — project scaffold + vectorbt/quantstats smoke test
- [x] Step 2 — `DataSource` ABC + `InstrumentSpec` + adapters (yfinance,
      Alpaca, CCXT, Databento, IBKR, CSV/Parquet) + parquet cache + tests
- [ ] Step 3 — `Strategy` ABC + reference strategies + no-look-ahead tests
- [ ] Step 4 — engine runner + metrics
- [ ] Step 5 — benchmark module
- [ ] Step 6 — Monte Carlo module
- [ ] Step 7 — optimizer
- [ ] Step 8 — SQLite persistence
- [ ] Step 9 — Streamlit dashboard
- [ ] Step 10 — final docs

## Quickstart

```bash
# 1. Install (uv recommended)
uv sync --extra dev

# 2. Copy and fill in API keys
cp .env.sample .env

# 3. Smoke test the engine
uv run python -m quant_dashboard.smoke

# 4. Run the test suite
uv run pytest

# 5. (Once step 9 lands) launch the dashboard
uv run streamlit run dashboard/app.py
```

## Data sources

All sources implement `quant_dashboard.data.base.DataSource` and return a
normalized OHLCV DataFrame:

- UTC `DatetimeIndex` named `timestamp`
- columns: `open`, `high`, `low`, `close`, `volume` (all `float64`)
- ascending by time, deduped, no NaN rows in OHLC

| Source     | Asset class       | Requires                      | Notes |
|------------|-------------------|-------------------------------|-------|
| `yfinance` | equities, ETFs    | nothing                       | Free, daily-friendly history. |
| `alpaca`   | equities          | `ALPACA_API_KEY` / `_SECRET`  | Intraday-friendly alternative. |
| `ccxt`     | crypto spot       | nothing (Binance default)     | Exchange configurable. |
| `databento`| futures           | `DATABENTO_API_KEY`           | Continuous + raw contracts. |
| `ibkr`     | futures / equities| running IB Gateway / TWS      | Live or paper account. |
| `file`     | anything          | local CSV / Parquet           | Documented schema below. |

Each adapter attaches an `InstrumentSpec` (multiplier, tick size, tick value,
session calendar hint) so downstream fees, slippage, and P&L use the right
units regardless of asset class.

### Local file schema

CSV or Parquet, columns (case-insensitive): `timestamp, open, high, low,
close, volume`. `timestamp` parsed as UTC.

## Adding a strategy

(Step 3 — coming next.)

## Adding a data source

Subclass `quant_dashboard.data.base.DataSource`, implement `fetch(...)`, and
register the adapter in `quant_dashboard.data.__init__`.

## Calibrating fees

(Step 4 — coming next.)
