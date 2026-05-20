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
- [x] Step 3 — `Strategy` ABC + reference strategies (SMA cross, RSI mean
      reversion) + registry + no-look-ahead test
- [x] Step 4 — engine runner + metrics (vectorbt + QuantStats tearsheet)
- [x] Step 5 — benchmark module (alpha / beta / IR / correlation + relative
      equity, defaults to buy-and-hold on the same instrument)
- [x] Step 6 — Monte Carlo (bootstrap + trade-shuffle, ruin / goal
      probabilities, CI fan chart, percentile rank of realized result)
- [x] Step 7 — optimizer (vbt exhaustive grid search, 2-D heatmaps,
      neighborhood parameter-stability scoring, large-grid guardrail)
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

1. Subclass `quant_dashboard.strategies.base.Strategy`.
2. Set a unique `name` class attribute and a one-line `description`.
3. Implement `param_specs()` returning a list of `ParamSpec` (the dashboard
   reads these to build the UI and the optimizer reads them to build the
   grid).
4. Implement `_compute_signals(data, **params) -> Signals`. Signals are
   aligned to the *decision bar* — the engine shifts them by one bar before
   submitting to vectorbt, so don't shift inside the strategy.
5. Decorate the class with `@register` (from `quant_dashboard.strategies`).

The bundled `SMACross` and `RSIMeanReversion` are short reference
implementations to copy from.

## Adding a data source

Subclass `quant_dashboard.data.base.DataSource`, implement `_fetch_raw(...)`
returning a raw OHLCV frame, override `instrument_spec(symbol)` to attach
contract metadata, and add the class to `quant_dashboard.data.__init__`.
`fetch()` (the public entry point) handles caching, normalization, slicing,
and validation — adapters only deal with the vendor.

## Calibrating fees

The engine uses fractional `fees` and `slippage` (fraction of trade
notional) applied per fill. Defaults are conservative (5 bps / 5 bps).

| Asset class | How to calibrate |
|-------------|------------------|
| Crypto      | Exchange's taker fee (e.g. Binance: `0.001` = 10 bps). |
| Equities    | Broker commission + half-spread, expressed as a fraction of typical trade value (commission-free brokers: `slippage ≈ 0.0001`-`0.0005`). |
| Futures     | `commission_per_contract / (typical_price * multiplier)`. E.g. ES at 5000 with $50 mult and $2.50 commission ≈ `2.50 / (5000*50) = 0.00001`. Slippage: `tick_value / (typical_price * multiplier)` per tick of slippage assumed. |

Monte Carlo and optimization results are meaningless if fees aren't
calibrated — always sanity-check that round-trip costs match what your
broker actually charges before reading anything into the metrics.
