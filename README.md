# Quant Backtesting Dashboard

A local-first quantitative backtesting dashboard built on
[`vectorbt`](https://github.com/polakowo/vectorbt) and
[`quantstats`](https://github.com/ranaroussi/quantstats), wrapped behind
clean data + analysis interfaces and surfaced through a multipage Streamlit
UI.

The four pillars:

1. **Full backtest** with pro metrics (Sharpe, Sortino, Calmar, CAGR, max
   drawdown, profit factor, expectancy, win rate, exposure, fees, tail ratio,
   VaR/CVaR).
2. **Benchmark comparison** — strategy vs. buy-and-hold or any external
   returns series (alpha, beta, IR, correlation, relative equity).
3. **Monte Carlo** edge analysis — trade-order shuffles + per-bar returns
   bootstrap, probability of ruin / hitting a return goal, a CI fan chart
   with the realized equity overlaid, and a percentile rank of the realized
   Sharpe and total return inside the bootstrap distribution. The single
   best "is the edge real or luck?" view.
4. **Grid-search optimization** — exhaustive, vectorized parameter sweeps
   with heatmaps and a neighborhood-stability score to expose overfit
   isolated peaks.

> **Architecture note.** `vectorbt` is a dependency, not a fork. We wrap it
> behind our own interfaces so upstream releases keep working and our code
> stays small.

## Status

All ten build steps are complete.

| Step | Module                          | Status |
|------|---------------------------------|--------|
| 1    | scaffold + smoke test           | done |
| 2    | `data/` adapters + cache        | done |
| 3    | `strategies/` + registry        | done |
| 4    | `engine/runner` + `metrics`     | done |
| 5    | `engine/benchmark`              | done |
| 6    | `engine/montecarlo`             | done |
| 7    | `engine/optimizer`              | done |
| 8    | `storage/` SQLite run store     | done |
| 9    | `dashboard/` Streamlit pages    | done |
| 10   | docs                            | done |

98 tests passing.

## Quickstart

```bash
# 1. Install (uv recommended)
uv sync --extra dev

# 2. Copy and fill in API keys (only for the adapters you'll use)
cp .env.sample .env

# 3. Smoke test the engine
uv run python -m quant_dashboard.smoke

# 4. Run the test suite
uv run pytest

# 5. Launch the dashboard
uv run streamlit run dashboard/app.py
```

The dashboard opens at <http://localhost:8501> with four pages in the sidebar:
**Backtest**, **Benchmark**, **MonteCarlo**, **Optimization**. The landing
page shows the most recent saved runs and optimizations.

## Library decisions (locked)

These are deliberate and not up for re-litigation:

- **Engine.** `vectorbt==0.26.2` (Apache + Commons-Clause).
- **Metrics + tearsheet.** `quantstats==0.0.62` (`qs.stats.*`, `qs.reports.html`).
- **Plotting.** `plotly>=5.18,<5.22` (vectorbt 0.26 references the
  `heatmapgl` trace which plotly 5.22 removed — pin keeps imports clean).
- **UI.** Streamlit, multipage, pure Python.
- **Optimization.** Exhaustive grid search only. No Optuna, no Bayesian
  search. The guardrail keeps you out of trouble; the vectorized engine
  makes brute force fast enough.

## Project layout

```
src/quant_dashboard/
  data/           # DataSource ABC, InstrumentSpec, adapters, parquet cache
  strategies/    # Strategy ABC, ParamSpec, registry, reference strategies
  engine/        # runner, metrics, benchmark, montecarlo, optimizer
  storage/       # SQLite run store
  smoke.py       # 5-line vectorbt + quantstats sanity test
dashboard/
  app.py          # Streamlit entry
  _shared.py      # widgets: data-source picker, strategy picker, cost editor
  pages/          # 1_Backtest / 2_Benchmark / 3_MonteCarlo / 4_Optimization
tests/            # 98 tests covering every module
```

## Data sources

All sources implement `quant_dashboard.data.base.DataSource` and return a
normalized `OHLCV` bundle (DataFrame + `InstrumentSpec`):

- UTC `DatetimeIndex` named `timestamp`
- columns `open, high, low, close, volume` as `float64`
- ascending, deduped, no NaN in OHLC

| Source     | Asset class       | Requires                      | Notes |
|------------|-------------------|-------------------------------|-------|
| `yfinance` | equities / ETFs   | nothing                       | Free, daily-friendly history. |
| `alpaca`   | equities          | `ALPACA_API_KEY` / `_SECRET`  | Intraday alternative. |
| `ccxt`     | crypto spot       | nothing (Binance default)     | Any CCXT exchange id. Paginated. |
| `databento`| futures           | `DATABENTO_API_KEY`           | Continuous + raw contracts. |
| `ibkr`     | futures / equities| running IB Gateway / TWS      | Live or paper. |
| `file`     | anything          | local CSV / Parquet           | Schema below. |

Network-bound adapters are gated behind config and fail with an actionable
`DataSourceError` if a key or Gateway is missing — never crashes the UI.

### Local-file schema

CSV or Parquet, columns (case-insensitive): `timestamp, open, high, low,
close, volume`. `timestamp` is parsed as UTC.

### Instrument specs

`InstrumentSpec` (multiplier, tick size, tick value, session) travels with
every OHLCV so fees / slippage / P&L are computed in the right units per
asset class. A small registry seeds common futures roots
(ES/MES/NQ/MNQ/CL/GC/ZN); unknown roots get sane defaults and accept
`spec_overrides` to plug the gap.

## Adding a strategy

1. Subclass `quant_dashboard.strategies.base.Strategy`.
2. Set a unique `name` class attribute and a one-line `description`.
3. Implement `param_specs()` returning a list of `ParamSpec` (the dashboard
   reads these to build the parameter widgets and the optimizer reads them
   to build the grid).
4. Implement `_compute_signals(data, **params) -> Signals`. **Signals are
   aligned to the decision bar** — the engine shifts them by one bar before
   submitting to vectorbt. Do not shift inside the strategy.
5. Decorate the class with `@register` (from `quant_dashboard.strategies`).

That's it — the dashboard's sidebar selector and the optimizer's grid
builder pick it up automatically.

Minimal example:

```python
from quant_dashboard.strategies import (
    ParamSpec, Signals, Strategy, register, boolify,
)


@register
class AlwaysLong(Strategy):
    name = "always_long"
    description = "Buy on bar 1, never sell. Sanity-check strategy."

    @classmethod
    def param_specs(cls):
        return []

    def _compute_signals(self, data, **params):
        idx = data.df.index
        entries = boolify(idx == idx[0])
        exits = boolify(idx == idx[-1])
        return Signals(entries=entries, exits=exits)
```

The no-look-ahead test in `tests/test_strategies.py` is parametrized over
every registered strategy — new strategies are checked automatically.

## Adding a data source

1. Subclass `quant_dashboard.data.base.DataSource`. Set `name` and
   `default_asset_class`.
2. Implement `_fetch_raw(symbol, timeframe, start, end)` returning a raw
   DataFrame. Vendor-shaped input is fine — `fetch()` calls
   `normalize_ohlcv()` for you.
3. Override `instrument_spec(symbol)` if the asset class needs contract
   metadata.
4. Add the adapter to `quant_dashboard.data.__init__`.

The public `fetch()` handles caching, normalization, slicing, validation,
and `InstrumentSpec` attachment, so adapters stay short.

## Calibrating fees

The engine uses fractional `fees` and `slippage` (fraction of trade
notional) applied per fill. Defaults are conservative (5 bps each).

| Asset class | Calibration |
|-------------|-------------|
| Crypto      | Exchange's taker fee (e.g. Binance: `0.001` = 10 bps). |
| Equities    | Broker commission + half-spread, as a fraction of typical trade value. Commission-free brokers: `slippage ≈ 0.0001`–`0.0005`. |
| Futures     | `commission_per_contract / (typical_price * multiplier)`. ES at 5000 with $50 multiplier and $2.50 commission ≈ `2.50 / (5000*50) = 0.00001`. Slippage: assume *k* ticks of slippage, then `k * tick_value / (typical_price * multiplier)`. |

> **Monte Carlo and optimization results are meaningless if fees aren't
> calibrated.** Always sanity-check round-trip costs against what your
> broker actually charges before reading anything into the metrics.

## Run persistence

Every backtest is saved to a local SQLite DB (path configurable via
`QUANT_DASHBOARD_DB`, default `.cache/quant-dashboard/runs.db`) with its
full config (source, symbol, timeframe, window, strategy, params, costs),
the computed metrics, and a deterministic 16-char `config_hash` so
duplicates are detectable. Monte Carlo summaries link to their parent
backtest. Optimization runs persist the full ranking table so heatmaps can
be re-rendered without re-running the search.

Python API:

```python
from quant_dashboard.storage import RunStore
store = RunStore("runs.db")
runs  = store.list_runs(strategy_name="sma_cross", limit=20)
row   = store.get_run(runs.iloc[0]["id"])
```

## Programmatic use

```python
from quant_dashboard.data import YFinanceSource
from quant_dashboard.strategies import SMACross
from quant_dashboard.engine import (
    CostsConfig, run_backtest, compute_metrics,
    compute_benchmark_comparison, run_monte_carlo,
    run_grid_search, heatmap, parameter_stability,
    tearsheet_html,
)

data = YFinanceSource().fetch("SPY", "1d", start="2018-01-01", end="2024-01-01")

result  = run_backtest(data, SMACross(), {"fast": 20, "slow": 100},
                       costs=CostsConfig(fees=0.0005, slippage=0.0005))
metrics = compute_metrics(result)
bm      = compute_benchmark_comparison(result)
mc      = run_monte_carlo(result, n_simulations=1000)
opt     = run_grid_search(data, SMACross,
                          {"fast": range(5, 30, 5), "slow": range(30, 150, 10)},
                          metric="sharpe")

tearsheet_html(result, "out/spy.html")
print(metrics.to_dict())
print(bm.to_dict())
print(mc.to_summary())
print(opt.rankings.head())
print(parameter_stability(opt).query("is_isolated_peak").head())
```

## Testing

```bash
uv run pytest                 # full suite (~17 s, 98 tests)
uv run pytest tests/test_data.py
uv run pytest -k no_lookahead # parametrized over every registered strategy
```

## License

MIT.
