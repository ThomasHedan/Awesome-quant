"""Shared widgets and helpers for the Streamlit pages.

Keeps the per-page files thin: data-source selector, strategy picker,
cost-config widget, and the singleton :class:`RunStore` all live here so
pages just compose them.
"""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from quant_dashboard.data import (
    AlpacaSource,
    AssetClass,
    CCXTSource,
    DatabentoSource,
    DataSource,
    DataSourceError,
    FileSource,
    IBKRSource,
    OHLCV,
    YFinanceSource,
)
from quant_dashboard.engine.runner import CostsConfig
from quant_dashboard.storage import RunStore
from quant_dashboard.strategies import (
    Strategy,
    available_strategies,
    get_strategy,
)

# ---------------------------------------------------------------------------

DB_PATH = Path(os.environ.get("QUANT_DASHBOARD_DB", ".cache/quant-dashboard/runs.db"))


@st.cache_resource(show_spinner=False)
def get_store() -> RunStore:
    """One RunStore per Streamlit process (cache_resource = no re-instantiate)."""
    return RunStore(DB_PATH)


# ---------------------------------------------------------------------------


SOURCE_CHOICES = ["yfinance", "ccxt", "alpaca", "databento", "ibkr", "file"]


def _make_source(source: str, *, file_path: str | None = None) -> DataSource:
    if source == "yfinance":
        return YFinanceSource()
    if source == "ccxt":
        exchange = st.session_state.get("ccxt_exchange", "binance")
        return CCXTSource(exchange=exchange)
    if source == "alpaca":
        return AlpacaSource()
    if source == "databento":
        return DatabentoSource()
    if source == "ibkr":
        contract_type = st.session_state.get("ibkr_contract_type", "future")
        return IBKRSource(contract_type=contract_type)
    if source == "file":
        # Asset class flows through the InstrumentSpec for downstream cost
        # decisions; default to OTHER so it doesn't lie.
        return FileSource(asset_class=AssetClass.OTHER)
    raise ValueError(f"unknown source '{source}'")


def data_source_picker(*, key_prefix: str = "ds") -> tuple[str, str, str, date, date]:
    """Render the source/symbol/timeframe/date controls in the sidebar.

    Returns ``(source, symbol, timeframe, start, end)``.
    """
    source = st.sidebar.selectbox(
        "Data source", SOURCE_CHOICES, key=f"{key_prefix}_source"
    )

    if source == "ccxt":
        st.sidebar.text_input(
            "Exchange (CCXT id)", value="binance",
            key="ccxt_exchange",
            help="Any CCXT exchange id (binance, coinbase, kraken, ...).",
        )
    if source == "ibkr":
        st.sidebar.selectbox(
            "IB contract type", ["future", "cont_future", "stock"],
            key="ibkr_contract_type",
        )

    if source == "file":
        symbol = st.sidebar.text_input(
            "Path to CSV / Parquet",
            value=str(DB_PATH.parent / "sample.csv"),
            key=f"{key_prefix}_symbol",
            help="Columns: timestamp, open, high, low, close, volume.",
        )
    else:
        default_symbol = {
            "yfinance": "SPY",
            "alpaca": "SPY",
            "ccxt": "BTC/USDT",
            "databento": "ESZ4",
            "ibkr": "ES",
        }[source]
        symbol = st.sidebar.text_input(
            "Symbol", value=default_symbol, key=f"{key_prefix}_symbol"
        )

    timeframe = st.sidebar.selectbox(
        "Timeframe", ["1d", "1h", "30m", "15m", "5m", "1m"],
        key=f"{key_prefix}_timeframe",
    )

    today = date.today()
    default_start = today - timedelta(days=365 * 3)
    start = st.sidebar.date_input(
        "Start", value=default_start, key=f"{key_prefix}_start"
    )
    end = st.sidebar.date_input("End", value=today, key=f"{key_prefix}_end")
    return source, symbol, timeframe, start, end


def load_ohlcv(
    source: str,
    symbol: str,
    timeframe: str,
    start: date | None,
    end: date | None,
) -> OHLCV:
    """Fetch (and cache) OHLCV. Surfaces DataSourceError as a friendly stop."""
    ds = _make_source(source)
    try:
        return ds.fetch(
            symbol, timeframe,
            start=pd.Timestamp(start) if start else None,
            end=pd.Timestamp(end) if end else None,
        )
    except DataSourceError as exc:
        st.error(str(exc))
        st.stop()


# ---------------------------------------------------------------------------


def strategy_picker(*, key_prefix: str = "strat") -> tuple[type[Strategy], dict[str, Any]]:
    """Sidebar strategy selector + a params widget built from the schema."""
    name = st.sidebar.selectbox(
        "Strategy", available_strategies(), key=f"{key_prefix}_name"
    )
    cls = get_strategy(name)
    st.sidebar.caption(cls.description)
    params: dict[str, Any] = {}
    st.sidebar.markdown("**Parameters**")
    for spec in cls.param_specs():
        widget_key = f"{key_prefix}_{name}_{spec.name}"
        if spec.kind == "int":
            params[spec.name] = st.sidebar.number_input(
                spec.name, value=int(spec.default),
                min_value=int(spec.low) if spec.low is not None else None,
                max_value=int(spec.high) if spec.high is not None else None,
                step=int(spec.step) if spec.step else 1,
                help=spec.description, key=widget_key,
            )
        elif spec.kind == "float":
            params[spec.name] = st.sidebar.number_input(
                spec.name, value=float(spec.default),
                min_value=float(spec.low) if spec.low is not None else None,
                max_value=float(spec.high) if spec.high is not None else None,
                step=float(spec.step) if spec.step else 0.1,
                help=spec.description, key=widget_key,
            )
        elif spec.kind == "choice":
            params[spec.name] = st.sidebar.selectbox(
                spec.name, list(spec.choices or ()),
                index=list(spec.choices or ()).index(spec.default),
                help=spec.description, key=widget_key,
            )
    return cls, params


def costs_widget(*, key_prefix: str = "costs") -> CostsConfig:
    st.sidebar.markdown("**Costs**")
    fees_bps = st.sidebar.number_input(
        "Fees (bps)", value=5.0, min_value=0.0, max_value=200.0, step=0.5,
        key=f"{key_prefix}_fees_bps",
    )
    slip_bps = st.sidebar.number_input(
        "Slippage (bps)", value=5.0, min_value=0.0, max_value=200.0, step=0.5,
        key=f"{key_prefix}_slip_bps",
    )
    init_cash = st.sidebar.number_input(
        "Initial cash", value=100_000.0, min_value=1.0, step=10_000.0,
        key=f"{key_prefix}_init_cash",
    )
    return CostsConfig(
        fees=fees_bps / 10_000.0,
        slippage=slip_bps / 10_000.0,
        init_cash=init_cash,
    )


# ---------------------------------------------------------------------------


def rerun_from_saved(run_id: int):
    """Re-create a :class:`BacktestResult` from a saved run.

    The DB stores the config, not the portfolio - we replay the backtest
    using the same source/params/costs. The parquet cache means this is
    near-instant on the second call.
    """
    from quant_dashboard.engine import CostsConfig, run_backtest
    from quant_dashboard.strategies import get_strategy

    row = get_store().get_run(run_id)
    data = load_ohlcv(
        row["source"], row["symbol"], row["timeframe"],
        pd.to_datetime(row["start_ts"]).date() if row["start_ts"] else None,
        pd.to_datetime(row["end_ts"]).date() if row["end_ts"] else None,
    )
    strat_cls = get_strategy(row["strategy_name"])
    costs = CostsConfig(
        fees=row["costs"]["fees"],
        slippage=row["costs"]["slippage"],
        init_cash=row["costs"]["init_cash"],
        size=row["costs"].get("size"),
        size_type=row["costs"].get("size_type", "percent"),
    )
    return run_backtest(data, strat_cls(), row["params"], costs=costs), data


def run_picker(label: str = "Backtest run", *, key: str = "run_id") -> int | None:
    """Pick a previously saved backtest run. Returns the run id or None."""
    store = get_store()
    runs = store.list_runs(limit=200)
    if runs.empty:
        st.info("No saved backtest runs yet. Run one on the Backtest page first.")
        return None
    runs["label"] = runs.apply(
        lambda r: f"#{r['id']} - {r['strategy_name']} / {r['symbol']} "
                  f"({r['timeframe']}) - {r['created_at'][:19]}",
        axis=1,
    )
    chosen = st.selectbox(label, runs["label"], key=key)
    return int(runs.loc[runs["label"] == chosen, "id"].iloc[0])
