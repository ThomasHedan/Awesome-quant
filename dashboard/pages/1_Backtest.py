"""Backtest page — fetch data, run a strategy, save + render results."""

from __future__ import annotations

import json
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

from dashboard._shared import (
    costs_widget,
    data_source_picker,
    get_store,
    load_ohlcv,
    strategy_picker,
)
from quant_dashboard.engine import (
    EngineError,
    compute_metrics,
    run_backtest,
    tearsheet_html,
)

st.set_page_config(page_title="Backtest", layout="wide")
st.title("Backtest")

st.sidebar.header("Data")
source, symbol, timeframe, start, end = data_source_picker(key_prefix="bt")
st.sidebar.header("Strategy")
strategy_cls, params = strategy_picker(key_prefix="bt")
st.sidebar.header("Costs")
costs = costs_widget(key_prefix="bt")

notes = st.text_input("Notes (optional)", value="")

if st.button("Run backtest", type="primary"):
    with st.spinner(f"Loading {symbol} from {source}..."):
        data = load_ohlcv(source, symbol, timeframe, start, end)
    st.success(f"Loaded {len(data.df):,} bars of {symbol} ({timeframe}).")

    try:
        with st.spinner("Running backtest..."):
            result = run_backtest(data, strategy_cls(), params, costs=costs)
            metrics = compute_metrics(result)
    except EngineError as exc:
        st.error(str(exc))
        st.stop()

    run_id = get_store().save_backtest(result, metrics, notes=notes)
    st.session_state["last_run_id"] = run_id
    st.success(f"Saved as run #{run_id}.")

    # --- Metrics table ---------------------------------------------------
    st.subheader("Metrics")
    md = metrics.to_dict()
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total return", f"{md['total_return']:.2%}")
    c1.metric("CAGR", f"{md['cagr']:.2%}")
    c2.metric("Sharpe", f"{md['sharpe']:.2f}")
    c2.metric("Sortino", f"{md['sortino']:.2f}")
    c3.metric("Max drawdown", f"{md['max_drawdown']:.2%}")
    c3.metric("Calmar", f"{md['calmar']:.2f}")
    c4.metric("Trades", f"{md['num_trades']}")
    c4.metric("Win rate", f"{md['win_rate']:.2%}" if md["num_trades"] else "n/a")

    with st.expander("All metrics"):
        st.json(md)

    # --- Equity curve ----------------------------------------------------
    st.subheader("Equity curve")
    eq = result.equity_curve
    bh_close = data.df["close"].copy()
    if bh_close.index.tz is not None:
        bh_close = bh_close.tz_localize(None)
    bh = bh_close / bh_close.iloc[0] * costs.init_cash

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=eq.index, y=eq.values, name="Strategy", line=dict(width=2)))
    fig.add_trace(go.Scatter(x=bh.index, y=bh.values, name="Buy & Hold",
                              line=dict(width=1, dash="dot")))
    fig.update_layout(
        height=420, margin=dict(l=10, r=10, t=20, b=10),
        legend=dict(orientation="h"),
        yaxis_title="Equity", xaxis_title="",
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Drawdowns -------------------------------------------------------
    running_max = eq.cummax()
    dd = eq / running_max - 1.0
    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(x=dd.index, y=dd.values, name="Drawdown",
                                 fill="tozeroy", line=dict(width=1)))
    fig_dd.update_layout(
        height=200, margin=dict(l=10, r=10, t=20, b=10),
        yaxis_title="Drawdown", yaxis_tickformat=".0%",
    )
    st.plotly_chart(fig_dd, use_container_width=True)

    # --- Tearsheet download ---------------------------------------------
    with st.expander("QuantStats tearsheet"):
        if st.button("Generate tearsheet HTML"):
            out_dir = Path(".cache/quant-dashboard/tearsheets")
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"run_{run_id}.html"
            with st.spinner("Rendering tearsheet..."):
                tearsheet_html(result, out_path)
            st.success(f"Wrote {out_path}")
            st.download_button(
                "Download tearsheet",
                data=out_path.read_bytes(),
                file_name=out_path.name,
                mime="text/html",
            )

    # --- Run config (for reproducibility) -------------------------------
    with st.expander("Run config"):
        st.code(json.dumps({
            "run_id": run_id,
            "source": data.source, "symbol": data.spec.symbol,
            "timeframe": data.timeframe,
            "strategy": result.strategy_name, "params": params,
            "costs": {"fees": costs.fees, "slippage": costs.slippage,
                      "init_cash": costs.init_cash},
        }, indent=2, default=str), language="json")
