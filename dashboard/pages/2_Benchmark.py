"""Benchmark page — strategy vs. buy-and-hold or external series."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from _shared import load_ohlcv, rerun_from_saved, run_picker
from quant_dashboard.engine import (
    BenchmarkError,
    build_buy_and_hold_returns,
    compute_benchmark_comparison,
)

st.set_page_config(page_title="Benchmark", layout="wide")
st.title("Benchmark")

run_id = run_picker("Backtest run to analyze", key="bench_run")
if run_id is None:
    st.stop()

st.sidebar.header("Benchmark")
mode = st.sidebar.radio(
    "Benchmark series",
    ["Buy & hold same instrument", "External symbol (yfinance)"],
    key="bench_mode",
)

bench_returns = None
bench_name = None
if mode == "External symbol (yfinance)":
    bench_symbol = st.sidebar.text_input("Benchmark symbol", value="SPY", key="bench_sym")
    bench_timeframe = st.sidebar.selectbox(
        "Benchmark timeframe", ["1d"], index=0, key="bench_tf",
        help="External benchmark must match (or be coarser than) the strategy timeframe.",
    )
else:
    bench_symbol = None
    bench_timeframe = None

rf_pct = st.sidebar.number_input(
    "Risk-free rate (annual %)", value=0.0, min_value=0.0, max_value=20.0, step=0.1,
    key="bench_rf",
)

with st.spinner("Replaying backtest..."):
    result, data = rerun_from_saved(run_id)

if mode == "External symbol (yfinance)":
    with st.spinner(f"Fetching {bench_symbol}..."):
        bench_data = load_ohlcv(
            "yfinance", bench_symbol, bench_timeframe,
            data.df.index[0].date(), data.df.index[-1].date(),
        )
    bench_returns = build_buy_and_hold_returns(bench_data)
    bench_name = bench_symbol

try:
    comp = compute_benchmark_comparison(
        result, benchmark_returns=bench_returns,
        benchmark_name=bench_name, risk_free_annual=rf_pct / 100.0,
    )
except BenchmarkError as exc:
    st.error(str(exc))
    st.stop()

st.subheader(f"Strategy vs. {comp.benchmark_name}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Alpha (annual)", f"{comp.alpha:.2%}")
c1.metric("Beta", f"{comp.beta:.3f}")
c2.metric("Information ratio", f"{comp.information_ratio:.2f}")
c2.metric("Correlation", f"{comp.correlation:.3f}")
c3.metric("Strategy total return", f"{comp.strategy_total_return:.2%}")
c3.metric("Benchmark total return", f"{comp.benchmark_total_return:.2%}")
c4.metric("Excess total return", f"{comp.excess_total_return:.2%}")
c4.metric("R-squared", f"{comp.r_squared:.3f}")

# --- Cumulative equity overlay -----------------------------------------------
strat_eq = (1.0 + comp.strategy_returns).cumprod()
bench_eq = (1.0 + comp.benchmark_returns).cumprod()
fig = go.Figure()
fig.add_trace(go.Scatter(x=strat_eq.index, y=strat_eq.values, name="Strategy"))
fig.add_trace(go.Scatter(x=bench_eq.index, y=bench_eq.values, name=comp.benchmark_name,
                          line=dict(dash="dot")))
fig.update_layout(
    height=380, margin=dict(l=10, r=10, t=20, b=10),
    yaxis_title="Cumulative return (base 1)", legend=dict(orientation="h"),
)
st.subheader("Cumulative return")
st.plotly_chart(fig, use_container_width=True)

# --- Relative equity --------------------------------------------------------
st.subheader("Relative equity (strategy / benchmark)")
fig2 = go.Figure()
fig2.add_trace(go.Scatter(x=comp.relative_equity.index, y=comp.relative_equity.values,
                           name="Strategy / Benchmark"))
fig2.add_hline(y=1.0, line_dash="dot", line_color="grey")
fig2.update_layout(height=260, margin=dict(l=10, r=10, t=20, b=10))
st.plotly_chart(fig2, use_container_width=True)

# --- Excess returns histogram ----------------------------------------------
st.subheader("Excess returns distribution")
fig3 = go.Figure()
fig3.add_trace(go.Histogram(x=comp.excess_returns.values, nbinsx=50,
                             name="Excess returns"))
fig3.update_layout(
    height=260, margin=dict(l=10, r=10, t=20, b=10),
    xaxis_title="Excess return per bar",
)
st.plotly_chart(fig3, use_container_width=True)

with st.expander("Comparison summary"):
    st.json(comp.to_dict())
