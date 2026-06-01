"""Multi-Chart page — run one strategy across many symbols / timeframes."""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from _shared import costs_widget, data_source_picker, strategy_picker
from quant_dashboard.data import YFinanceSource
from quant_dashboard.engine import MultiChartError, run_multi_chart

st.set_page_config(page_title="Multi-Chart", layout="wide")
st.title("Multi-Chart")
st.caption(
    "Run a single strategy across multiple instruments or timeframes "
    "and compare key metrics side-by-side."
)

# --- sidebar ---
st.sidebar.header("Strategy")
strategy_cls, params = strategy_picker(key_prefix="mc")
st.sidebar.header("Costs")
costs = costs_widget(key_prefix="mc")
st.sidebar.header("Date range")
start = st.sidebar.text_input("Start", value="2020-01-01", key="mc_start")
end = st.sidebar.text_input("End (blank = today)", value="", key="mc_end") or None

# --- instrument list ---
st.subheader("Instruments")
st.info(
    "Enter one instrument per line in the format `SYMBOL TIMEFRAME` "
    "(e.g. `SPY 1d`). "
    "All instruments use **yfinance** — switch to CCXTSource for crypto."
)
default_instruments = "SPY 1d\nQQQ 1d\nIWM 1d\nGLD 1d\nBTC-USD 1d"
raw = st.text_area("Instruments", value=default_instruments, height=150)

lines = [l.strip() for l in raw.strip().splitlines() if l.strip()]
sources_list = []
parse_errors = []
for line in lines:
    parts = line.split()
    if len(parts) != 2:
        parse_errors.append(f"Cannot parse: {line!r} — expected 'SYMBOL TIMEFRAME'")
        continue
    symbol, timeframe = parts
    sources_list.append((YFinanceSource(), symbol, timeframe))

if parse_errors:
    for e in parse_errors:
        st.warning(e)

if st.button("Run multi-chart", type="primary", disabled=not sources_list):
    if not sources_list:
        st.error("No valid instruments.")
        st.stop()

    strategy = strategy_cls()
    with st.spinner(f"Running {strategy.name} on {len(sources_list)} instruments..."):
        try:
            mc_result = run_multi_chart(
                sources_list, strategy, params, costs=costs,
                start=start, end=end,
            )
        except MultiChartError as exc:
            st.error(str(exc))
            st.stop()

    if mc_result.failed:
        st.warning(
            f"{len(mc_result.failed)} instrument(s) failed: "
            + ", ".join(f"{ir.symbol}/{ir.timeframe}: {ir.error}"
                        for ir in mc_result.failed)
        )

    if not mc_result.successful:
        st.error("All instruments failed. Check data sources / date range.")
        st.stop()

    # --- Summary table ---
    st.subheader("Summary")
    summary = mc_result.summary_df
    st.dataframe(
        summary.style.background_gradient(subset=["sharpe", "cagr_pct"], cmap="RdYlGn")
        .format({
            "total_return_pct": "{:.2f}%",
            "cagr_pct": "{:.2f}%",
            "sharpe": "{:.3f}",
            "sortino": "{:.3f}",
            "max_drawdown_pct": "{:.2f}%",
            "calmar": "{:.3f}",
            "win_rate_pct": "{:.2f}%",
            "profit_factor": "{:.3f}",
        }),
        use_container_width=True,
    )

    # --- Equity curves ---
    st.subheader("Equity curves")
    fig = go.Figure()
    for ir in mc_result.successful:
        eq = ir.result.equity_curve
        eq_norm = eq / eq.iloc[0]  # normalize to 1.0 at start
        fig.add_trace(go.Scatter(
            x=eq_norm.index, y=eq_norm.values,
            mode="lines", name=f"{ir.symbol} {ir.timeframe}",
        ))
    fig.update_layout(
        yaxis_title="Normalized equity (start = 1.0)",
        xaxis_title="Date",
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Sharpe bar chart ---
    st.subheader("Sharpe ratio comparison")
    success_df = summary[summary.get("error", pd.Series(dtype=str)).isna()
                         if "error" in summary.columns else summary.index >= 0]
    if "sharpe" in success_df.columns:
        import pandas as pd  # noqa: PLC0415
        bar_df = success_df.dropna(subset=["sharpe"]).copy()
        bar_df["label"] = bar_df["symbol"] + " " + bar_df["timeframe"]
        bar_fig = go.Figure(go.Bar(
            x=bar_df["label"], y=bar_df["sharpe"],
            marker_color=["green" if v >= 1 else "orange" if v >= 0 else "red"
                          for v in bar_df["sharpe"]],
        ))
        bar_fig.update_layout(yaxis_title="Sharpe", height=350)
        st.plotly_chart(bar_fig, use_container_width=True)
