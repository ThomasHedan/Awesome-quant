"""Streamlit entry point — multipage app.

Launch with: ``uv run streamlit run dashboard/app.py``

Streamlit auto-discovers files in ``pages/`` and renders them as sidebar
links. This top-level file is the landing page.
"""

from __future__ import annotations

import streamlit as st

from _shared import DB_PATH, get_store
from quant_dashboard.strategies import available_strategies

st.set_page_config(
    page_title="Quant Backtesting Dashboard",
    page_icon=":chart_with_upwards_trend:",
    layout="wide",
)

st.title("Quant Backtesting Dashboard")
st.write(
    "Local-first backtesting on top of `vectorbt` + `quantstats`. "
    "Four analyses, all wired to the same data layer."
)

store = get_store()
recent = store.list_runs(limit=10)
opts = store.list_optimizations(limit=10)

col1, col2, col3 = st.columns(3)
col1.metric("Saved backtests", len(store.list_runs(limit=100_000)))
col2.metric("Saved optimizations", len(store.list_optimizations(limit=100_000)))
col3.metric("Registered strategies", len(available_strategies()))

st.caption(f"Database: `{DB_PATH}`")

st.markdown("### Pages")
st.markdown(
    "- **Backtest** - run a strategy on any data source and save the result.\n"
    "- **Benchmark** - compare a saved backtest to buy-and-hold or any series.\n"
    "- **MonteCarlo** - bootstrap / trade-shuffle to test whether the edge is real.\n"
    "- **Optimization** - exhaustive grid search with heatmaps and stability scoring."
)

st.markdown("### Recent backtests")
if recent.empty:
    st.info("No runs yet. Head to the Backtest page to create your first.")
else:
    st.dataframe(recent, use_container_width=True, hide_index=True)

st.markdown("### Recent optimizations")
if opts.empty:
    st.info("No optimizations yet.")
else:
    st.dataframe(opts, use_container_width=True, hide_index=True)
