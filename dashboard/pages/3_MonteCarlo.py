"""Monte Carlo page — edge analysis via bootstrap or trade-shuffle."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from _shared import get_store, rerun_from_saved, run_picker
from quant_dashboard.engine import MonteCarloError, run_monte_carlo

st.set_page_config(page_title="Monte Carlo", layout="wide")
st.title("Monte Carlo edge analysis")

run_id = run_picker("Backtest run to analyze", key="mc_run")
if run_id is None:
    st.stop()

st.sidebar.header("Monte Carlo")
method = st.sidebar.radio(
    "Method", ["bootstrap", "shuffle"], key="mc_method",
    help=(
        "bootstrap = resample per-bar returns; tests whether this Sharpe "
        "could come from the distribution by chance. shuffle = reorder "
        "realized trades; tests whether a different trade sequence would "
        "have looked the same."
    ),
)
n_sims = st.sidebar.slider(
    "Simulations", min_value=100, max_value=5000, value=1000, step=100, key="mc_n",
)
seed = st.sidebar.number_input("Seed", value=42, step=1, key="mc_seed")
goal_pct = st.sidebar.number_input(
    "Return goal (%)", value=20.0, min_value=1.0, max_value=500.0, step=1.0,
    key="mc_goal",
)
ruin_pct = st.sidebar.number_input(
    "Ruin drawdown threshold (%)", value=50.0, min_value=5.0, max_value=99.0, step=1.0,
    key="mc_ruin",
)

save = st.sidebar.checkbox("Save result to DB", value=True, key="mc_save")

with st.spinner("Replaying backtest..."):
    result, _ = rerun_from_saved(run_id)

try:
    with st.spinner(f"Running {n_sims} simulations..."):
        mc = run_monte_carlo(
            result, method=method, n_simulations=int(n_sims), seed=int(seed),
            goal_return=goal_pct / 100.0, ruin_threshold=ruin_pct / 100.0,
        )
except MonteCarloError as exc:
    st.error(str(exc))
    st.stop()

if save:
    get_store().save_monte_carlo(run_id, mc)

# --- Edge cards -------------------------------------------------------------
c1, c2, c3, c4 = st.columns(4)
c1.metric("Realized total return", f"{mc.realized_total_return:.2%}")
c1.metric("Return percentile",
          f"{mc.return_percentile:.0f}" if np.isfinite(mc.return_percentile) else "n/a",
          help="Percentage of simulations the realized result beats. >95 = strong; <50 = weaker than median.")
c2.metric("Realized Sharpe",
          f"{mc.realized_sharpe:.2f}" if np.isfinite(mc.realized_sharpe) else "n/a")
c2.metric("Sharpe percentile",
          f"{mc.sharpe_percentile:.0f}" if np.isfinite(mc.sharpe_percentile) else "n/a")
c3.metric("Prob >= goal", f"{mc.prob_goal:.1%}",
          help=f"Fraction of simulations terminating at >= +{goal_pct:.0f}%.")
c3.metric("Prob ruin", f"{mc.prob_ruin:.1%}",
          help=f"Fraction of simulations whose max drawdown reaches -{ruin_pct:.0f}%.")
c4.metric("Realized max drawdown", f"{mc.realized_max_drawdown:.2%}")
c4.metric("Simulations", f"{mc.n_simulations}")

# --- CI fan chart ----------------------------------------------------------
st.subheader("Equity-curve confidence fan")
ci = mc.equity_curves_ci

# x-axis: use the realized equity's timestamps for bootstrap (same length);
# fall back to a step index for shuffle (per-trade, not per-bar).
if method == "bootstrap":
    x = mc.realized_equity_normalized.index
else:
    x = list(range(len(ci)))

fig = go.Figure()
# Shaded 5-95 band.
fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1],
                          y=list(ci["p95"]) + list(ci["p05"])[::-1],
                          fill="toself", fillcolor="rgba(31,119,180,0.15)",
                          line=dict(color="rgba(0,0,0,0)"), name="5-95 band",
                          hoverinfo="skip"))
# 25-75 band.
fig.add_trace(go.Scatter(x=list(x) + list(x)[::-1],
                          y=list(ci["p75"]) + list(ci["p25"])[::-1],
                          fill="toself", fillcolor="rgba(31,119,180,0.30)",
                          line=dict(color="rgba(0,0,0,0)"), name="25-75 band",
                          hoverinfo="skip"))
fig.add_trace(go.Scatter(x=x, y=ci["p50"].values, name="Median", line=dict(width=1.5)))
if method == "bootstrap":
    realized = mc.realized_equity_normalized
    fig.add_trace(go.Scatter(x=realized.index, y=realized.values,
                              name="Realized", line=dict(width=2.5, color="firebrick")))
fig.update_layout(
    height=420, margin=dict(l=10, r=10, t=20, b=10),
    yaxis_title="Equity (base 1)", legend=dict(orientation="h"),
)
st.plotly_chart(fig, use_container_width=True)

# --- Distributions ---------------------------------------------------------
st.subheader("Distributions")
d1, d2, d3 = st.columns(3)

with d1:
    fig_r = go.Figure()
    fig_r.add_trace(go.Histogram(x=mc.terminal_returns, nbinsx=40, name="Terminal returns"))
    fig_r.add_vline(x=mc.realized_total_return, line_color="firebrick",
                     annotation_text="Realized", annotation_position="top")
    fig_r.update_layout(height=300, title="Terminal returns",
                         margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_r, use_container_width=True)

with d2:
    fig_s = go.Figure()
    fig_s.add_trace(go.Histogram(x=mc.sharpes[np.isfinite(mc.sharpes)], nbinsx=40,
                                  name="Sharpe"))
    if np.isfinite(mc.realized_sharpe):
        fig_s.add_vline(x=mc.realized_sharpe, line_color="firebrick",
                         annotation_text="Realized", annotation_position="top")
    fig_s.update_layout(height=300, title="Sharpe",
                         margin=dict(l=10, r=10, t=40, b=10))
    st.plotly_chart(fig_s, use_container_width=True)

with d3:
    fig_d = go.Figure()
    fig_d.add_trace(go.Histogram(x=mc.max_drawdowns, nbinsx=40, name="Max drawdown"))
    fig_d.add_vline(x=mc.realized_max_drawdown, line_color="firebrick",
                     annotation_text="Realized", annotation_position="top")
    fig_d.update_layout(height=300, title="Max drawdown",
                         margin=dict(l=10, r=10, t=40, b=10),
                         xaxis_tickformat=".0%")
    st.plotly_chart(fig_d, use_container_width=True)

with st.expander("Run summary"):
    st.json(mc.to_summary())
