"""Optimization page — exhaustive grid search with heatmaps and stability."""

from __future__ import annotations

import plotly.express as px
import streamlit as st

from _shared import (
    costs_widget,
    data_source_picker,
    get_store,
    load_ohlcv,
    strategy_picker,
)
from quant_dashboard.engine import (
    GridSearchError,
    estimate_combo_count,
    estimate_ram_bytes,
    heatmap,
    parameter_stability,
    run_grid_search,
)
from quant_dashboard.engine.optimizer import _METRIC_DIRECTION

st.set_page_config(page_title="Optimization", layout="wide")
st.title("Parameter optimization (grid search)")

st.sidebar.header("Data")
source, symbol, timeframe, start, end = data_source_picker(key_prefix="opt")
st.sidebar.header("Strategy")
strategy_cls, _ = strategy_picker(key_prefix="opt")
st.sidebar.header("Costs")
costs = costs_widget(key_prefix="opt")

st.sidebar.header("Optimization")
metric = st.sidebar.selectbox(
    "Ranking metric", sorted(_METRIC_DIRECTION), index=sorted(_METRIC_DIRECTION).index("sharpe"),
    key="opt_metric",
)
max_combos = st.sidebar.number_input(
    "Max combinations", value=10_000, min_value=10, max_value=500_000, step=1_000,
    key="opt_max",
)
force = st.sidebar.checkbox("Force run even if guardrails fail", value=False, key="opt_force")

# --- Grid editor -----------------------------------------------------------
st.subheader(f"Grid for {strategy_cls.name}")
grids: dict[str, list] = {}
cols = st.columns(min(3, len(strategy_cls.param_specs())))
for i, spec in enumerate(strategy_cls.param_specs()):
    with cols[i % len(cols)]:
        st.markdown(f"**{spec.name}** ({spec.kind})")
        st.caption(spec.description)
        if spec.kind == "choice":
            choices = list(spec.choices or ())
            picked = st.multiselect(
                "Values", choices, default=choices, key=f"opt_grid_{spec.name}",
            )
            grids[spec.name] = picked
        else:
            low = st.number_input(
                "low", value=float(spec.low if spec.low is not None else 0.0),
                step=float(spec.step or 1.0), key=f"opt_low_{spec.name}",
            )
            high = st.number_input(
                "high", value=float(spec.high if spec.high is not None else low * 5),
                step=float(spec.step or 1.0), key=f"opt_high_{spec.name}",
            )
            step = st.number_input(
                "step", value=float(spec.step or 1.0), min_value=1e-9,
                step=float(spec.step or 1.0), key=f"opt_step_{spec.name}",
            )
            n = max(0, int(round((high - low) / step)) + 1)
            values = [low + i * step for i in range(n)]
            if spec.kind == "int":
                values = [int(round(v)) for v in values]
            grids[spec.name] = values

# --- Combo count + guardrail preview --------------------------------------
n_combos = estimate_combo_count(grids) if all(grids.values()) else 0
ram_gb = estimate_ram_bytes(n_combos, 5_000) / 1024**3
c1, c2 = st.columns(2)
c1.metric("Combinations", f"{n_combos:,}")
c2.metric("Est. RAM (per 5,000 bars)", f"{ram_gb:.2f} GiB")
if n_combos > max_combos and not force:
    st.warning(
        f"Grid exceeds max_combos={max_combos:,}. Tighten the grid or tick "
        "'Force run'."
    )

if st.button("Run grid search", type="primary"):
    with st.spinner(f"Loading {symbol}..."):
        data = load_ohlcv(source, symbol, timeframe, start, end)
    try:
        with st.spinner(f"Running {n_combos:,} backtests..."):
            opt = run_grid_search(
                data, strategy_cls, grids, metric=metric,
                costs=costs, max_combos=int(max_combos), force=force,
            )
    except GridSearchError as exc:
        st.error(str(exc))
        st.stop()

    opt_id = get_store().save_optimization(
        opt, source=data.source, symbol=data.spec.symbol,
        timeframe=data.timeframe,
        start=data.df.index[0], end=data.df.index[-1], grids=grids,
    )
    st.success(
        f"Saved optimization #{opt_id} - {opt.n_combos} combos, "
        f"{len(opt.failed_runs)} failed."
    )

    st.subheader("Best parameters")
    bp_cols = st.columns(len(opt.best_params) + 1)
    for i, (k, v) in enumerate(opt.best_params.items()):
        bp_cols[i].metric(k, str(v))
    bp_cols[-1].metric(f"Best {metric}", f"{opt.best_metric_value:.4f}")

    # --- Top-N table -----------------------------------------------------
    st.subheader("Top 20")
    st.dataframe(opt.rankings.head(20), use_container_width=True, hide_index=True)

    # --- Heatmap ---------------------------------------------------------
    param_names = list(opt.param_names)
    if len(param_names) >= 2:
        st.subheader("Heatmap")
        h1, h2 = st.columns(2)
        x_param = h1.selectbox("X axis", param_names, index=0, key="opt_hx")
        y_param = h2.selectbox("Y axis", param_names, index=1, key="opt_hy")
        if x_param != y_param:
            pivot = heatmap(opt, x_param=x_param, y_param=y_param)
            fig = px.imshow(
                pivot, aspect="auto", origin="lower",
                color_continuous_scale="RdYlGn",
                labels=dict(x=x_param, y=y_param, color=metric),
            )
            fig.update_layout(height=480, margin=dict(l=10, r=10, t=20, b=10))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Pick two different parameters for the heatmap axes.")

    # --- Stability -------------------------------------------------------
    st.subheader("Parameter stability")
    st.caption(
        "`stability_score` near 1.0 = neighbours perform like the combo itself. "
        "`is_isolated_peak = True` flags suspiciously sharp optima (likely overfit)."
    )
    stab = parameter_stability(opt)
    peak_count = int(stab["is_isolated_peak"].sum())
    if peak_count > 0:
        st.warning(f"{peak_count} combo(s) flagged as isolated peaks - inspect before trusting.")
    show_cols = [c for c in stab.columns if c.startswith("param_")] + [
        metric, "neighborhood_mean", "stability_score", "is_isolated_peak",
    ]
    st.dataframe(stab[show_cols].head(50), use_container_width=True, hide_index=True)
