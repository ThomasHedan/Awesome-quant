"""HTML report of a portfolio of strategies."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

from fxgen.reporting.html_report import TEMPLATES, _fmt, _layout

SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
PORTFOLIO_COLOR = "#7b7a74"


def _equity_figure(series: dict[str, dict[str, pd.Series]], weights: np.ndarray, risk: float) -> go.Figure:
    fig = go.Figure()
    total: pd.Series | None = None
    for i, ((tid, splits), w) in enumerate(zip(series.items(), weights)):
        daily = pd.concat([splits[s] for s in ("train", "validation", "holdout") if s in splits]) * w * risk
        total = daily if total is None else total.add(daily, fill_value=0.0)
        fig.add_trace(go.Scatter(x=daily.index, y=daily.cumsum(), mode="lines", name=tid,
                                 line={"color": SERIES_COLORS[i % len(SERIES_COLORS)], "width": 1.5}, opacity=0.8))
    if total is not None:
        fig.add_trace(go.Scatter(x=total.index, y=total.cumsum(), mode="lines", name="portfolio",
                                 line={"color": PORTFOLIO_COLOR, "width": 2.5}))
    return _layout(fig, "Cumulative return % (members and portfolio)", "%")


def _correlation_figure(corr: dict[str, dict[str, float]]) -> go.Figure:
    names = list(corr)
    z = [[corr[a][b] for b in names] for a in names]
    fig = go.Figure(go.Heatmap(z=z, x=names, y=names, zmin=-1, zmax=1, colorscale=[[0, "#2a78d6"], [0.5, "#f3f2ef"], [1, "#e34948"]],
                               text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}",
                               hovertemplate="%{x} / %{y}: %{z:.2f}<extra></extra>"))
    fig.update_layout(hovermode="closest")
    return _layout(fig, "Correlation of daily R (train + validation)", "")


def render_portfolio(portfolio: dict[str, Any], risk_pct: float, confidence: float, out_path: Path) -> Path:
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.filters["fmt"] = _fmt
    figures = [_equity_figure(portfolio["_series"], portfolio["_weights"], risk_pct)]
    if len(portfolio["members"]) > 1:
        figures.append(_correlation_figure(portfolio["result"]["correlation"]))
    charts = [f.to_html(full_html=False, include_plotlyjs=i == 0, config={"displaylogo": False}) for i, f in enumerate(figures)]
    html = env.get_template("portfolio.html.j2").render(
        p=portfolio, r=portfolio["result"], charts=charts, conf=confidence,
        metric_keys=["sharpe", "annual_return_pct", "return_pct", "max_dd_pct", "worst_day_pct",
                     "positive_months_share", "mean_member_sharpe", "diversification_ratio"],
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
