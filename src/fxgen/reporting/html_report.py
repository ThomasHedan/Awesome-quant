"""Self-contained HTML report for one tree."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from jinja2 import Environment, FileSystemLoader, select_autoescape

TEMPLATES = Path(__file__).parent / "templates"
SPLIT_ORDER = ["train", "validation", "holdout"]
SPLIT_COLORS = {"train": "#2a78d6", "validation": "#eb6834", "holdout": "#1baf7a"}
NEUTRAL = "#52514e"


@dataclass
class ReportData:
    tree_id: str
    rules: str
    tree_json: dict[str, Any]
    depth: int
    n_internal: int
    trades: pd.DataFrame
    metrics: dict[str, dict[str, Any]]
    per_pair: dict[str, dict[str, dict[str, Any]]]
    risk_pct: float
    robustness: dict[str, Any] | None = None
    run_id: str | None = None
    notes: list[str] = field(default_factory=list)


def _layout(fig: go.Figure, title: str, y_title: str) -> go.Figure:
    fig.update_layout(
        title={"text": title, "x": 0, "font": {"size": 15}},
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin={"l": 50, "r": 16, "t": 48, "b": 40},
        height=340,
        legend={"orientation": "h", "y": 1.08, "x": 1, "xanchor": "right"},
        hovermode="x unified",
        font={"family": "Inter, system-ui, sans-serif", "size": 12},
    )
    fig.update_yaxes(title=y_title, gridcolor="rgba(128,128,128,0.18)", zerolinecolor="rgba(128,128,128,0.4)")
    fig.update_xaxes(gridcolor="rgba(128,128,128,0.10)")
    return fig


def equity_figure(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure()
    offset = 0.0
    for split in SPLIT_ORDER:
        part = trades[trades["split"] == split].sort_values("exit_time")
        if part.empty:
            continue
        equity = offset + part["r_multiple"].cumsum()
        fig.add_trace(go.Scatter(
            x=part["exit_time"], y=equity, mode="lines", name=split,
            line={"color": SPLIT_COLORS[split], "width": 2},
            hovertemplate="%{x|%Y-%m-%d}<br>%{y:.1f} R<extra>" + split + "</extra>",
        ))
        offset = float(equity.iloc[-1])
    return _layout(fig, "Equity curve (cumulative R, net of costs)", "R")


def pair_figure(per_pair: dict[str, dict[str, dict[str, Any]]]) -> go.Figure:
    fig = go.Figure()
    for split in SPLIT_ORDER:
        if split not in per_pair:
            continue
        pairs = sorted(per_pair[split])
        fig.add_trace(go.Bar(
            x=pairs, y=[per_pair[split][p]["total_r"] for p in pairs], name=split,
            marker={"color": SPLIT_COLORS[split], "line": {"width": 0}},
            hovertemplate="%{x}: %{y:.1f} R<extra>" + split + "</extra>",
        ))
    fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.08, hovermode="closest")
    return _layout(fig, "Total R per pair", "R")


def distribution_figure(trades: pd.DataFrame) -> go.Figure:
    fig = go.Figure(go.Histogram(
        x=trades["r_multiple"], nbinsx=40, marker={"color": SPLIT_COLORS["train"], "line": {"width": 1, "color": "rgba(255,255,255,0.6)"}},
        hovertemplate="%{x} R: %{y} trades<extra></extra>", name="trades",
    ))
    fig.update_layout(hovermode="closest", showlegend=False)
    return _layout(fig, "Trade distribution (R multiple, all splits)", "trades")


def per_year_table(trades: pd.DataFrame, risk_pct: float) -> list[dict[str, Any]]:
    if trades.empty:
        return []
    grouped = trades.groupby([trades["exit_time"].dt.year.rename("year"), "split"])["r_multiple"]
    df = pd.DataFrame({"n_trades": grouped.size(), "total_r": grouped.sum(),
                       "win_rate": grouped.apply(lambda s: float((s > 0).mean()))}).reset_index()
    df["return_pct"] = df["total_r"] * risk_pct
    return df.to_dict("records")


def _fmt(value: Any) -> str:
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    if isinstance(value, (float, np.floating)):
        if not np.isfinite(value):
            return "inf" if value > 0 else "-"
        return f"{value:,.2f}"
    return "-" if value is None else str(value)


def render_report(data: ReportData, out_path: Path) -> Path:
    """Render the report to ``out_path`` (Plotly loaded from its CDN)."""
    env = Environment(loader=FileSystemLoader(TEMPLATES), autoescape=select_autoescape(["html"]))
    env.filters["fmt"] = _fmt
    figures = [equity_figure(data.trades), pair_figure(data.per_pair), distribution_figure(data.trades)] if not data.trades.empty else []
    first = True
    charts = []
    for fig in figures:
        charts.append(fig.to_html(full_html=False, include_plotlyjs="cdn" if first else False, config={"displaylogo": False}))
        first = False
    metric_keys = ["n_trades", "trades_per_year", "sharpe", "profit_factor", "win_rate", "expectancy_r",
                   "total_r", "return_pct", "max_dd_pct", "worst_day_pct", "avg_bars_held", "long_share"]
    pairs = sorted({p for split in data.per_pair.values() for p in split})
    html = env.get_template("report.html.j2").render(
        d=data,
        splits=[s for s in SPLIT_ORDER if s in data.metrics],
        metric_keys=metric_keys,
        pairs=pairs,
        per_year=per_year_table(data.trades, data.risk_pct),
        charts=charts,
        tree_json=json.dumps(data.tree_json, indent=2),
        colors=SPLIT_COLORS,
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html)
    return out_path
