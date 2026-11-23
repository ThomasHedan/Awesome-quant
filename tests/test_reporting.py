from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fxgen.backtest.runner import run_backtest
from fxgen.errors import TreeError
from fxgen.reporting.html_report import ReportData, render_report
from fxgen.reporting.pine import export_pine
from fxgen.reporting.ranking import RANK_COLUMNS, ranking_frame, write_ranking
from fxgen.tree.node import Action, Leaf, Node, depth, n_internal
from fxgen.tree.serialize import to_dict, to_rules
from fxgen.workspace import Workspace

TREE = Node(
    "D1.dist_ema50>p75",
    Node("H1.macd_hist:rising", Leaf(Action.LONG), Leaf(Action.NO_TRADE)),
    Node("H4.ema20_50:cross_down", Leaf(Action.SHORT), Node("H1.cal:session_london", Leaf(Action.LONG), Leaf(Action.NO_TRADE))),
)


def test_export_pine_structure(ws: Workspace) -> None:
    conds = {c.id: c for c in ws.tree_conditions(TREE)}
    code = export_pine(TREE, conds, ws.features, "abc123")
    lines = code.splitlines()
    assert lines[0] == "//@version=6"
    assert 'indicator("fxgen tree abc123", overlay = true)' in code
    assert 'request.security(syminfo.tickerid, "1D", c0_fn(), lookahead = barmerge.lookahead_off)' in code
    assert 'request.security(syminfo.tickerid, "240", c2_fn(), lookahead = barmerge.lookahead_off)' in code
    assert "bool c1 = c1_fn()" in code
    assert "bool c3 = sessionHour >= 7 and sessionHour < 16" in code
    assert "ta.crossunder(_v, 0.0)" in code and "_v > _v[3]" in code
    assert "[_macd, _signal, _hist] = ta.macd(close, 12, 26, 9)" in code
    assert "int action = (c0 ? (c1 ? 1 : 0) : (c2 ? -1 : (c3 ? 1 : 0)))" in code
    assert 'array.from("TT", "TF", "FT", "FFT", "FFF")' in code
    assert "table." not in code and "lookahead_on" not in code
    assert repr(conds["D1.dist_ema50>p75"].threshold) in code
    with pytest.raises(TreeError):
        export_pine(TREE, {}, ws.features, "x")


def test_ranking_frame_and_files(tmp_path: Path) -> None:
    df = ranking_frame([{"tree_id": "a", "validation_dsr": 0.2}, {"tree_id": "b", "validation_dsr": 0.9}])
    assert list(df.columns) == RANK_COLUMNS and df["tree_id"].tolist() == ["b", "a"]
    csv_path, json_path = write_ranking(df, tmp_path)
    assert csv_path.exists() and json_path.read_text().startswith("[")


def test_render_report(ws: Workspace, tmp_path: Path) -> None:
    cfg = ws.cfg
    trades, metrics, per_pair = [], {}, {}
    for split in ("train", "validation"):
        res = run_backtest(TREE, ws.tree_dataset(TREE, ws.period(split)), ws.rules, cfg.costs, 0.5)
        trades.append(res.trades.assign(split=split))
        metrics[split], per_pair[split] = res.metrics, res.per_pair
    data = ReportData(
        tree_id="abc", rules=to_rules(TREE, ws.describe), tree_json=to_dict(TREE), depth=depth(TREE),
        n_internal=n_internal(TREE), trades=pd.concat(trades, ignore_index=True), metrics=metrics,
        per_pair=per_pair, risk_pct=0.5,
        robustness={"checks": {"dsr_train": False}, "dsr": {}, "walkforward": {"windows": []},
                    "sensitivity": {"base_sharpe": 0.1, "variants": {}}, "montecarlo": {}, "permutation": {},
                    "train_fitness": 0.1, "survives": False},
    )
    path = render_report(data, tmp_path / "r.html")
    html = path.read_text()
    assert "<title>Tree abc</title>" in html and "Equity curve" in html and "fails robustness" in html
    assert "EURUSD" in html and "Per year" in html
