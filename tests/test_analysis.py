from __future__ import annotations

import numpy as np
import pandas as pd

from fxgen.analysis.classify import behaviour_scores, classify, feature_profile
from fxgen.conditions.builder import Condition, parse_condition_id
from fxgen.indicators.base import all_features
from fxgen.tree.node import Action, Leaf, Node

FEATURES = all_features()
L, S, N = Leaf(Action.LONG), Leaf(Action.SHORT), Leaf(Action.NO_TRADE)


def _conds(*ids: str) -> dict[str, Condition]:
    out = {}
    for cid in ids:
        c = parse_condition_id(cid, FEATURES)
        if c.kind == "above" and c.threshold is None:
            c = Condition(**{**c.to_dict(), "threshold": 0.0})
        out[cid] = c
    return out


def test_every_feature_has_a_profile() -> None:
    assert feature_profile("dist_hma21") == ("trend", 1)
    assert feature_profile("rsi14") == ("momentum", 1)
    assert feature_profile("dist_high50") == ("breakout", -1)
    assert feature_profile("atr_rank") == ("filter", 0)
    assert feature_profile("bear_engulfing") == ("momentum", -1)
    directional = [f.name for f in FEATURES if feature_profile(f.name)[0] != "filter"]
    assert len(directional) > 120


def test_trend_following_tree() -> None:
    tree = Node("D1.dist_ema50>p70", L, Node("D1.dist_ema50>p30", N, S))
    c = classify(tree, _conds("D1.dist_ema50>p70", "D1.dist_ema50>p30"))
    assert c.label == "trend following" and c.confidence > 0.9


def test_mean_reversion_tree() -> None:
    tree = Node("H1.rsi14>p20", Node("H1.rsi14>p80", S, N), L)
    c = classify(tree, _conds("H1.rsi14>p20", "H1.rsi14>p80"))
    assert c.label == "mean reversion" and c.structure["mean_reversion"] > 0


def test_breakout_and_filters() -> None:
    tree = Node("H4.breakout_up20:cross_up", Node("H1.atr_rank>p50", L, N), N)
    c = classify(tree, _conds("H4.breakout_up20:cross_up", "H1.atr_rank>p50"))
    assert c.label == "breakout" and "volatility filter" in c.tags
    cal = Node("H1.cal:session_london", L, S)
    assert classify(cal, _conds("H1.cal:session_london")).label == "mixed"


def test_behaviour_reading() -> None:
    rng = np.random.default_rng(0)
    move = rng.normal(0, 1, 200)
    trades = pd.DataFrame({"direction": np.sign(move).astype(int), "move_1d": move, "move_5d": move,
                           "r_multiple": rng.normal(0.1, 1, 200), "bars_held": 10, "branch": "T"})
    b = behaviour_scores(trades)
    assert b["with_move_5d"] == 1.0 and 0 < b["win_rate"] < 1
    tree = Node("H1.rsi14>p20", Node("H1.rsi14>p80", S, N), L)
    c = classify(tree, _conds("H1.rsi14>p20", "H1.rsi14>p80"), trades)
    assert "disagrees" in c.agreement
    trades["direction"] = -trades["direction"]
    assert "agree" in classify(tree, _conds("H1.rsi14>p20", "H1.rsi14>p80"), trades).agreement
