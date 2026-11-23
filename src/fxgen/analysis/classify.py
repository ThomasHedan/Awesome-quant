"""Strategy-type identification: trend following, momentum, breakout, mean reversion.

Two independent readings are combined:

* **Structure** - every condition on the path to a LONG/SHORT leaf is read as
  "with" or "against" the price move it describes. Each feature has a style
  (trend, momentum, breakout, filter) and a polarity (+1 when a high value is
  bullish, -1 when bearish, 0 when it carries no direction, e.g. volatility).
  Buying when ``dist_ema50 > p70`` is trend following; buying when
  ``rsi14 > p20`` is *false* (RSI very low) is mean reversion.
* **Behaviour** - at each entry, was the trade in the direction of the move of
  the previous day / week (``move_1d`` / ``move_5d``)? Trend followers buy
  strength over days; mean reverters buy recent weakness. The payoff profile
  (win rate, average win / average loss) is reported alongside.
"""

from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from fxgen.conditions.builder import Condition
from fxgen.tree.node import Action, Leaf, Node, Tree

STYLES = ("trend", "momentum", "breakout", "filter")
LABELS = ("trend following", "momentum", "breakout", "mean reversion", "mixed")

_RULES: list[tuple[str, str, int]] = [
    # (regex on feature name, style, polarity)
    (r"^(dist_(ema|sma|wma|hma|alma|dema|tema|kama)\d+|(ema|sma|hma)\d+_\d+|slope_\w+|linreg_slope\d+"
     r"|di_diff\d+|macd_line\w*|aroon_osc\d+|dist_supertrend\w+|dist_psar|ichimoku_tk|dist_kijun|dist_cloud"
     r"|cloud_thickness|vortex\d+|trix\d+|dpo\d+|ema_stack|coppock)$", "trend", 1),
    (r"^(rsi\d+|stoch\w*|rsi_signal|williams_r\d+|cci\d+|roc\d+_atr|roc\d+_rank|momentum_rank|macd_hist\w*"
     r"|tsi\w*|cmo\d+|ultimate|fisher\d+|awesome|elder_bull|elder_bear|mfi\d+|cmf\d+|force\d+|obv_flow\d+"
     r"|streak|body_sum\d+|up_bars\d+|prev_bar_return|body_atr|body_ratio|close_location|close_range_pos\d+"
     r"|bb_pctb\d*|keltner_pos|dist_vwma\d+|lower_wick)$", "momentum", 1),
    (r"^upper_wick$", "momentum", -1),
    (r"^(donchian_pos\d+|dist_low\d+|bars_since_low\d+|breakout_up\d+|dist_donchian_mid\d+)$", "breakout", 1),
    (r"^(dist_high\d+|bars_since_high\d+|breakout_down\d+)$", "breakout", -1),
    (r"^(bull_engulfing|bull_pinbar|marubozu_up|three_up|higher_high)$", "momentum", 1),
    (r"^(bear_engulfing|bear_pinbar|marubozu_down|three_down|lower_low)$", "momentum", -1),
]


def feature_profile(name: str) -> tuple[str, int]:
    """(style, polarity) of a feature; volatility, volume level, calendar and neutral patterns are filters."""
    for pattern, style, polarity in _RULES:
        if re.match(pattern, name):
            return style, polarity
    return "filter", 0


def _implied_direction(cond: Condition, state: bool) -> tuple[float, float]:
    """(direction implied by this branch, strength in [0, 1])."""
    style, polarity = feature_profile(cond.feature)
    if polarity == 0 or cond.kind == "calendar":
        return 0.0, 0.0
    if cond.kind == "above":
        if cond.percentile is not None:
            strength = cond.percentile / 100 if state else (100 - cond.percentile) / 100
        else:
            strength = 0.5
        return (polarity if state else -polarity), strength
    if cond.kind == "rising":
        return (polarity if state else -polarity), 0.5
    if not state:
        return 0.0, 0.0
    return (polarity if cond.kind == "cross_up" else -polarity), 0.75


@dataclass
class Classification:
    label: str
    confidence: float
    structure: dict[str, float] = field(default_factory=dict)
    behaviour: dict[str, float] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)
    agreement: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _paths(tree: Tree, prefix: list[tuple[str, bool]] | None = None, path: str = ""):  # type: ignore[no-untyped-def]
    prefix = prefix or []
    if isinstance(tree, Leaf):
        yield path or "root", tree, prefix
        return
    assert isinstance(tree, Node)
    yield from _paths(tree.if_true, [*prefix, (tree.condition, True)], path + "T")
    yield from _paths(tree.if_false, [*prefix, (tree.condition, False)], path + "F")


def structure_scores(tree: Tree, conditions: dict[str, Condition], branch_weights: dict[str, float] | None = None
                     ) -> tuple[dict[str, float], list[str]]:
    """Signed alignment per style (+ = with the move, - = against it) and filter tags."""
    with_move: dict[str, float] = defaultdict(float)
    against: dict[str, float] = defaultdict(float)
    tags: set[str] = set()
    trading = [(p, leaf, conds) for p, leaf, conds in _paths(tree) if leaf.action != Action.NO_TRADE]
    for path, leaf, conds in trading:
        weight = (branch_weights or {}).get(path, 1.0 / max(1, len(trading)))
        for cid, state in conds:
            cond = conditions[cid]
            style, _ = feature_profile(cond.feature)
            if cond.kind == "calendar":
                tags.add("calendar filter")
                continue
            if style == "filter":
                tags.add(f"{cond.family} filter")
                continue
            direction, strength = _implied_direction(cond, state)
            alignment = int(leaf.action) * direction * strength * weight
            if alignment > 0:
                with_move[style] += alignment
            elif alignment < 0:
                against[style] += -alignment
    scores = {
        "trend": with_move["trend"],
        "momentum": with_move["momentum"],
        "breakout": with_move["breakout"],
        "mean_reversion": against["momentum"] + against["breakout"] + against["trend"],
    }
    return scores, sorted(tags)


def behaviour_scores(trades: pd.DataFrame) -> dict[str, float]:
    """Alignment of entries with the prior 1-day / 5-day move and the payoff profile."""
    if trades.empty:
        return {}
    r = trades["r_multiple"].to_numpy(dtype=float)
    wins, losses = r[r > 0], r[r < 0]
    out = {
        "n_trades": float(len(trades)),
        "win_rate": float((r > 0).mean()),
        "payoff_ratio": float(wins.mean() / -losses.mean()) if len(wins) and len(losses) else float("nan"),
        "avg_bars_held": float(trades["bars_held"].mean()),
        "long_share": float((trades["direction"] > 0).mean()),
    }
    for col in ("move_1d", "move_5d"):
        if col in trades and trades[col].notna().any():
            m = trades[col].to_numpy(dtype=float)
            d = trades["direction"].to_numpy(dtype=float)
            ok = ~np.isnan(m)
            out[f"with_{col}"] = float(np.mean(np.sign(m[ok]) * d[ok]))
    return out


def classify(tree: Tree, conditions: dict[str, Condition], trades: pd.DataFrame | None = None) -> Classification:
    """Label a tree. ``trades`` (any split, ideally train + validation) adds the behavioural reading."""
    weights = None
    if trades is not None and not trades.empty and "branch" in trades:
        counts = trades["branch"].value_counts(normalize=True)
        weights = {str(k): float(v) for k, v in counts.items()}
    scores, tags = structure_scores(tree, conditions, weights)
    total = sum(scores.values())
    behaviour = behaviour_scores(trades) if trades is not None else {}

    if total <= 0:
        struct_label, struct_conf = "mixed", 0.0
    else:
        key = max(scores, key=lambda k: scores[k])
        struct_label = {"trend": "trend following", "momentum": "momentum", "breakout": "breakout",
                        "mean_reversion": "mean reversion"}[key]
        struct_conf = scores[key] / total
        if struct_conf < 0.45:
            struct_label = "mixed"

    behav_label = ""
    with_5d, with_1d = behaviour.get("with_move_5d", 0.0), behaviour.get("with_move_1d", 0.0)
    if "with_move_5d" in behaviour or "with_move_1d" in behaviour:
        if with_5d > 0.15 or with_1d > 0.15:
            behav_label = "with the move"
        elif with_5d < -0.15 or with_1d < -0.15:
            behav_label = "against the move"
        else:
            behav_label = "no directional bias"

    label, confidence, agreement = struct_label, struct_conf, "structure only"
    if behav_label:
        follows = struct_label in ("trend following", "momentum", "breakout")
        if (follows and behav_label == "with the move") or (struct_label == "mean reversion" and behav_label == "against the move"):
            agreement, confidence = "structure and behaviour agree", min(1.0, struct_conf + 0.2)
        elif behav_label == "no directional bias":
            agreement, confidence = "behaviour neutral", struct_conf * 0.8
        else:
            agreement, confidence = f"behaviour disagrees ({behav_label})", struct_conf * 0.5
            if struct_label == "mixed" or confidence < 0.3:
                label = "mean reversion" if behav_label == "against the move" else "trend following"
                agreement += "; label taken from behaviour"
    if behaviour.get("long_share", 0.5) > 0.8:
        tags.append("long bias")
    elif behaviour.get("long_share", 0.5) < 0.2:
        tags.append("short bias")
    return Classification(
        label=label,
        confidence=round(float(confidence), 3),
        structure={k: round(v, 3) for k, v in scores.items()},
        behaviour={k: round(v, 3) for k, v in behaviour.items()},
        tags=tags,
        agreement=agreement,
    )
