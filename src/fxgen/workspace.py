"""Everything derived from the raw data for one configuration.

A ``Workspace`` loads the universe, computes the aligned features, fits the
condition thresholds on the training split only and filters redundant
conditions. It then hands out split datasets; the holdout requires an
explicit ``allow_holdout=True`` that only the final holdout stage passes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

from fxgen.backtest.engine import ExitRules
from fxgen.conditions.builder import Condition, generate_candidates, parse_condition_id
from fxgen.conditions.redundancy import RedundancyReport, filter_conditions
from fxgen.conditions.thresholds import fit_thresholds
from fxgen.config import Config
from fxgen.data.loader import PairData, load_universe
from fxgen.data.splits import Period, Split, split_period
from fxgen.errors import TreeError
from fxgen.indicators.base import Feature, all_features
from fxgen.market import Dataset, PairFrame, build_pair_frame, is_available, make_dataset
from fxgen.tree.node import Tree, conditions_used

logger = logging.getLogger(__name__)


@dataclass
class Workspace:
    cfg: Config
    frames: dict[str, PairFrame]
    features: list[Feature]
    conditions: list[Condition]
    redundancy: RedundancyReport
    _extra: dict[str, Condition] = field(default_factory=dict)

    @property
    def train(self) -> Period:
        return split_period(self.cfg.splits, "train")

    @property
    def rules(self) -> ExitRules:
        return ExitRules.from_config(self.cfg.backtest)

    def period(self, split: Split, allow_holdout: bool = False) -> Period:
        return split_period(self.cfg.splits, split, allow_holdout=allow_holdout)

    def dataset(self, period: Period, conditions: list[Condition] | None = None) -> Dataset:
        return make_dataset(self.frames, conditions if conditions is not None else self.conditions, period)

    def split_dataset(self, split: Split, allow_holdout: bool = False) -> Dataset:
        return self.dataset(self.period(split, allow_holdout))

    def condition(self, cid: str) -> Condition:
        """Fitted condition by id, parsing and fitting it on train if not in the library."""
        for cond in self.conditions:
            if cond.id == cid:
                return cond
        if cid not in self._extra:
            parsed = parse_condition_id(cid, self.features)
            if not all(is_available(f, parsed) for f in self.frames.values()):
                raise TreeError(f"condition {cid!r} cannot be evaluated on this data (missing feature or volume)")
            self._extra[cid] = fit_thresholds(self.frames, [parsed], self.train)[0]
        return self._extra[cid]

    def tree_conditions(self, tree: Tree) -> list[Condition]:
        return [self.condition(cid) for cid in dict.fromkeys(conditions_used(tree))]

    def tree_dataset(self, tree: Tree, period: Period) -> Dataset:
        """Small dataset holding only the conditions a tree uses."""
        return self.dataset(period, self.tree_conditions(tree))

    def describe(self, cid: str) -> str:
        try:
            return self.condition(cid).describe()
        except TreeError:
            return cid


def build_frames(cfg: Config, universe: dict[str, PairData], features: list[Feature]) -> dict[str, PairFrame]:
    return {
        pair: build_pair_frame(
            data,
            features,
            cfg.data.execution_timeframe,
            cfg.data.context_timeframes,
            cfg.backtest.atr_period,
            atr_timeframe=cfg.backtest.atr_timeframe,
        )
        for pair, data in universe.items()
    }


def build_library(cfg: Config, frames: dict[str, PairFrame], features: list[Feature]) -> RedundancyReport:
    """Generate candidates, fit thresholds on train, drop inactive and redundant conditions."""
    train = split_period(cfg.splits, "train")
    timeframes = [cfg.data.execution_timeframe, *cfg.data.context_timeframes]
    candidates = generate_candidates(features, timeframes, cfg.conditions.percentiles)
    fitted = fit_thresholds(frames, candidates, train)
    return filter_conditions(
        frames,
        fitted,
        train,
        threshold=cfg.conditions.redundancy_threshold,
        min_rate=cfg.conditions.min_activation_rate,
        max_rate=cfg.conditions.max_activation_rate,
        seed=cfg.seed,
    )


def build_workspace(cfg: Config, universe: dict[str, PairData] | None = None) -> Workspace:
    """Load data (unless given), compute features and the filtered condition library."""
    features = all_features()
    universe = universe if universe is not None else load_universe(cfg)
    frames = build_frames(cfg, universe, features)
    report = build_library(cfg, frames, features)
    return Workspace(cfg=cfg, frames=frames, features=features, conditions=report.kept, redundancy=report)
