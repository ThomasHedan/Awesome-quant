"""Permutation test of the whole search process.

Each run shuffles the order of the execution-timeframe bars of every pair
(close-to-close return, opening gap and intrabar shape move together), which
destroys any temporal structure while keeping the return distribution. Context
timeframes are rebuilt from the permuted bars. The full pipeline (features,
train-only thresholds, redundancy filter, GA) is then re-run. A real winner
must clearly beat the distribution of best fitnesses obtained on noise.
With ``n`` runs the smallest attainable p-value is ``1 / (n + 1)``, so a 5%
significance needs at least 19 runs.
Only bars before the holdout are used.
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from fxgen.backtest.engine import ExitRules
from fxgen.config import Config
from fxgen.data.loader import PairData
from fxgen.data.resample import resample
from fxgen.data.splits import split_period
from fxgen.ga.evolve import evolve
from fxgen.indicators.base import all_features
from fxgen.market import make_dataset
from fxgen.workspace import build_frames, build_library

logger = logging.getLogger(__name__)


@dataclass
class PermutationResult:
    real_best: float
    permuted_best: list[float] = field(default_factory=list)
    p_value: float = 1.0
    z_score: float = 0.0
    passed: bool = False
    n_trials: int = 0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def permute_bars(bars: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Shuffle bar shapes relative to the previous close and rebuild a price path."""
    logp = np.log(bars[["open", "high", "low", "close"]].to_numpy())
    prev_close = np.concatenate([[logp[0, 0]], logp[:-1, 3]])
    rel = logp - prev_close[:, None]
    order = np.concatenate([[0], 1 + rng.permutation(len(bars) - 1)])
    rel = rel[order]
    closes = logp[0, 0] + np.cumsum(rel[:, 3])
    prev = np.concatenate([[logp[0, 0]], closes[:-1]])
    out = pd.DataFrame(np.exp(prev[:, None] + rel), index=bars.index, columns=["open", "high", "low", "close"])
    out["volume"] = bars["volume"].to_numpy()[order]
    return out


def permute_universe(universe: dict[str, PairData], cfg: Config, rng: np.random.Generator) -> dict[str, PairData]:
    before = split_period(cfg.splits, "holdout", allow_holdout=True).start
    exec_tf = cfg.data.execution_timeframe
    permuted: dict[str, PairData] = {}
    for pair, data in universe.items():
        bars = data.frames[exec_tf]
        shuffled = permute_bars(bars[bars.index < before], rng)
        frames = {exec_tf: shuffled, **{tf: resample(shuffled, tf) for tf in cfg.data.context_timeframes}}
        permuted[pair] = PairData(pair=pair, frames=frames)
    return permuted


def run_permutation_test(universe: dict[str, PairData], cfg: Config, real_best: float, seed: int) -> PermutationResult:
    """Distribution of best GA fitness on permuted data and the p-value of ``real_best``."""
    perm = cfg.validation.permutation
    features = all_features()
    rng = np.random.default_rng(seed)
    best: list[float] = []
    n_trials = 0
    for k in range(perm.n_runs):
        frames = build_frames(cfg, permute_universe(universe, cfg, rng), features)
        library = build_library(cfg, frames, features)
        train = make_dataset(frames, library.kept, split_period(cfg.splits, "train"))
        evo = evolve(train, library.kept, cfg, ExitRules.from_config(cfg.backtest), seed=seed + 1000 + k,
                     population_size=perm.population_size, generations=perm.generations)
        n_trials += evo.n_trials
        top = evo.best(1)
        best.append(top[0].fitness.value if top else 0.0)
        logger.info("permutation %d/%d: best fitness %.3f (real %.3f)", k + 1, perm.n_runs, best[-1], real_best)
    exceed = sum(b >= real_best for b in best)
    p_value = (1 + exceed) / (1 + len(best))
    spread = float(np.std(best, ddof=1)) if len(best) > 1 else 0.0
    z_score = (real_best - float(np.mean(best))) / spread if spread > 0 else 0.0
    return PermutationResult(
        real_best=float(real_best),
        permuted_best=[float(b) for b in best],
        p_value=float(p_value),
        z_score=float(z_score),
        passed=bool(p_value <= perm.significance),
        n_trials=n_trials,
    )

