"""Feature definition and registry.

A feature is a causal, normalised series computed on one timeframe's OHLCV.
Normalisation (ratio to ATR, bounded oscillators, rolling percentile ranks or
z-scores) makes thresholds comparable from one pair to another. Each feature
carries the equivalent Pine Script v6 expression for the TradingView export.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pandas as pd

Family = str
FAMILIES: tuple[Family, ...] = ("trend", "momentum", "volatility", "structure", "volume", "time")


@dataclass(frozen=True)
class Feature:
    """A normalised indicator series.

    ``pine`` is a Pine Script v6 function body: one expression, or several
    lines whose last line is the returned value.

    ``signed`` means zero is a meaningful level (e.g. distance to a moving
    average), which enables the ``> 0`` and zero-cross conditions.
    """

    name: str
    family: Family
    label: str
    compute: Callable[[pd.DataFrame], pd.Series]
    pine: str
    signed: bool = False
    crosses: bool = False
    needs_volume: bool = False


def all_features() -> list[Feature]:
    """Every feature of the library, in a stable order."""
    from fxgen.indicators import momentum, structure, trend, volatility, volume

    features = [*trend.FEATURES, *momentum.FEATURES, *volatility.FEATURES, *structure.FEATURES, *volume.FEATURES]
    names = [f.name for f in features]
    duplicates = {n for n in names if names.count(n) > 1}
    if duplicates:
        raise ValueError(f"duplicate feature names: {sorted(duplicates)}")
    return features


def compute_features(df: pd.DataFrame, features: list[Feature], has_volume: bool) -> pd.DataFrame:
    """Compute a feature frame (float32) for one timeframe of one pair."""
    columns = {f.name: f.compute(df) for f in features if has_volume or not f.needs_volume}
    out = pd.DataFrame(columns, index=df.index)
    return out.replace([float("inf"), float("-inf")], float("nan")).astype("float32")
