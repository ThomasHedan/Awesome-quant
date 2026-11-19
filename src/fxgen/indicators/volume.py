"""Volume features (tick volume for FX). Skipped when the data has no volume."""

from __future__ import annotations

import pandas as pd

from fxgen.indicators import core
from fxgen.indicators.base import Feature


def _obv_flow(df: pd.DataFrame) -> pd.Series:
    obv = core.obv(df)
    return (obv - obv.shift(10)) / df["volume"].rolling(10, min_periods=10).sum()


FEATURES: list[Feature] = [
    Feature(
        name="volume_z",
        family="volume",
        label="zscore(volume, 50)",
        compute=lambda df: core.zscore(df["volume"], 50),
        pine="(volume - ta.sma(volume, 50)) / ta.stdev(volume, 50)",
        signed=True,
        needs_volume=True,
    ),
    Feature(
        name="obv_flow10",
        family="volume",
        label="10-bar OBV change / 10-bar volume",
        compute=_obv_flow,
        pine="(ta.obv - ta.obv[10]) / math.sum(volume, 10)",
        signed=True,
        needs_volume=True,
    ),
    Feature(
        name="mfi14",
        family="volume",
        label="Money flow index(14)",
        compute=lambda df: core.mfi(df, 14),
        pine="ta.mfi(hlc3, 14)",
        needs_volume=True,
    ),
]
