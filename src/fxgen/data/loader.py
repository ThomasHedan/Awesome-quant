"""Generic OHLCV CSV loader: one pair and one timeframe per file."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from fxgen.config import Config, DataConfig
from fxgen.data.validation import DataQuality, clean_and_validate
from fxgen.errors import DataError
from fxgen.timeframes import duration

logger = logging.getLogger(__name__)

COLUMNS = ["open", "high", "low", "close", "volume"]


@dataclass
class PairData:
    """All timeframes loaded for one pair, each indexed by UTC bar open time."""

    pair: str
    frames: dict[str, pd.DataFrame]
    quality: dict[str, DataQuality] = field(default_factory=dict)


def load_csv(path: str | Path, pair: str, timeframe: str, data_cfg: DataConfig) -> tuple[pd.DataFrame, DataQuality]:
    """Load and validate one OHLCV CSV.

    Returns a frame indexed by UTC bar *open* time with float64 columns
    ``open, high, low, close, volume`` (volume is 0 when unavailable).
    """
    path = Path(path)
    if not path.exists():
        raise DataError(f"data file not found: {path} (check data.directory and data.file_pattern)")
    cols = data_cfg.columns
    raw = pd.read_csv(path)
    mapping = {cols.time: "time", cols.open: "open", cols.high: "high", cols.low: "low", cols.close: "close"}
    if cols.volume:
        mapping[cols.volume] = "volume"
    missing = [src for src in mapping if src not in raw.columns]
    if missing and missing != [cols.volume]:
        raise DataError(f"{path.name}: missing columns {missing}; found {list(raw.columns)}")
    raw = raw.rename(columns=mapping)
    if "volume" not in raw.columns:
        raw["volume"] = 0.0

    times = pd.to_datetime(raw["time"], errors="coerce")
    if times.isna().any():
        raise DataError(f"{path.name}: {int(times.isna().sum())} unparseable timestamps in column {cols.time!r}")
    index = pd.DatetimeIndex(times)
    index = index.tz_localize(data_cfg.source_timezone) if index.tz is None else index
    index = index.tz_convert("UTC")
    if data_cfg.timestamp_label == "close":
        index = index - duration(timeframe)

    df = raw[COLUMNS].apply(pd.to_numeric, errors="coerce").astype(np.float64)
    df.index = index.rename("time")
    df["volume"] = df["volume"].fillna(0.0)
    return clean_and_validate(df, pair, timeframe)


def file_path(cfg: Config, pair: str, timeframe: str) -> Path:
    """Path of the CSV holding ``pair`` at ``timeframe``."""
    name = cfg.data.file_pattern.format(pair=pair, timeframe=timeframe)
    return cfg.resolve(cfg.data.directory) / name


def load_pair(cfg: Config, pair: str, timeframes: list[str] | None = None) -> PairData:
    """Load the execution and context timeframes for one pair."""
    tfs = timeframes or [cfg.data.execution_timeframe, *cfg.data.context_timeframes]
    frames: dict[str, pd.DataFrame] = {}
    quality: dict[str, DataQuality] = {}
    for tf in tfs:
        frames[tf], quality[tf] = load_csv(file_path(cfg, pair, tf), pair, tf, cfg.data)
    return PairData(pair=pair, frames=frames, quality=quality)


def load_universe(cfg: Config, pairs: list[str] | None = None) -> dict[str, PairData]:
    """Load every configured pair."""
    selected = pairs or cfg.data.pairs
    unknown = [p for p in selected if p not in cfg.data.pairs]
    if unknown:
        raise DataError(f"pairs {unknown} are not in data.pairs {cfg.data.pairs}")
    universe = {pair: load_pair(cfg, pair) for pair in selected}
    logger.info("loaded %d pairs", len(universe))
    return universe


def discover_datasets(cfg: Config) -> list[dict[str, object]]:
    """List the CSV files present in the data directory with their quality report."""
    directory = cfg.resolve(cfg.data.directory)
    if not directory.exists():
        return []
    pattern = re.escape(cfg.data.file_pattern).replace(r"\{pair\}", r"(?P<pair>\w+?)").replace(
        r"\{timeframe\}", r"(?P<timeframe>H1|H4|D1)"
    )
    regex = re.compile(f"^{pattern}$")
    found: list[dict[str, object]] = []
    for path in sorted(directory.iterdir()):
        match = regex.match(path.name)
        if not match:
            continue
        try:
            _, quality = load_csv(path, match["pair"], match["timeframe"], cfg.data)
            found.append(quality.to_dict())
        except DataError as exc:
            found.append({"pair": match["pair"], "timeframe": match["timeframe"], "error": str(exc)})
    return found
