"""Per-pair feature frames on the execution timeframe and split datasets.

``PairFrame`` holds everything computed once per pair: execution OHLC, ATR,
all features aligned to the execution bars (context timeframes only via
closed bars) and the native-timeframe rising / cross / calendar states.
``Dataset`` is a split-restricted, zero-copy view of those arrays. Conditions
are evaluated lazily, column by column, when a tree needs them: memory does
not grow with the number of thresholds or conditions.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fxgen.conditions.builder import Condition, calendar_rule_name, evaluate_above, native_state, state_suffixes
from fxgen.data.loader import PairData
from fxgen.data.mtf import align_to_execution
from fxgen.data.splits import Period
from fxgen.errors import TreeError
from fxgen.indicators import core
from fxgen.indicators.base import Feature, compute_features
from fxgen.indicators.time import CALENDAR_RULES
from fxgen.timeframes import duration

logger = logging.getLogger(__name__)


@dataclass
class PairFrame:
    pair: str
    execution_tf: str
    ohlc: pd.DataFrame
    atr: pd.Series
    features: pd.DataFrame
    states: pd.DataFrame
    feature_values: np.ndarray = field(init=False, repr=False)
    state_values: np.ndarray = field(init=False, repr=False)
    feature_cols: dict[str, int] = field(init=False, repr=False)
    state_cols: dict[str, int] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self.feature_values = np.asfortranarray(self.features.to_numpy(np.float32))
        self.state_values = np.asfortranarray(self.states.to_numpy(bool))
        self.feature_cols = {c: i for i, c in enumerate(self.features.columns)}
        self.state_cols = {c: i for i, c in enumerate(self.states.columns)}


def build_pair_frame(
    data: PairData,
    features: list[Feature],
    execution_tf: str,
    context_tfs: list[str],
    atr_period: int = 14,
    atr_timeframe: str | None = None,
) -> PairFrame:
    """Compute all features for one pair and align them to the execution bars.

    ``atr_timeframe`` selects the ATR used to size stops and targets (closed
    bars only when it is a context timeframe). Rising and cross states are
    computed on each feature's own timeframe, then aligned.
    """
    exec_df = data.frames[execution_tf]
    has_volume = bool((exec_df["volume"] > 0).mean() > 0.5)
    feature_cols: list[pd.DataFrame] = []
    state_cols: dict[str, np.ndarray] = {}
    for tf in [execution_tf, *context_tfs]:
        native = compute_features(data.frames[tf], features, has_volume)
        states: dict[str, pd.Series] = {}
        for f in features:
            if f.name not in native:
                continue
            column = f"{tf}.{f.name}"
            for suffix, kind, level in state_suffixes(f):
                states[f"{column}{suffix}"] = native_state(native[f.name], kind, level)
        native.columns = [f"{tf}.{c}" for c in native.columns]
        state_frame = pd.DataFrame(states, index=native.index)
        if tf != execution_tf:
            native = align_to_execution(native, tf, exec_df.index, execution_tf)
            state_frame = align_to_execution(state_frame, tf, exec_df.index, execution_tf)
        feature_cols.append(native.astype("float32"))
        state_cols.update({c: state_frame[c].fillna(0.0).to_numpy() > 0.5 for c in state_frame.columns})

    decision_times = exec_df.index + duration(execution_tf)
    for rule in CALENDAR_RULES:
        state_cols[f"{execution_tf}.cal:{rule.name}"] = rule.evaluate(decision_times)
    atr_tf = atr_timeframe or execution_tf
    exit_atr = core.atr(data.frames[atr_tf], atr_period)
    if atr_tf != execution_tf:
        exit_atr = align_to_execution(exit_atr.to_frame("atr"), atr_tf, exec_df.index, execution_tf)["atr"]
    return PairFrame(
        pair=data.pair,
        execution_tf=execution_tf,
        ohlc=exec_df[["open", "high", "low", "close"]],
        atr=exit_atr,
        features=_single_block(pd.concat(feature_cols, axis=1), np.float32),
        states=_single_block(pd.DataFrame(state_cols, index=exec_df.index), bool),
    )


def _single_block(df: pd.DataFrame, dtype: type) -> pd.DataFrame:
    # One Fortran-ordered block: the frame and its numpy view share memory.
    return pd.DataFrame(np.asfortranarray(df.to_numpy(dtype)), index=df.index, columns=df.columns)


def state_key(cond: Condition) -> str:
    return f"{cond.timeframe}.cal:{calendar_rule_name(cond)}" if cond.kind == "calendar" else cond.id


def is_available(frame: PairFrame, cond: Condition) -> bool:
    """Whether the frame holds the data needed to evaluate ``cond``."""
    if cond.kind == "above":
        return cond.column in frame.feature_cols
    return state_key(cond) in frame.state_cols


def condition_column(frame: PairFrame, cond: Condition, mask: np.ndarray | None = None) -> np.ndarray:
    """Boolean column of a fitted condition for one pair (NaN/warmup -> False)."""
    if cond.kind == "above":
        if cond.threshold is None:
            raise ValueError(f"condition {cond.id} has no fitted threshold")
        values = frame.feature_values[:, frame.feature_cols[cond.column]]
        out = evaluate_above(values if mask is None else values[mask], cond.threshold)
        return out
    out = frame.state_values[:, frame.state_cols[state_key(cond)]]
    return out if mask is None else out[mask]


@dataclass
class PairSplit:
    """Views on one pair's arrays restricted to one period."""

    pair: str
    index: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    features: np.ndarray
    states: np.ndarray
    feature_cols: dict[str, int]
    state_cols: dict[str, int]

    def __len__(self) -> int:
        return len(self.index)


@dataclass
class Dataset:
    """Every pair over one period plus the fitted conditions trees may use."""

    period: Period
    conditions: list[Condition]
    pairs: dict[str, PairSplit]
    _by_id: dict[str, Condition] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._by_id = {c.id: c for c in self.conditions}

    def has(self, cid: str) -> bool:
        return cid in self._by_id

    def column(self, data: PairSplit, cid: str) -> np.ndarray:
        """Boolean array of condition ``cid`` on ``data`` (computed on demand)."""
        cond = self._by_id.get(cid)
        if cond is None:
            raise TreeError(f"condition {cid!r} is not in this dataset's condition library")
        if cond.kind == "above":
            if cond.threshold is None:
                raise TreeError(f"condition {cid} has no fitted threshold")
            return evaluate_above(data.features[:, data.feature_cols[cond.column]], cond.threshold)
        return data.states[:, data.state_cols[state_key(cond)]]

    def lookup(self, data: PairSplit) -> Callable[[str], np.ndarray]:
        return lambda cid: self.column(data, cid)


def make_dataset(frames: dict[str, PairFrame], conditions: list[Condition], period: Period) -> Dataset:
    """Zero-copy views of every pair over ``period``.

    Bars where the ATR is not yet defined are dropped (indicator warmup).
    """
    pairs: dict[str, PairSplit] = {}
    for pair, frame in frames.items():
        missing = [c.id for c in conditions if not is_available(frame, c)]
        if missing:
            raise TreeError(f"{pair}: conditions {missing[:5]} cannot be evaluated (missing feature or volume)")
        mask = period.mask(frame.ohlc.index).to_numpy() & frame.atr.notna().to_numpy()
        rows = np.flatnonzero(mask)
        sel: slice | np.ndarray = slice(rows[0], rows[-1] + 1) if rows.size and rows[-1] - rows[0] + 1 == rows.size else rows
        o = frame.ohlc.iloc[sel]
        pairs[pair] = PairSplit(
            pair=pair,
            index=o.index,
            open=o["open"].to_numpy(np.float64),
            high=o["high"].to_numpy(np.float64),
            low=o["low"].to_numpy(np.float64),
            close=o["close"].to_numpy(np.float64),
            atr=frame.atr.to_numpy(np.float64)[sel],
            features=frame.feature_values[sel],
            states=frame.state_values[sel],
            feature_cols=frame.feature_cols,
            state_cols=frame.state_cols,
        )
    return Dataset(period=period, conditions=list(conditions), pairs=pairs)
