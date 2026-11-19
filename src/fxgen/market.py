"""Per-pair feature frames on the execution timeframe and split datasets.

``PairFrame`` holds everything computed once per pair: execution OHLC, ATR,
all features aligned to the execution bars (context timeframes only via
closed bars) and the native-timeframe rising / cross / calendar states.
``Dataset`` is a split-restricted view with the boolean condition matrix the
GA and the backtester work on.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

from fxgen.conditions.builder import Condition, calendar_rule_name, evaluate_above, native_state
from fxgen.data.loader import PairData
from fxgen.data.mtf import align_to_execution
from fxgen.data.splits import Period
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


def build_pair_frame(
    data: PairData,
    features: list[Feature],
    execution_tf: str,
    context_tfs: list[str],
    atr_period: int = 14,
    state_kinds: dict[str, list[str]] | None = None,
) -> PairFrame:
    """Compute all features for one pair and align them to the execution bars.

    ``state_kinds`` maps a feature column (``"H4.rsi14"``) to the rising/cross
    kinds to precompute; by default every feature gets ``rising`` and crossing
    features also get ``cross_up``/``cross_down``.
    """
    exec_df = data.frames[execution_tf]
    has_volume = bool((exec_df["volume"] > 0).mean() > 0.5)
    feature_cols: list[pd.DataFrame] = []
    state_cols: dict[str, pd.Series] = {}
    for tf in [execution_tf, *context_tfs]:
        native = compute_features(data.frames[tf], features, has_volume)
        states: dict[str, pd.Series] = {}
        for f in features:
            if f.name not in native:
                continue
            column = f"{tf}.{f.name}"
            kinds = state_kinds.get(column, []) if state_kinds is not None else (
                ["rising", "cross_up", "cross_down"] if f.crosses else ["rising"]
            )
            for kind in kinds:
                states[f"{column}:{kind}"] = native_state(native[f.name], kind)  # type: ignore[arg-type]
        native.columns = [f"{tf}.{c}" for c in native.columns]
        state_frame = pd.DataFrame(states, index=native.index)
        if tf != execution_tf:
            native = align_to_execution(native, tf, exec_df.index, execution_tf)
            state_frame = align_to_execution(state_frame, tf, exec_df.index, execution_tf)
        feature_cols.append(native.astype("float32"))
        state_cols.update({c: state_frame[c] for c in state_frame.columns})

    decision_times = exec_df.index + duration(execution_tf)
    for rule in CALENDAR_RULES:
        state_cols[f"{execution_tf}.cal:{rule.name}"] = pd.Series(
            rule.evaluate(decision_times).astype("float32"), index=exec_df.index
        )
    return PairFrame(
        pair=data.pair,
        execution_tf=execution_tf,
        ohlc=exec_df[["open", "high", "low", "close"]],
        atr=core.atr(exec_df, atr_period),
        features=pd.concat(feature_cols, axis=1),
        states=pd.DataFrame(state_cols, index=exec_df.index),
    )


def state_key(cond: Condition) -> str:
    return f"{cond.timeframe}.cal:{calendar_rule_name(cond)}" if cond.kind == "calendar" else cond.id


def is_available(frame: PairFrame, cond: Condition) -> bool:
    """Whether the frame holds the data needed to evaluate ``cond``."""
    if cond.kind == "above":
        return cond.column in frame.features.columns
    return state_key(cond) in frame.states.columns


def condition_column(frame: PairFrame, cond: Condition, mask: np.ndarray | None = None) -> np.ndarray:
    """Boolean column of a fitted condition for one pair (NaN/warmup -> False)."""
    if cond.kind == "above":
        if cond.threshold is None:
            raise ValueError(f"condition {cond.id} has no fitted threshold")
        values = frame.features[cond.column].to_numpy()
        out = evaluate_above(values, cond.threshold)
    else:
        out = np.nan_to_num(frame.states[state_key(cond)].to_numpy(), nan=0.0) > 0.5
    return out if mask is None else out[mask]


@dataclass
class PairSplit:
    """Arrays of one pair restricted to one period."""

    pair: str
    index: pd.DatetimeIndex
    open: np.ndarray
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    atr: np.ndarray
    X: np.ndarray

    def __len__(self) -> int:
        return len(self.index)


@dataclass
class Dataset:
    """Condition matrices for every pair over one period."""

    period: Period
    conditions: list[Condition]
    pairs: dict[str, PairSplit]

    def column_of(self) -> dict[str, int]:
        return {c.id: i for i, c in enumerate(self.conditions)}


def make_dataset(frames: dict[str, PairFrame], conditions: list[Condition], period: Period) -> Dataset:
    """Build the boolean condition matrices for ``period``.

    Bars where the ATR is not yet defined are dropped (indicator warmup).
    """
    pairs: dict[str, PairSplit] = {}
    for pair, frame in frames.items():
        idx = frame.ohlc.index
        mask = period.mask(idx).to_numpy() & frame.atr.notna().to_numpy()
        X = np.empty((int(mask.sum()), len(conditions)), dtype=bool)
        for j, cond in enumerate(conditions):
            X[:, j] = condition_column(frame, cond, mask)
        o = frame.ohlc[mask]
        pairs[pair] = PairSplit(
            pair=pair,
            index=o.index,
            open=o["open"].to_numpy(np.float64),
            high=o["high"].to_numpy(np.float64),
            low=o["low"].to_numpy(np.float64),
            close=o["close"].to_numpy(np.float64),
            atr=frame.atr[mask].to_numpy(np.float64),
            X=X,
        )
    return Dataset(period=period, conditions=list(conditions), pairs=pairs)
