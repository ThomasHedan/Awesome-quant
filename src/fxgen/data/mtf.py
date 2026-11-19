"""Multi-timeframe alignment without look-ahead.

A bar labelled by its open time ``T`` on timeframe ``tf`` is only known once it
has closed, at ``T + duration(tf)``. Decisions on the execution timeframe are
taken at the close of each execution bar. A context bar is therefore usable for
the execution bar opening at ``t`` only if ``T + duration(tf) <= t + duration(exec)``.
"""

from __future__ import annotations

import pandas as pd

from fxgen.timeframes import duration


def align_to_execution(
    context: pd.DataFrame,
    context_tf: str,
    execution_index: pd.DatetimeIndex,
    execution_tf: str,
) -> pd.DataFrame:
    """Project context-timeframe columns onto the execution bars.

    Each execution row receives the values of the latest context bar that had
    *closed* by the execution bar's decision time (its close). Rows before the
    first closed context bar are NaN.
    """
    ctx = context.copy()
    ctx["_available_at"] = ctx.index + duration(context_tf)
    ctx = ctx.sort_values("_available_at")
    decision = pd.DataFrame(
        {"_decision_at": execution_index + duration(execution_tf)}, index=execution_index
    )
    merged = pd.merge_asof(
        decision.sort_values("_decision_at"),
        ctx,
        left_on="_decision_at",
        right_on="_available_at",
        direction="backward",
        allow_exact_matches=True,
    )
    merged.index = execution_index
    return merged.drop(columns=["_decision_at", "_available_at"])
