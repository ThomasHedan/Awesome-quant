"""Strategy abstractions.

A strategy maps an :class:`OHLCV` bundle and a parameter dict to a pair of
boolean Series (``entries``, ``exits``). The signals are *aligned to the bar
on which the decision is made* — the engine is responsible for shifting them
one bar forward before they reach vectorbt, which keeps the no-look-ahead
invariant in a single place (and lets us test it once for every strategy).

Strategies declare their tunable parameters via :class:`ParamSpec`, which is
what the dashboard introspects to build the UI and what the optimizer uses to
construct the search grid.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Iterable, Literal

import pandas as pd

from quant_dashboard.data import OHLCV


class StrategyError(RuntimeError):
    """Raised for user-actionable strategy problems (bad params, etc.)."""


ParamType = Literal["int", "float", "choice"]


@dataclass(frozen=True)
class ParamSpec:
    """Schema for a single strategy parameter.

    For ``int`` / ``float`` provide ``low``, ``high``, and ``step`` so the UI
    can render a slider and the optimizer can build a grid. For ``choice`` set
    ``choices`` to the allowed values.
    """

    name: str
    kind: ParamType
    default: Any
    low: float | int | None = None
    high: float | int | None = None
    step: float | int | None = None
    choices: tuple[Any, ...] | None = None
    description: str = ""

    def grid(self) -> list[Any]:
        """Default sweep values for the optimizer. Subclasses / callers can
        override per-run; this is the "use the schema as-is" path."""
        if self.kind == "choice":
            if not self.choices:
                raise StrategyError(f"{self.name}: choice param missing choices")
            return list(self.choices)
        if self.low is None or self.high is None or self.step is None:
            raise StrategyError(
                f"{self.name}: numeric param missing low/high/step for grid()"
            )
        out: list[Any] = []
        v = self.low
        # Use a count loop to avoid float drift accumulating across additions.
        n = int(round((self.high - self.low) / self.step)) + 1
        for i in range(n):
            x = self.low + i * self.step
            if self.kind == "int":
                out.append(int(round(x)))
            else:
                out.append(float(x))
        return out

    def validate(self, value: Any) -> Any:
        if self.kind == "int":
            v = int(value)
        elif self.kind == "float":
            v = float(value)
        elif self.kind == "choice":
            if self.choices is None or value not in self.choices:
                raise StrategyError(
                    f"{self.name}: value {value!r} not in {self.choices!r}"
                )
            return value
        else:  # pragma: no cover - exhausted by Literal
            raise StrategyError(f"{self.name}: unknown kind {self.kind!r}")
        if self.low is not None and v < self.low:
            raise StrategyError(f"{self.name}: {v} < low {self.low}")
        if self.high is not None and v > self.high:
            raise StrategyError(f"{self.name}: {v} > high {self.high}")
        return v


@dataclass
class Signals:
    """Decision-bar-aligned entry / exit signals.

    Both Series share the OHLCV index, contain booleans, and have no NaN.
    Long-only by default — short legs are optional and default to all-False.
    The engine shifts these by one bar before submitting to vectorbt.
    """

    entries: pd.Series
    exits: pd.Series
    short_entries: pd.Series | None = None
    short_exits: pd.Series | None = None
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name, s in [("entries", self.entries), ("exits", self.exits)]:
            if not isinstance(s, pd.Series):
                raise StrategyError(f"{name} must be a pandas Series")
            if s.isna().any():
                raise StrategyError(f"{name} contains NaN")
            if s.dtype != bool:
                # Coerce in place so downstream is uniform.
                setattr(self, name, s.astype(bool))


class Strategy(ABC):
    """Base class for every strategy.

    Subclasses set :attr:`name` and implement :meth:`param_specs` and
    :meth:`_compute_signals`. The public :meth:`generate_signals` validates
    params, runs the computation, and enforces invariants (boolean dtype,
    no NaN, index alignment).
    """

    name: str = "base"
    description: str = ""

    # --- public API ------------------------------------------------------

    @classmethod
    @abstractmethod
    def param_specs(cls) -> list[ParamSpec]:
        """Tunable parameters with defaults, ranges, and a UI description."""

    @classmethod
    def default_params(cls) -> dict[str, Any]:
        return {p.name: p.default for p in cls.param_specs()}

    def generate_signals(self, data: OHLCV, **params: Any) -> Signals:
        validated = self._validate_params(params)
        sig = self._compute_signals(data, **validated)
        if not sig.entries.index.equals(data.df.index):
            raise StrategyError(
                f"{self.name}: entries index does not match OHLCV index"
            )
        if not sig.exits.index.equals(data.df.index):
            raise StrategyError(
                f"{self.name}: exits index does not match OHLCV index"
            )
        return sig

    # --- subclass hooks --------------------------------------------------

    @abstractmethod
    def _compute_signals(self, data: OHLCV, **params: Any) -> Signals:
        """Build entry/exit signals aligned to the decision bar."""

    # --- helpers ---------------------------------------------------------

    @classmethod
    def _validate_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        specs = {p.name: p for p in cls.param_specs()}
        out: dict[str, Any] = {}
        for spec in specs.values():
            raw = params.get(spec.name, spec.default)
            out[spec.name] = spec.validate(raw)
        unknown = set(params) - set(specs)
        if unknown:
            raise StrategyError(f"{cls.name}: unknown params {sorted(unknown)}")
        return out


def empty_bool_series(index: pd.Index) -> pd.Series:
    """Helper for strategies that emit only long signals."""
    return pd.Series(False, index=index, dtype=bool)


def boolify(s: pd.Series) -> pd.Series:
    """Fill NaN with False and cast to bool — common at the tail of an
    indicator pipeline where the leading window is NaN."""
    return s.fillna(False).astype(bool)


def iter_param_grid(
    grids: dict[str, Iterable[Any]],
) -> Iterable[dict[str, Any]]:
    """Cartesian product of ``{name: values}`` -> dicts. Used by the optimizer."""
    from itertools import product

    keys = list(grids.keys())
    for combo in product(*[list(grids[k]) for k in keys]):
        yield dict(zip(keys, combo, strict=True))
