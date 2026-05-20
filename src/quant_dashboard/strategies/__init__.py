"""Strategy abstractions and the bundled reference strategies."""

from quant_dashboard.strategies.base import (
    ParamSpec,
    Signals,
    Strategy,
    StrategyError,
)
from quant_dashboard.strategies.registry import (
    available_strategies,
    get_strategy,
    register,
    strategy_registry,
)
from quant_dashboard.strategies.sma_cross import SMACross
from quant_dashboard.strategies.rsi_meanrev import RSIMeanReversion

__all__ = [
    "ParamSpec",
    "RSIMeanReversion",
    "SMACross",
    "Signals",
    "Strategy",
    "StrategyError",
    "available_strategies",
    "get_strategy",
    "register",
    "strategy_registry",
]
