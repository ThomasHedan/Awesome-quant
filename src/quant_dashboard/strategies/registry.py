"""Strategy registry.

Strategies register themselves via the :func:`register` decorator. The
dashboard and the optimizer discover available strategies and their param
schemas by calling :func:`available_strategies` / :func:`get_strategy`.
"""

from __future__ import annotations

from typing import Type

from quant_dashboard.strategies.base import Strategy, StrategyError

strategy_registry: dict[str, Type[Strategy]] = {}


def register(cls: Type[Strategy]) -> Type[Strategy]:
    name = getattr(cls, "name", None)
    if not name or name == "base":
        raise StrategyError(f"{cls.__name__} must set a unique 'name' attribute")
    if name in strategy_registry and strategy_registry[name] is not cls:
        raise StrategyError(f"strategy name '{name}' is already registered")
    strategy_registry[name] = cls
    return cls


def available_strategies() -> list[str]:
    return sorted(strategy_registry)


def get_strategy(name: str) -> Type[Strategy]:
    if name not in strategy_registry:
        raise StrategyError(
            f"unknown strategy '{name}'. Available: {available_strategies()}"
        )
    return strategy_registry[name]
