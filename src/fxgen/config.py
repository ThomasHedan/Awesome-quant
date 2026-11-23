"""Centralised configuration loaded from YAML and validated with pydantic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"

Timeframe = Literal["H1", "H4", "D1"]


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ColumnsConfig(_Strict):
    time: str = "time"
    open: str = "open"
    high: str = "high"
    low: str = "low"
    close: str = "close"
    volume: str | None = "volume"


class DataConfig(_Strict):
    directory: str = "data/synthetic"
    file_pattern: str = "{pair}_{timeframe}.csv"
    columns: ColumnsConfig = ColumnsConfig()
    source_timezone: str = "UTC"
    timestamp_label: Literal["open", "close"] = "open"
    pairs: list[str]
    execution_timeframe: Timeframe = "H1"
    context_timeframes: list[Timeframe] = Field(default_factory=lambda: ["H4", "D1"])

    @model_validator(mode="after")
    def _context_above_execution(self) -> DataConfig:
        order = {"H1": 0, "H4": 1, "D1": 2}
        for tf in self.context_timeframes:
            if order[tf] <= order[self.execution_timeframe]:
                raise ValueError(
                    f"context timeframe {tf} must be higher than execution timeframe "
                    f"{self.execution_timeframe}"
                )
        return self


class SplitsConfig(_Strict):
    train_start: str
    validation_start: str
    holdout_start: str
    end: str

    @model_validator(mode="after")
    def _ordered(self) -> SplitsConfig:
        dates = [pd.Timestamp(d) for d in (self.train_start, self.validation_start, self.holdout_start, self.end)]
        if not all(a < b for a, b in zip(dates, dates[1:])):
            raise ValueError("splits must satisfy train_start < validation_start < holdout_start < end")
        return self


class ConditionsConfig(_Strict):
    """Condition library. Glob filters select what the GA may use (e.g. ``"*.rsi*"``, ``"D1.*"``)."""

    percentiles: list[int] = Field(default_factory=lambda: [5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95])
    families: list[Literal["trend", "momentum", "volatility", "structure", "volume", "time"]] | None = None
    exclude_features: list[str] = Field(default_factory=list)
    include_conditions: list[str] = Field(default_factory=list)
    exclude_conditions: list[str] = Field(default_factory=list)
    normalization_window: int = Field(250, ge=20)
    redundancy_threshold: float = Field(0.9, gt=0, le=1)
    min_activation_rate: float = Field(0.03, ge=0, lt=0.5)
    max_activation_rate: float = Field(0.97, gt=0.5, le=1)

    @field_validator("percentiles")
    @classmethod
    def _valid_percentiles(cls, v: list[int]) -> list[int]:
        if not v or any(p <= 0 or p >= 100 for p in v):
            raise ValueError("percentiles must be in (0, 100)")
        return sorted(set(v))


class TreeConfig(_Strict):
    max_depth: int = Field(3, ge=1, le=5)
    min_depth: int = Field(1, ge=1)


class BacktestConfig(_Strict):
    stop_atr: float = Field(1.5, gt=0)
    take_profit_atr: float | None = Field(3.0, gt=0)
    max_holding_bars: int | None = Field(240, gt=0)
    exit_on_leaf_change: bool = False
    atr_period: int = Field(14, ge=2)
    atr_timeframe: Timeframe = "D1"
    risk_per_trade_pct: float = Field(0.5, gt=0, le=5)


class CostsConfig(_Strict):
    commission_pips_round_turn: float = Field(0.7, ge=0)
    default_spread_pips: float = Field(1.5, ge=0)
    spread_pips: dict[str, float] = Field(default_factory=dict)
    swap_pips_long: float = -0.5
    swap_pips_short: float = -0.5

    def spread_for(self, pair: str) -> float:
        return self.spread_pips.get(pair, self.default_spread_pips)


class GAConfig(_Strict):
    population_size: int = Field(200, ge=10)
    generations: int = Field(30, ge=1)
    tournament_size: int = Field(4, ge=2)
    crossover_prob: float = Field(0.7, ge=0, le=1)
    mutation_prob: float = Field(0.3, ge=0, le=1)
    elite_size: int = Field(5, ge=0)
    sharing_radius: float = Field(0.8, gt=0, le=1)
    n_jobs: int = -1


class FitnessConfig(_Strict):
    min_trades_total: int = Field(150, ge=1)
    min_trades_per_pair: int = Field(10, ge=0)
    max_trades_per_pair_year: float | None = Field(100, gt=0)
    complexity_penalty: float = Field(0.02, ge=0)
    max_pair_pnl_share: float = Field(0.4, gt=0, le=1)
    n_subperiods: int = Field(4, ge=1)
    min_positive_subperiods: int = Field(3, ge=0)


class WalkForwardConfig(_Strict):
    n_windows: int = Field(4, ge=0)
    train_years: float = Field(3, gt=0)
    test_years: float = Field(1, gt=0)
    generations: int = Field(15, ge=1)
    population_size: int = Field(100, ge=10)


class PermutationConfig(_Strict):
    """GA budget per permuted run; ``None`` reuses the main GA budget (fair comparison)."""

    n_runs: int = Field(19, ge=0)
    generations: int | None = Field(None, ge=1)
    population_size: int | None = Field(None, ge=10)
    significance: float = Field(0.05, gt=0, lt=1)


class SensitivityConfig(_Strict):
    percentile_shift: int = Field(5, ge=1, le=20)
    exit_scale: list[float] = Field(default_factory=lambda: [0.8, 1.2])
    max_degradation: float = Field(0.5, gt=0, le=1)


class MonteCarloConfig(_Strict):
    n_sims: int = Field(1000, ge=10)
    confidence: float = Field(0.95, gt=0.5, lt=1)


class ValidationConfig(_Strict):
    walkforward: WalkForwardConfig = WalkForwardConfig()
    permutation: PermutationConfig = PermutationConfig()
    sensitivity: SensitivityConfig = SensitivityConfig()
    montecarlo: MonteCarloConfig = MonteCarloConfig()
    dsr_min: float = Field(0.95, gt=0, lt=1)
    n_candidates: int = Field(30, ge=1)
    n_finalists: int = Field(10, ge=1)


class ChallengePhase(_Strict):
    name: str
    profit_target_pct: float = Field(gt=0)
    max_days: int | None = Field(None, ge=1)
    min_trading_days: int = Field(0, ge=0)


class ChallengeConfig(_Strict):
    """Prop-firm challenge simulator (bootstrap of daily returns, train + validation only)."""

    phases: list[ChallengePhase] = Field(default_factory=lambda: [
        ChallengePhase(name="phase 1", profit_target_pct=8.0, min_trading_days=4),
        ChallengePhase(name="phase 2", profit_target_pct=5.0, min_trading_days=4),
    ])
    drawdown_mode: Literal["static", "trailing"] = "static"
    no_limit_cap_days: int = Field(365, ge=20)
    funded_days: int = Field(120, ge=0)
    fee: float = Field(0.0, ge=0)
    risk_grid: list[float] = Field(default_factory=lambda: [0.25, 0.5, 0.75, 1.0, 1.5, 2.0])
    n_sims: int = Field(5000, ge=100)
    block_days: int = Field(5, ge=1)
    edge_haircut: float = Field(0.5, ge=0, le=1)

    @field_validator("risk_grid")
    @classmethod
    def _positive(cls, v: list[float]) -> list[float]:
        if not v or any(r <= 0 for r in v):
            raise ValueError("risk_grid needs positive risks per trade (%)")
        return sorted(set(v))


class PropFirmConfig(_Strict):
    max_total_drawdown_pct: float = Field(10.0, gt=0)
    max_daily_loss_pct: float = Field(5.0, gt=0)
    min_risk_per_trade_pct: float = Field(0.25, gt=0)
    challenge: ChallengeConfig = ChallengeConfig()


class PortfolioConfig(_Strict):
    """Selection of decorrelated strategies (validation data only, never the holdout)."""

    max_strategies: int = Field(5, ge=1, le=30)
    max_correlation: float = Field(0.5, gt=0, le=1)
    select_by: Literal["validation_dsr", "validation_sharpe", "train_fitness"] = "validation_dsr"
    min_validation_sharpe: float = 0.0
    survivors_only: bool = True
    max_per_type: int | None = Field(2, ge=1)
    weighting: Literal["equal", "inverse_vol"] = "equal"


class StorageConfig(_Strict):
    database: str = "output/fxgen.db"
    output_dir: str = "output"


class Config(_Strict):
    seed: int = 42
    data: DataConfig
    splits: SplitsConfig
    conditions: ConditionsConfig = ConditionsConfig()
    tree: TreeConfig = TreeConfig()
    backtest: BacktestConfig = BacktestConfig()
    costs: CostsConfig = CostsConfig()
    ga: GAConfig = GAConfig()
    fitness: FitnessConfig = FitnessConfig()
    validation: ValidationConfig = ValidationConfig()
    prop_firm: PropFirmConfig = PropFirmConfig()
    portfolio: PortfolioConfig = PortfolioConfig()
    storage: StorageConfig = StorageConfig()

    def resolve(self, relative: str) -> Path:
        """Resolve a config path relative to the project root."""
        path = Path(relative)
        return path if path.is_absolute() else PROJECT_ROOT / path

    def config_hash(self) -> str:
        """Stable short hash of the full configuration."""
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode()).hexdigest()[:16]


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in overrides.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def load_config(path: str | Path | None = None, overrides: dict[str, Any] | None = None) -> Config:
    """Load a YAML config, apply nested overrides, and validate it."""
    config_path = Path(path) if path else DEFAULT_CONFIG_PATH
    with config_path.open() as fh:
        raw = yaml.safe_load(fh) or {}
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config.model_validate(raw)
