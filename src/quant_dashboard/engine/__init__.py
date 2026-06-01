"""Backtest engine: runner, metrics, benchmark, Monte Carlo, optimizer."""

from quant_dashboard.engine.benchmark import (
    BenchmarkComparison,
    BenchmarkError,
    build_buy_and_hold_returns,
    compute_benchmark_comparison,
)
from quant_dashboard.engine.metrics import Metrics, compute_metrics, tearsheet_html
from quant_dashboard.engine.montecarlo import (
    MonteCarloError,
    MonteCarloResult,
    run_monte_carlo,
)
from quant_dashboard.engine.optimizer import (
    GridSearchError,
    OptimizationResult,
    estimate_combo_count,
    estimate_ram_bytes,
    heatmap,
    parameter_stability,
    run_grid_search,
)
from quant_dashboard.engine.multicharter import (
    InstrumentResult,
    MultiChartError,
    MultiChartResult,
    run_multi_chart,
)
from quant_dashboard.engine.runner import (
    BacktestResult,
    CostsConfig,
    EngineError,
    periods_per_year,
    run_backtest,
)

__all__ = [
    "BacktestResult",
    "BenchmarkComparison",
    "BenchmarkError",
    "CostsConfig",
    "EngineError",
    "GridSearchError",
    "Metrics",
    "MonteCarloError",
    "MonteCarloResult",
    "OptimizationResult",
    "build_buy_and_hold_returns",
    "compute_benchmark_comparison",
    "compute_metrics",
    "estimate_combo_count",
    "estimate_ram_bytes",
    "heatmap",
    "parameter_stability",
    "periods_per_year",
    "run_backtest",
    "run_grid_search",
    "run_monte_carlo",
    "run_multi_chart",
    "tearsheet_html",
    "InstrumentResult",
    "MultiChartError",
    "MultiChartResult",
]
