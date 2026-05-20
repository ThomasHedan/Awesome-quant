"""Tiny smoke test: build a synthetic OHLCV series, run an SMA-cross backtest
through vectorbt, and dump a handful of QuantStats metrics.

Run with: ``uv run python -m quant_dashboard.smoke``

Intentionally self-contained — no data adapters, no UI. The point is to
confirm the engine boots and the API signatures we'll build against are
present in the installed versions.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _synthetic_close(n: int = 500, seed: int = 7) -> pd.Series:
    rng = np.random.default_rng(seed)
    # Geometric Brownian motion-ish drift + noise; nothing fancy.
    rets = rng.normal(loc=0.0005, scale=0.012, size=n)
    price = 100.0 * np.exp(np.cumsum(rets))
    idx = pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC")
    return pd.Series(price, index=idx, name="close")


def main() -> None:
    import vectorbt as vbt
    import quantstats as qs

    print(f"vectorbt {vbt.__version__}")
    print(f"quantstats {qs.__version__}")

    close = _synthetic_close()

    fast = vbt.MA.run(close, window=10).ma
    slow = vbt.MA.run(close, window=30).ma
    entries = fast.vbt.crossed_above(slow)
    exits = fast.vbt.crossed_below(slow)

    pf = vbt.Portfolio.from_signals(
        close=close,
        entries=entries,
        exits=exits,
        fees=0.001,
        slippage=0.0005,
        freq="1D",
        init_cash=10_000.0,
    )

    print("\n--- vectorbt ---")
    print(f"Total return:    {pf.total_return():.4f}")
    print(f"Sharpe (annual): {pf.sharpe_ratio():.4f}")
    print(f"Max drawdown:    {pf.max_drawdown():.4f}")

    returns = pf.returns()
    if hasattr(returns, "tz_localize"):
        returns = returns.tz_localize(None) if returns.index.tz is not None else returns

    print("\n--- quantstats ---")
    print(f"CAGR:            {qs.stats.cagr(returns):.4f}")
    print(f"Sortino:         {qs.stats.sortino(returns):.4f}")
    print(f"Calmar:          {qs.stats.calmar(returns):.4f}")
    print(f"Tail ratio:      {qs.stats.tail_ratio(returns):.4f}")
    print(f"VaR (95%):       {qs.stats.value_at_risk(returns):.4f}")

    print("\nOK")


if __name__ == "__main__":
    main()
