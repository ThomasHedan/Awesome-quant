"""Synthetic FX dataset for running the pipeline end to end.

Each pair is simulated at 15-minute resolution from a GARCH(1,1) process with
common currency factors (so pairs are correlated), intraday session seasonality
and the FX weekend close. 15-minute steps are aggregated into H1 bars, then H1
is resampled into H4 and D1 using the 22:00 UTC daily roll.

``edge`` plants a weak, deliberate time-series momentum effect: the drift of
each hour is ``edge * sigma * sign(return over the previous 24 hours)``. With
``edge=0`` the data is pure noise and no strategy should pass validation.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numba
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_PAIRS: dict[str, float] = {
    "EURUSD": 1.10, "GBPUSD": 1.30, "USDJPY": 110.0, "USDCHF": 0.95, "AUDUSD": 0.72, "NZDUSD": 0.68,
    "USDCAD": 1.30, "EURGBP": 0.85, "EURJPY": 125.0, "GBPJPY": 145.0, "AUDJPY": 80.0, "EURCHF": 1.08,
}
CURRENCIES = ["USD", "EUR", "GBP", "JPY", "CHF", "AUD", "NZD", "CAD"]
SUBSTEPS = 4


def _trading_hours(start: str, end: str) -> pd.DatetimeIndex:
    hours = pd.date_range(start, end, freq="1h", tz="UTC", inclusive="left")
    dow, hour = hours.day_of_week, hours.hour
    closed = (dow == 5) | ((dow == 4) & (hour >= 22)) | ((dow == 6) & (hour < 22))
    return hours[~closed]


def _session_multiplier(hours: pd.DatetimeIndex) -> np.ndarray:
    h = hours.hour.to_numpy()
    mult = np.full(len(h), 0.6)
    mult[(h >= 7) & (h < 16)] = 1.2
    mult[(h >= 12) & (h < 16)] = 1.5
    mult[(h >= 16) & (h < 21)] = 0.9
    return mult


@numba.njit(cache=True)
def _simulate(shocks: np.ndarray, session: np.ndarray, base_vol: float, edge: float) -> np.ndarray:
    n = shocks.shape[0]
    omega, alpha, beta = 0.05, 0.06, 0.89
    var = 1.0
    out = np.empty(n)
    window = 24
    running = 0.0
    for t in range(n):
        sigma = base_vol * session[t] * np.sqrt(var)
        drift = 0.0
        if edge != 0.0 and t >= window:
            drift = edge * sigma * np.sign(running)
        r = drift + sigma * shocks[t]
        out[t] = r
        running += r
        if t >= window:
            running -= out[t - window]
        z = shocks[t]
        var = omega + alpha * z * z + beta * var
    return out


def _pair_shocks(pair: str, factors: dict[str, np.ndarray], rng: np.random.Generator) -> np.ndarray:
    base, quote = pair[:3], pair[3:]
    idio = rng.standard_normal(len(factors["USD"]))
    combined = 0.6 * (factors[base] - factors[quote]) / np.sqrt(2) + 0.8 * idio
    return combined / combined.std()


def _hourly_to_bars(
    log_returns: np.ndarray, hours: pd.DatetimeIndex, start_price: float, base_vol: float, rng: np.random.Generator
) -> pd.DataFrame:
    n = len(log_returns)
    sub = rng.standard_normal((n, SUBSTEPS)) * base_vol / np.sqrt(SUBSTEPS)
    sub = sub - sub.mean(axis=1, keepdims=True) + log_returns[:, None] / SUBSTEPS
    path = np.log(start_price) + np.cumsum(sub.ravel()).reshape(n, SUBSTEPS)
    close_log = path[:, -1]
    open_log = np.concatenate([[np.log(start_price)], close_log[:-1]])
    high_log = np.maximum(path.max(axis=1), open_log)
    low_log = np.minimum(path.min(axis=1), open_log)
    volume = np.round(np.abs(log_returns) / base_vol * 400 + rng.uniform(50, 150, n))
    return pd.DataFrame(
        {
            "open": np.exp(open_log),
            "high": np.exp(high_log),
            "low": np.exp(low_log),
            "close": np.exp(close_log),
            "volume": volume,
        },
        index=hours.rename("time"),
    )


def resample(h1: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    """Resample H1 bars into H4 or D1 aligned on the 22:00 UTC daily roll."""
    rule = {"H4": "4h", "D1": "24h"}[timeframe]
    out = h1.resample(rule, origin="epoch", offset="22h", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )
    return out.dropna(subset=["open"])


def generate_universe(
    start: str = "2016-01-03",
    end: str = "2025-01-01",
    pairs: dict[str, float] | None = None,
    edge: float = 0.01,
    seed: int = 42,
) -> dict[str, dict[str, pd.DataFrame]]:
    """Simulate H1/H4/D1 bars for every pair. Returns ``{pair: {tf: frame}}``."""
    pairs = pairs or DEFAULT_PAIRS
    rng = np.random.default_rng(seed)
    hours = _trading_hours(start, end)
    session = _session_multiplier(hours)
    factors = {ccy: rng.standard_normal(len(hours)) for ccy in CURRENCIES}
    base_vol = 0.0012
    universe: dict[str, dict[str, pd.DataFrame]] = {}
    for pair, price in pairs.items():
        shocks = _pair_shocks(pair, factors, rng)
        returns = _simulate(shocks, session, base_vol, edge)
        h1 = _hourly_to_bars(returns, hours, price, base_vol * 0.5, rng)
        universe[pair] = {"H1": h1, "H4": resample(h1, "H4"), "D1": resample(h1, "D1")}
    return universe


def write_universe(
    directory: str | Path,
    start: str = "2016-01-03",
    end: str = "2025-01-01",
    edge: float = 0.01,
    seed: int = 42,
    file_pattern: str = "{pair}_{timeframe}.csv",
) -> list[Path]:
    """Generate the synthetic universe and write one CSV per pair and timeframe."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for pair, frames in generate_universe(start, end, edge=edge, seed=seed).items():
        for tf, frame in frames.items():
            path = directory / file_pattern.format(pair=pair, timeframe=tf)
            out = frame.copy()
            out.index = out.index.strftime("%Y-%m-%d %H:%M:%S")
            out.to_csv(path, index_label="time", float_format="%.6f")
            written.append(path)
    logger.info("wrote %d synthetic files to %s (edge=%.3f, seed=%d)", len(written), directory, edge, seed)
    return written
