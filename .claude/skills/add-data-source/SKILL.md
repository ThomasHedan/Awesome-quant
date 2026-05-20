---
name: add-data-source
description: Scaffold a new DataSource adapter in src/quant_dashboard/data/, register it in the package init, and add the test pattern (mocked vendor calls + gating). Use when the user says "add support for X" where X is a market data vendor (Polygon, Tiingo, IEX Cloud, Kaiko, ...), or asks to read a non-standard local file format.
---

# Add a new data source

Goal: a new adapter that returns a validated `OHLCV` bundle, gated behind
config, with tests that don't touch the network.

## Step 1 — clarify

- **Vendor / SDK.** If a Python SDK exists (`polygon-api-client`, `tiingo`,
  ...), use it. If only REST, use `requests` (already transitively
  available).
- **Asset class.** Equities / crypto / futures / FX / other. Determines the
  default `InstrumentSpec`.
- **Auth.** Which env vars? Add them to `.env.sample` AND document them in
  the adapter's docstring AND in the error message when missing.
- **Timeframes.** Which timeframes does the user need? Map them to our
  canonical strings (`1m, 5m, 15m, 30m, 1h, 4h, 1d, 1w`).

## Step 2 — add the dependency

If a new SDK is needed:

1. Edit `pyproject.toml` -> `[project] dependencies`.
2. Run `uv sync --extra dev`.

If you only need `requests`, skip.

**Do not** introduce a dependency without surfacing it to the user first
("add `polygon-api-client>=1.13`?").

## Step 3 — create the adapter file

Path: `src/quant_dashboard/data/<vendor>_source.py`. Use this template:

```python
"""<Vendor> adapter - <one-line description, asset class, auth model>."""

from __future__ import annotations

import os

import pandas as pd

from quant_dashboard.data.base import (
    AssetClass, DataSource, DataSourceError, InstrumentSpec,
)


_TIMEFRAME_MAP = {
    "1m": "<vendor's 1m string>",
    "5m": "<vendor's 5m string>",
    "1h": "<vendor's 1h string>",
    "1d": "<vendor's 1d string>",
}


class <Vendor>Source(DataSource):
    """<one-line summary>.

    Reads credentials from <ENV_VAR>. Fails loudly with an actionable
    message if missing.
    """

    name = "<vendor>"
    default_asset_class = AssetClass.<EQUITY|CRYPTO|FUTURE|OTHER>

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        # store any vendor-specific config here

    def _client(self):
        try:
            import <vendor_sdk>
        except ImportError as exc:    # pragma: no cover
            raise DataSourceError("<vendor_sdk> is not installed.") from exc
        key = os.environ.get("<VENDOR>_API_KEY")
        if not key:
            raise DataSourceError(
                "<Vendor> API key missing: set <VENDOR>_API_KEY in your "
                "environment (see .env.sample)."
            )
        return <vendor_sdk>.Client(key)

    def _fetch_raw(self, symbol, timeframe, start, end) -> pd.DataFrame:
        if timeframe not in _TIMEFRAME_MAP:
            raise DataSourceError(
                f"<vendor>: unsupported timeframe '{timeframe}'. "
                f"Try one of: {sorted(_TIMEFRAME_MAP)}"
            )
        client = self._client()
        try:
            # ... fetch and return a DataFrame with at least:
            # open, high, low, close, volume columns and a UTC timestamp.
            # normalize_ohlcv() will rename / coerce as needed.
            ...
        except Exception as exc:
            raise DataSourceError(f"<vendor> fetch failed: {exc}") from exc

        if df.empty:
            raise DataSourceError(
                f"<vendor>: empty result for {symbol} {timeframe}"
            )
        return df

    def instrument_spec(self, symbol: str) -> InstrumentSpec:
        # Override if asset class needs contract metadata.
        return InstrumentSpec(
            symbol=symbol,
            asset_class=AssetClass.<...>,
            quote_currency="<USD/USDT/...>",
            multiplier=1.0,
            tick_size=0.01,
            tick_value=0.01,
            exchange="<vendor>",
            session="<24x7|regular|futures>",
        )
```

Hard rules:

1. **Always** raise `DataSourceError` with a plain-English message at every
   failure point. The dashboard surfaces `str(exc)` directly.
2. **Lazy-import** the vendor SDK inside `_client()` / `_fetch_raw()` so
   the dashboard boots even if the SDK is broken or absent.
3. **Don't normalize manually.** `DataSource.fetch()` (the public method,
   on the base class) calls `normalize_ohlcv()` for you. Just produce a
   DataFrame that has the OHLCV columns (any case) and a timestamp.
4. **InstrumentSpec is mandatory** and asset-class-correct. For futures,
   look at `_build_futures_spec` in `futures_source.py` for the contract
   registry pattern.

## Step 4 — register

In `src/quant_dashboard/data/__init__.py`:

- Import the class.
- Add it to `__all__`.

In `dashboard/_shared.py`:

- Add the vendor id to `SOURCE_CHOICES`.
- Add a branch in `_make_source()`.
- Add a default symbol in `data_source_picker()` (the dict near the bottom).
- Add any adapter-specific sidebar widget (e.g. CCXT has an exchange
  picker; IBKR has a contract-type picker).

## Step 5 — add to .env.sample

```
# --- <Vendor> ---
# <one-line description + URL>
<VENDOR>_API_KEY=
```

## Step 6 — add tests

Append to `tests/test_data.py`. At minimum:

- **Gating test**: with the env var missing, calling `fetch()` raises
  `DataSourceError` with a message containing the env var name.

  ```python
  def test_<vendor>_missing_keys_raises(cache_dir, monkeypatch):
      monkeypatch.delenv("<VENDOR>_API_KEY", raising=False)
      src = <Vendor>Source()
      with pytest.raises(DataSourceError, match="<VENDOR>_API_KEY"):
          src.fetch("ABC", "1d", start="2024-01-01", end="2024-01-31")
  ```

- **Mocked fetch**: with the SDK mocked, verify a happy-path fetch
  produces a valid `OHLCV` with the right `InstrumentSpec`.

  ```python
  def test_<vendor>_fetch_mocked(cache_dir, monkeypatch):
      monkeypatch.setenv("<VENDOR>_API_KEY", "fake")
      src = <Vendor>Source()
      fake_df = _synthetic_df(10)  # already returns proper OHLCV
      src._client = lambda: <mock returning fake_df via _fetch_raw>
      bundle = src.fetch("ABC", "1d", start="2024-01-01", end="2024-01-31")
      assert bundle.spec.asset_class == AssetClass.<...>
      assert len(bundle.df) == 10
  ```

- **Timeframe rejection**: passing an unsupported timeframe raises.

**Tests must never hit the network.** Use `unittest.mock` + the fixtures
already in `tests/test_data.py`.

## Step 7 — run the suite

```bash
uv run pytest tests/test_data.py
```

All tests must pass. The dashboard test boot already verifies the page
shell loads.

## Step 8 — report back

- file paths created / modified,
- env vars added,
- test count (before/after),
- a one-line example invocation.
