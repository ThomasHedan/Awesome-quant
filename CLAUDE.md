# CLAUDE.md — fxgen

Genetic decision-tree generator for FX swing trading (H1/H4/D1) with strict robustness validation.
The priority is avoiding data-mining bias, not pretty backtests. See README.md for usage.

## Invariants (never break them)

1. **No look-ahead.** Decision at the close of bar `t`, fill at the open of `t+1`. Context timeframes are
   aligned on their *close* time (`data/mtf.py`). Every indicator is causal. The corruption tests in
   `tests/test_indicators.py`, `tests/test_data.py` and `tests/test_backtest.py` must stay green.
2. **Thresholds are fitted on the train split only** (`conditions/thresholds.py`).
3. **The GA only sees train data.** `ga/evolve.check_not_holdout` refuses anything overlapping the holdout.
4. **The holdout is read only through `validation/holdout.py`**, which records every access. Holdout
   accesses and manual backtests count as trials in the deflated Sharpe ratio. Never add a path that reads
   holdout data without going through the ledger, and never let the MCP server override `splits`.
5. **Reproducibility.** Every random draw is seeded from the config.
6. **Portfolios and classifications only read stored train / validation results**, never the holdout.

## Conventions

- Python 3.11+, `from __future__ import annotations`, full type hints, `logging` (never `print`; stdout is
  reserved for MCP stdio and CLI JSON output).
- `uv sync --extra dev`, `uv run pytest`. Do not `pip install`.
- Errors: raise a subclass of `fxgen.errors.FxgenError` with a message that tells the user what to do.
- Comments only for non-obvious *why*. No emojis.
- New feature: add it to the family module in `indicators/` with a matching Pine v6 body, keep the
  look-ahead test green, and regenerate `docs/INDICATORS.md` (`uv run fxgen indicators > docs/INDICATORS.md`).
- `mcp` is pinned `<2` because FastMCP was renamed in 2.x.
- Conditions are evaluated lazily from zero-copy feature views (`market.Dataset.column`); do not build
  dense condition matrices.
- Never assume nanosecond timestamps (pandas 3 loads CSV dates in microseconds).
- Secrets (e.g. `LSE_API_KEY`) come from the environment only; never write them to files.

## Commands

```bash
uv run pytest                                   # full suite (~7 min)
uv run pytest -k "lookahead or causal"           # causality tests
uv run fxgen synth && uv run fxgen --config config/demo.yaml evolve && uv run fxgen --config config/demo.yaml holdout
uv run fxgen-mcp                                # MCP server (stdio), FXGEN_CONFIG selects the YAML
```
