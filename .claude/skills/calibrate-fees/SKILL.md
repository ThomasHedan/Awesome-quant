---
name: calibrate-fees
description: Convert real-world broker / exchange costs (commission per share, per contract, taker fee, half-spread, tick value) into the fractional fees + slippage that CostsConfig expects. Use when the user says "calibrate fees for X", "what fees should I use", or shows a real broker statement / fee schedule.
---

# Calibrate fees for a CostsConfig

Goal: turn vendor-specific cost language ("$2.50 commission per ES
contract", "10 bps taker fee", "$0.005 per share") into the two scalars
that `CostsConfig` actually consumes — both fractions of trade notional.

## Step 1 — pin down what we're calibrating

Ask via `AskUserQuestion` if any of these are missing:

1. **Asset class** — crypto / equity / future. The conversion formula is
   different for futures.
2. **Broker / exchange** — name (so any error references match real-world
   docs).
3. **Cost components** — commission (per share / contract / order),
   exchange/clearing fees, half-spread (slippage proxy), borrow fees if
   shorting.
4. **Typical trade notional** — for futures only. We need this to convert
   per-contract fees into a fraction.

## Step 2 — apply the right formula

### Crypto

```
fees     = taker_fee_fraction        # e.g. Binance spot = 0.001 = 10 bps
slippage = expected_half_spread      # ballpark: 0.0001 for top pairs, 0.0005 for thinner
```

### Equities (commission-free brokers)

```
fees     = 0.0                       # zero-commission retail
slippage = expected_half_spread      # 0.0001 - 0.0005 for liquid US equities
```

### Equities (per-share commission, e.g. IBKR Pro)

```
fees     = commission_per_share / typical_price
            # $0.005 per share at $200 -> 0.0000125 = 0.125 bps
slippage = expected_half_spread + exchange_fees_per_share / typical_price
```

### Futures

```
fees     = round_trip_commission_per_contract / (typical_price * multiplier)
            # ES at 5000, $50 multiplier, $2.50 commission -> 2.50 / 250_000 = 1e-5
slippage = ticks_slippage * tick_value / (typical_price * multiplier)
            # 1 tick on ES at 5000 -> 12.50 / 250_000 = 5e-5
```

`multiplier` and `tick_value` should match the futures registry in
`src/quant_dashboard/data/futures_source.py`. If the instrument isn't in the
registry, ask the user for the contract spec and pass it via
`spec_overrides` when constructing the data source.

## Step 3 — assemble a CostsConfig and verify

```python
from quant_dashboard.engine import CostsConfig
costs = CostsConfig(fees=<calc>, slippage=<calc>, init_cash=<their account>)
print(costs)
```

Then sanity-check by running a backtest with these costs vs. with
`CostsConfig()` (defaults) and showing the user how `total_fees` changes.
For a strategy that turns over once a week over 5 years, the order of
magnitude of `total_fees / init_cash` should match `(fees + slippage) *
2 * num_round_trips`.

## Step 4 — record the calibration

Recommend the user pin the calibrated values in a small script or a notes
file (don't auto-edit anything outside `.claude/` or `dashboard/`). Format:

```
# costs.<broker>.<instrument>.py
from quant_dashboard.engine import CostsConfig

# IBKR Pro, ES futures, calibrated 2026-01-15:
# commission $2.50 round-trip, slippage 1 tick avg.
ES_IBKR = CostsConfig(fees=1e-5, slippage=5e-5, init_cash=100_000.0)
```

## Common pitfalls

- **Half-spread vs. full-spread.** Slippage represents the *one-way* cost
  of crossing the spread, not the full bid-ask gap. Half the spread is
  the right number for a market order.
- **Per-order vs. per-share commission.** A $1 / order minimum on a 1-share
  order ($200) is 50 bps, not 0.5 bps. If the user trades small notional,
  the per-order minimum can dominate. Surface this.
- **Round-trip vs. one-way.** Some brokers quote round-trip commissions
  (open + close); some quote one-way. The `fees` in `CostsConfig` is
  applied **per fill**, so a round-trip $2.50 means $1.25 per fill in
  vectorbt terms. State your assumption explicitly when calibrating.
- **Borrow / financing.** Not modeled in v1. If shorting or trading on
  margin, point this out — the calibrated `fees` will underestimate the
  true cost of holding.
- **Currency-conversion fees.** Not modeled. If trading non-USD
  instruments through a USD account, flag it.

## Step 5 — close the loop

Report:

- the calibrated `fees` and `slippage` in bps (i.e. `fees * 10_000`),
- the assumed typical price / multiplier (for futures),
- the assumption about half-spread (where used),
- a runnable line that constructs the `CostsConfig`.
