from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxgen.analysis.challenge import FAIL_DAILY, FAIL_DD, PASS, TIMEOUT, run_phase, simulate_challenge, trading_days
from fxgen.config import ChallengeConfig, PropFirmConfig
from fxgen.data.splits import Period
from fxgen.errors import FxgenError


def _phase(pct: list[float], target: float | None = 8.0, trailing: bool = False, min_days: int = 0,
           traded: list[bool] | None = None):  # type: ignore[no-untyped-def]
    p = np.array([pct], dtype=float)
    t = np.array([traded if traded is not None else [True] * len(pct)])
    out, days, final = run_phase(p, t, target, daily_limit=5.0, dd_limit=10.0, trailing=trailing, min_trading_days=min_days)
    return int(out[0]), int(days[0]), float(final[0])


def test_phase_outcomes() -> None:
    assert _phase([3, 3, 3]) == (PASS, 3, 9.0)
    assert _phase([1, -5, 9])[0] == FAIL_DAILY
    assert _phase([-4, -4, -3])[0] == FAIL_DD
    assert _phase([1, 1, 1])[0] == TIMEOUT
    assert _phase([4, 4, 0, 0], min_days=3, traded=[True, True, False, True]) == (PASS, 4, 8.0)
    path = [4, 4, -4, -4, -3]
    assert _phase(path, target=20, trailing=False)[0] == TIMEOUT
    assert _phase(path, target=20, trailing=True)[0] == FAIL_DD
    assert _phase([1, -2, 1], target=None)[0] == TIMEOUT


def _series(mean: float, n: int = 500, seed: int = 0) -> tuple[pd.Series, pd.Series]:
    idx = pd.bdate_range("2020-01-01", periods=n, tz="UTC")
    r = pd.Series(np.random.default_rng(seed).normal(mean, 1.0, n), index=idx)
    return r, pd.Series(True, index=idx)


def test_simulation_behaves_sensibly() -> None:
    prop = PropFirmConfig()
    ch = ChallengeConfig(n_sims=400, edge_haircut=0.0, risk_grid=[0.25, 0.5, 1.0, 3.0])
    good = simulate_challenge(*_series(0.3), prop, seed=1, challenge=ch)
    bad = simulate_challenge(*_series(-0.3), prop, seed=1, challenge=ch)
    assert good["best_p_pass"] > 0.6 and bad["best_p_pass"] < 0.05
    rows = {r["risk_per_trade_pct"]: r for r in good["by_risk"]}
    assert rows[3.0]["p_fail_daily_loss"] > rows[0.25]["p_fail_daily_loss"]
    assert rows[0.25]["median_days_to_pass"] > rows[1.0]["median_days_to_pass"]
    assert all(r["p_pass_phase 2"] <= r["p_pass_phase 1"] for r in good["by_risk"])
    assert simulate_challenge(*_series(0.3), prop, seed=1, challenge=ch) == good
    cut = simulate_challenge(*_series(0.3), prop, seed=1, challenge=ch.model_copy(update={"edge_haircut": 0.8}))
    assert cut["best_p_pass"] < good["best_p_pass"]
    with pytest.raises(FxgenError):
        simulate_challenge(*_series(0.3, n=10), prop, seed=1, challenge=ch)


def test_trading_days_roll_weekends() -> None:
    period = Period(pd.Timestamp("2020-01-06", tz="UTC"), pd.Timestamp("2020-01-20", tz="UTC"))
    trades = pd.DataFrame({"entry_time": pd.to_datetime(["2020-01-07 10:00", "2020-01-12 22:00"], utc=True)})
    flags = trading_days(trades, period)
    assert flags[flags].index.strftime("%Y-%m-%d").tolist() == ["2020-01-07", "2020-01-13"]
