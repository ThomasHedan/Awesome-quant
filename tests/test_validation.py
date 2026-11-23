from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxgen.config import Config, MonteCarloConfig, PropFirmConfig
from fxgen.data.loader import load_universe
from fxgen.data.splits import Period
from fxgen.data.synthetic import write_universe
from fxgen.errors import HoldoutLockedError
from fxgen.ga.evolve import evolve
from fxgen.tree.node import Action, Leaf, Node
from fxgen.validation.battery import TrialContext, run_robustness
from fxgen.validation.dsr import deflated_sharpe, expected_max_sharpe
from fxgen.validation.holdout import MemoryLedger, run_holdout_stage
from fxgen.validation.montecarlo import run_monte_carlo
from fxgen.validation.permutation import permute_bars, permute_universe, run_permutation_test
from fxgen.validation.prop_firm import prop_firm_limits
from fxgen.validation.sensitivity import run_sensitivity
from fxgen.validation.walkforward import walkforward_process, walkforward_tree
from fxgen.workspace import Workspace, build_workspace

from conftest import hourly_frame, make_config


def _tree(ws: Workspace) -> Node:
    pct = [c.id for c in ws.conditions if c.kind == "above" and c.percentile is not None]
    return Node(pct[0], Node(pct[1], Leaf(Action.LONG), Leaf(Action.NO_TRADE)), Leaf(Action.SHORT))


def test_expected_max_sharpe_grows_with_trials() -> None:
    values = [expected_max_sharpe(n, 1.0) for n in (1, 10, 100, 10_000)]
    assert values[0] == 0.0 and values == sorted(values)
    assert expected_max_sharpe(1000, 1.0) == pytest.approx(3.25, abs=0.05)


def test_dsr_separates_skill_from_noise() -> None:
    rng = np.random.default_rng(0)
    strong = rng.normal(0.003, 0.01, 1000)
    noise = rng.normal(0.0, 0.01, 1000)
    assert deflated_sharpe(strong, n_trials=100).dsr > 0.95
    assert deflated_sharpe(noise, n_trials=100).dsr < 0.5
    assert deflated_sharpe(strong, n_trials=10**6).dsr < deflated_sharpe(strong, n_trials=10).dsr
    assert deflated_sharpe(np.zeros(10), 10).dsr == 0.0


def test_monte_carlo_is_seeded_and_consistent() -> None:
    r = np.random.default_rng(1).normal(0.1, 1.0, 300)
    mc, prop = MonteCarloConfig(n_sims=200, confidence=0.95), PropFirmConfig()
    a = run_monte_carlo(r, 0.5, mc, prop, seed=3)
    b = run_monte_carlo(r, 0.5, mc, prop, seed=3)
    assert a == b
    assert a.max_dd_r_at_conf >= a.max_dd_r_median > 0
    assert a.max_dd_pct_at_conf == pytest.approx(a.max_dd_r_at_conf * 0.5)
    assert a.max_risk_pct_for_limit == pytest.approx(prop.max_total_drawdown_pct / a.max_dd_r_at_conf)
    assert run_monte_carlo(np.ones(20), 0.5, mc, prop, seed=1).max_dd_r_at_conf == 0


def test_prop_firm_limits() -> None:
    period = Period(pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2020-03-01", tz="UTC"))
    trades = pd.DataFrame({
        "entry_time": pd.to_datetime(["2020-02-03 01:00", "2020-02-03 02:00", "2020-02-03 03:00"], utc=True),
        "exit_time": pd.to_datetime(["2020-02-03 10:00", "2020-02-03 11:00", "2020-02-05 10:00"], utc=True),
        "r_multiple": [-1.0, -1.0, 2.0],
    })
    res = prop_firm_limits(trades, period, PropFirmConfig(max_daily_loss_pct=5.0))
    assert res.worst_day_r == -2.0 and res.max_concurrent == 3
    assert res.max_risk_pct_daily == pytest.approx(2.5) and res.max_risk_pct_all_stopped == pytest.approx(5 / 3)
    assert prop_firm_limits(trades.iloc[:0], period, PropFirmConfig()).max_risk_pct_daily == float("inf")


def test_permute_bars_keeps_distribution_and_validity() -> None:
    bars = hourly_frame(300)
    bars["close"] = bars["close"] * np.exp(np.random.default_rng(2).normal(0, 0.001, 300).cumsum())
    bars["high"] = bars[["open", "close"]].max(axis=1) + 0.0003
    bars["low"] = bars[["open", "close"]].min(axis=1) - 0.0003
    out = permute_bars(bars, np.random.default_rng(0))
    rets = lambda df: np.sort(np.diff(np.log(df["close"].to_numpy())))  # noqa: E731
    np.testing.assert_allclose(rets(out), rets(bars), atol=1e-12)
    assert (out["high"] >= out[["open", "close"]].max(axis=1) - 1e-12).all()
    assert (out["low"] <= out[["open", "close"]].min(axis=1) + 1e-12).all()
    assert not np.allclose(out["close"], bars["close"])


def test_permuted_universe_excludes_holdout(session_cfg: Config) -> None:
    universe = load_universe(session_cfg, ["EURUSD"])
    perm = permute_universe(universe, session_cfg, np.random.default_rng(0))
    holdout_start = pd.Timestamp(session_cfg.splits.holdout_start, tz="UTC")
    assert perm["EURUSD"].frames["H1"].index.max() < holdout_start
    assert set(perm["EURUSD"].frames) == {"H1", "H4", "D1"}


def test_permutation_test_runs(session_cfg: Config) -> None:
    universe = load_universe(session_cfg)
    res = run_permutation_test(universe, session_cfg, real_best=10.0, seed=1)
    assert len(res.permuted_best) == session_cfg.validation.permutation.n_runs
    assert res.p_value == pytest.approx(1 / 3) and res.n_trials > 0


def test_sensitivity_and_walkforward(ws: Workspace, session_cfg: Config) -> None:
    tree = _tree(ws)
    sens = run_sensitivity(tree, ws, ws.period("validation"), session_cfg)
    assert any(k.endswith("+5") for k in sens.variants) and "exits_x0.8" in sens.variants
    wf = walkforward_tree(tree, ws, session_cfg)
    assert len(wf.windows) == 2
    proc = walkforward_process(ws, session_cfg, seed=0)
    assert len(proc.windows) == 2 and proc.n_trials > 0


def test_battery_report(ws: Workspace, session_cfg: Config) -> None:
    report = run_robustness(_tree(ws), ws, session_cfg, TrialContext(n_trials=50), train_fitness=0.5)
    out = report.to_dict()
    assert set(report.checks) == {"validation_positive", "dsr_train", "walkforward", "sensitivity", "montecarlo", "daily_loss"}
    assert out["prop_firm"]["max_concurrent"] >= 1
    assert set(out["dsr"]) == {"train", "validation"} and "survives" in out


def test_holdout_stage_runs_once(ws: Workspace) -> None:
    ledger = MemoryLedger()
    out = run_holdout_stage(ws, "run1", [_tree(ws)], ledger, prior_trials=100)
    entry = next(iter(out.values()))
    assert entry["dsr"]["n_trials"] == 101
    with pytest.raises(HoldoutLockedError):
        run_holdout_stage(ws, "run1", [_tree(ws)], ledger, prior_trials=100)


def test_nothing_survives_on_pure_noise(tmp_path: Path) -> None:
    directory = tmp_path / "noise"
    write_universe(directory, start="2016-01-03", end="2020-01-01", edge=0.0, seed=11)
    cfg = make_config(directory, tmp_path / "db.sqlite", ga={"population_size": 40, "generations": 4, "n_jobs": 1, "elite_size": 2})
    ws = build_workspace(cfg)
    evo = evolve(ws.split_dataset("train"), ws.conditions, cfg, ws.rules)
    ctx = TrialContext(n_trials=evo.n_trials)
    survivors = [e for e in evo.best(5) if run_robustness(e.tree, ws, cfg, ctx, e.fitness.value).survives]
    assert survivors == []
