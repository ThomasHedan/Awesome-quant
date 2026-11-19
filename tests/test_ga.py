from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxgen.backtest.runner import BacktestResult
from fxgen.config import Config, FitnessConfig
from fxgen.data.splits import Period
from fxgen.errors import HoldoutLockedError
from fxgen.ga.diversity import phenotype, shared_fitness, similarity_matrix
from fxgen.ga.evolve import evolve
from fxgen.ga.fitness import compute_fitness, concentration
from fxgen.ga.operators import ConditionPool, crossover, mutate, random_tree
from fxgen.conditions.builder import Condition
from fxgen.tree.node import Action, Leaf, Node, depth, trades_anything
from fxgen.tree.serialize import tree_hash
from fxgen.workspace import build_workspace

COND = [
    Condition("H1.rsi14>p25", "momentum", "H1", "rsi14", "above", 25, 40.0, "RSI"),
    Condition("H1.rsi14>p75", "momentum", "H1", "rsi14", "above", 75, 60.0, "RSI"),
    Condition("H1.adx14:rising", "trend", "H1", "adx14", "rising", label="ADX"),
    Condition("H1.cal:day_monday", "time", "H1", "cal", "calendar", label="Monday"),
]
PERIOD = Period(pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2022-01-01", tz="UTC"))


def test_pool_siblings() -> None:
    pool = ConditionPool(COND)
    assert pool.siblings("H1.rsi14>p25") == ["H1.rsi14>p75"]
    assert pool.siblings("H1.adx14:rising") == []


@pytest.mark.parametrize("max_depth", [1, 2, 3, 4])
def test_random_trees_and_operators_respect_depth(max_depth: int) -> None:
    rng = np.random.default_rng(0)
    pool = ConditionPool(COND)
    trees = [random_tree(rng, pool, max_depth) for _ in range(50)]
    assert all(1 <= depth(t) <= max_depth and trades_anything(t) for t in trees)
    for a, b in zip(trees, trees[1:]):
        ca, cb = crossover(a, b, rng, max_depth)
        assert depth(ca) <= max_depth and depth(cb) <= max_depth
        m = mutate(a, rng, pool, max_depth)
        assert depth(m) <= max_depth


def test_mutation_changes_tree() -> None:
    rng = np.random.default_rng(1)
    pool = ConditionPool(COND)
    tree = Node("H1.rsi14>p25", Leaf(Action.LONG), Leaf(Action.SHORT))
    changed = sum(mutate(tree, rng, pool, 3) != tree for _ in range(30))
    assert changed >= 28


def _result(per_pair_r: dict[str, list[float]]) -> BacktestResult:
    rows = []
    per_pair = {}
    for k, (pair, rs) in enumerate(per_pair_r.items()):
        times = pd.date_range("2020-01-06", periods=len(rs), freq="3B", tz="UTC") + pd.Timedelta(hours=k)
        rows.append(pd.DataFrame({"pair": pair, "exit_time": times, "r_multiple": rs, "bars_held": 5, "direction": 1}))
        per_pair[pair] = {"n_trades": len(rs), "total_r": float(sum(rs))}
    trades = pd.concat(rows, ignore_index=True).sort_values("exit_time", ignore_index=True)
    return BacktestResult(trades=trades, metrics={}, per_pair=per_pair)


def test_fitness_minimum_trades_gives_zero() -> None:
    cfg = FitnessConfig(min_trades_total=50, min_trades_per_pair=5)
    tree = Node("a", Leaf(Action.LONG), Leaf(Action.SHORT))
    res = compute_fitness(tree, _result({"A": [1.0] * 30, "B": [1.0] * 30}), PERIOD, cfg)
    assert res.value > 0
    res = compute_fitness(tree, _result({"A": [1.0] * 60, "B": [1.0] * 3}), PERIOD, cfg)
    assert res.value == 0 and not res.valid


def test_fitness_penalises_single_pair_dependence_and_complexity() -> None:
    cfg = FitnessConfig(min_trades_total=10, min_trades_per_pair=1, max_pair_pnl_share=0.5, n_subperiods=2, min_positive_subperiods=1)
    rng = np.random.default_rng(0)
    balanced = {p: list(rng.normal(0.3, 1, 40)) for p in "ABCD"}
    lopsided = {"A": list(rng.normal(1.5, 1, 40)), **{p: list(rng.normal(0.0, 0.01, 40)) for p in "BCD"}}
    tree = Node("a", Leaf(Action.LONG), Leaf(Action.SHORT))
    big = Node("a", Node("b", Leaf(Action.LONG), Leaf(Action.NO_TRADE)), Leaf(Action.SHORT))
    fb = compute_fitness(tree, _result(balanced), PERIOD, cfg)
    fl = compute_fitness(tree, _result(lopsided), PERIOD, cfg)
    assert fl.components["pair_concentration"] < 1.0 and fb.components["pair_concentration"] == 1.0
    assert compute_fitness(big, _result(balanced), PERIOD, cfg).value < fb.value
    assert concentration({"A": 3.0, "B": 1.0, "C": -5.0}) == pytest.approx(0.75)


def test_similarity_and_sharing() -> None:
    p = np.array([[1, 1, 0, -1], [1, 1, 0, -1], [-1, -1, 0, 1], [0, 1, 1, 0]], dtype=np.int8)
    sim = similarity_matrix(p)
    assert sim[0, 1] == pytest.approx(1.0) and sim[0, 2] == pytest.approx(0.0)
    assert sim[0, 3] == pytest.approx(1 / 4)
    shared = shared_fitness(np.array([2.0, 2.0, 2.0, 2.0]), p, radius=0.8)
    assert shared[0] == pytest.approx(1.0) and shared[2] == pytest.approx(2.0)
    assert phenotype([np.arange(30, dtype=np.int8)]).tolist() == [0, 24]


def test_evolve_is_deterministic_counts_trials_and_matches_parallel(cfg: Config) -> None:
    ws = build_workspace(cfg)
    train = ws.split_dataset("train")
    a = evolve(train, ws.conditions, cfg, ws.rules, seed=3)
    b = evolve(train, ws.conditions, cfg, ws.rules, seed=3)
    assert [e.hash for e in a.best(5)] == [e.hash for e in b.best(5)]
    assert a.n_trials == len(set(a.evaluations)) and a.n_trials >= cfg.ga.population_size
    assert len(a.history) == cfg.ga.generations
    assert a.history[-1].best >= a.history[0].best - 1e-9
    parallel_cfg = cfg.model_copy(update={"ga": cfg.ga.model_copy(update={"n_jobs": 2})})
    c = evolve(train, ws.conditions, parallel_cfg, ws.rules, seed=3)
    assert [e.hash for e in a.best(5)] == [e.hash for e in c.best(5)]
    assert all(tree_hash(e.tree) == e.hash for e in a.best(5))


def test_evolve_refuses_holdout(cfg: Config) -> None:
    ws = build_workspace(cfg)
    leaky = ws.dataset(Period(ws.train.start, ws.period("holdout", allow_holdout=True).end))
    with pytest.raises(HoldoutLockedError):
        evolve(leaky, ws.conditions, cfg, ws.rules)


def test_evolve_can_be_cancelled(cfg: Config) -> None:
    ws = build_workspace(cfg)
    res = evolve(ws.split_dataset("train"), ws.conditions, cfg, ws.rules, should_stop=lambda: True)
    assert res.cancelled and len(res.history) == 1
