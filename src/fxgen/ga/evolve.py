"""Generational GA: tournament selection, subtree crossover, mutations, elitism, sharing.

The GA only ever receives a dataset whose period ends before the holdout; it is
refused otherwise. Every unique tree evaluated is counted as a trial for the
Deflated Sharpe Ratio, and its train Sharpe is kept for the trial variance.
"""

from __future__ import annotations

import logging
import multiprocessing as mp
from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fxgen.backtest.engine import ExitRules
from fxgen.backtest.runner import run_backtest
from fxgen.conditions.builder import Condition
from fxgen.config import Config
from fxgen.data.splits import split_period
from fxgen.errors import HoldoutLockedError
from fxgen.ga.diversity import phenotype, shared_fitness
from fxgen.ga.fitness import FitnessResult, compute_fitness
from fxgen.ga.operators import ConditionPool, crossover, mutate, random_tree
from fxgen.market import Dataset
from fxgen.tree.evaluate import evaluate
from fxgen.tree.node import Tree
from fxgen.tree.serialize import tree_hash

logger = logging.getLogger(__name__)


@dataclass
class Evaluation:
    tree: Tree
    hash: str
    fitness: FitnessResult
    phenotype: np.ndarray


@dataclass
class GenerationStats:
    generation: int
    best: float
    median: float
    n_valid: int
    n_unique: int
    n_trials: int


@dataclass
class EvolutionResult:
    evaluations: dict[str, Evaluation]
    history: list[GenerationStats]
    cancelled: bool = False

    @property
    def n_trials(self) -> int:
        return len(self.evaluations)

    def trial_sharpes(self) -> np.ndarray:
        return np.array([e.fitness.sharpe for e in self.evaluations.values()])

    def best(self, n: int) -> list[Evaluation]:
        valid = [e for e in self.evaluations.values() if e.fitness.valid]
        return sorted(valid, key=lambda e: (-e.fitness.value, e.hash))[:n]

    def history_frame(self) -> pd.DataFrame:
        return pd.DataFrame([vars(h) for h in self.history])


_WORKER: dict[str, object] = {}


def _init_worker(dataset: Dataset, cfg: Config, rules: ExitRules) -> None:
    _WORKER.update(dataset=dataset, cfg=cfg, rules=rules)


def _evaluate(tree: Tree) -> Evaluation:
    dataset: Dataset = _WORKER["dataset"]  # type: ignore[assignment]
    cfg: Config = _WORKER["cfg"]  # type: ignore[assignment]
    rules: ExitRules = _WORKER["rules"]  # type: ignore[assignment]
    result = run_backtest(tree, dataset, rules, cfg.costs, cfg.backtest.risk_per_trade_pct)
    columns = dataset.column_of()
    actions = [evaluate(tree, p.X, columns)[0] for p in dataset.pairs.values()]
    fitness = compute_fitness(tree, result, dataset.period, cfg.fitness)
    return Evaluation(tree, tree_hash(tree), fitness, phenotype(actions))


def check_not_holdout(dataset: Dataset, cfg: Config) -> None:
    holdout_start = split_period(cfg.splits, "holdout", allow_holdout=True).start
    if dataset.period.end > holdout_start:
        raise HoldoutLockedError(
            f"optimisation refused: dataset period {dataset.period.label()} overlaps the locked holdout "
            f"(starts {holdout_start.date()})"
        )


def _tournament(rng: np.random.Generator, scores: np.ndarray, size: int) -> int:
    contenders = rng.choice(len(scores), size=min(size, len(scores)), replace=False)
    return int(contenders[np.argmax(scores[contenders])])


def evolve(
    dataset: Dataset,
    conditions: list[Condition],
    cfg: Config,
    rules: ExitRules,
    seed: int | None = None,
    population_size: int | None = None,
    generations: int | None = None,
    on_generation: Callable[[GenerationStats], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> EvolutionResult:
    """Run the GA on ``dataset`` (train data only) and return every evaluated tree."""
    check_not_holdout(dataset, cfg)
    ga = cfg.ga
    pop_size = population_size or ga.population_size
    n_gen = generations or ga.generations
    rng = np.random.default_rng(cfg.seed if seed is None else seed)
    pool = ConditionPool(conditions)
    max_depth = cfg.tree.max_depth
    cache: dict[str, Evaluation] = {}
    history: list[GenerationStats] = []
    n_jobs = mp.cpu_count() if ga.n_jobs == -1 else max(1, ga.n_jobs)

    worker_pool = None
    if n_jobs > 1:
        method = "fork" if "fork" in mp.get_all_start_methods() else "spawn"
        worker_pool = mp.get_context(method).Pool(n_jobs, initializer=_init_worker, initargs=(dataset, cfg, rules))
    else:
        _init_worker(dataset, cfg, rules)

    def evaluate_all(trees: list[Tree]) -> list[Evaluation]:
        todo = {tree_hash(t): t for t in trees if tree_hash(t) not in cache}
        if todo:
            items = list(todo.values())
            results = worker_pool.map(_evaluate, items, chunksize=max(1, len(items) // (4 * n_jobs))) if worker_pool else [
                _evaluate(t) for t in items
            ]
            cache.update({e.hash: e for e in results})
        return [cache[tree_hash(t)] for t in trees]

    cancelled = False
    try:
        population = [random_tree(rng, pool, max_depth, cfg.tree.min_depth) for _ in range(pop_size)]
        for gen in range(n_gen):
            evals = evaluate_all(population)
            raw = np.array([e.fitness.value for e in evals])
            shared = shared_fitness(raw, np.stack([e.phenotype for e in evals]), ga.sharing_radius)
            stats = GenerationStats(
                generation=gen,
                best=float(raw.max()),
                median=float(np.median(raw)),
                n_valid=int(sum(e.fitness.valid for e in evals)),
                n_unique=len({e.hash for e in evals}),
                n_trials=len(cache),
            )
            history.append(stats)
            logger.info(
                "gen %d: best=%.3f median=%.3f valid=%d unique=%d trials=%d",
                gen, stats.best, stats.median, stats.n_valid, stats.n_unique, stats.n_trials,
            )
            if on_generation:
                on_generation(stats)
            if should_stop and should_stop():
                cancelled = True
                break
            if gen == n_gen - 1:
                break
            population = _next_generation(population, evals, raw, shared, rng, pool, cfg)
    finally:
        if worker_pool:
            worker_pool.close()
            worker_pool.join()
    return EvolutionResult(evaluations=cache, history=history, cancelled=cancelled)


def _next_generation(
    population: list[Tree],
    evals: list[Evaluation],
    raw: np.ndarray,
    shared: np.ndarray,
    rng: np.random.Generator,
    pool: ConditionPool,
    cfg: Config,
) -> list[Tree]:
    ga, max_depth = cfg.ga, cfg.tree.max_depth
    elites: list[Tree] = []
    seen: set[str] = set()
    for i in np.argsort(-raw, kind="stable"):
        if len(elites) >= ga.elite_size:
            break
        if evals[i].hash not in seen and evals[i].fitness.valid:
            elites.append(population[i])
            seen.add(evals[i].hash)
    children: list[Tree] = list(elites)
    while len(children) < len(population):
        a = population[_tournament(rng, shared, ga.tournament_size)]
        b = population[_tournament(rng, shared, ga.tournament_size)]
        if rng.random() < ga.crossover_prob:
            a, b = crossover(a, b, rng, max_depth)
        for child in (a, b):
            if rng.random() < ga.mutation_prob:
                child = mutate(child, rng, pool, max_depth)
            for _ in range(3):
                if tree_hash(child) not in seen:
                    break
                child = mutate(child, rng, pool, max_depth)
            if tree_hash(child) in seen and rng.random() < 0.8:
                child = random_tree(rng, pool, max_depth, cfg.tree.min_depth)
            seen.add(tree_hash(child))
            children.append(child)
    return children[: len(population)]


