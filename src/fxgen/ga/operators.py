"""Random generation, subtree crossover and mutations of decision trees."""

from __future__ import annotations

from collections import defaultdict

import numpy as np

from fxgen.conditions.builder import Condition
from fxgen.tree.node import Action, Leaf, Node, Tree, depth, get_subtree, replace_subtree, simplify, subtrees, trades_anything

LEAF_ACTIONS = (Action.LONG, Action.SHORT, Action.NO_TRADE)
LEAF_WEIGHTS = np.array([0.35, 0.35, 0.30])


class ConditionPool:
    """Condition ids available to the GA, with siblings for threshold mutations."""

    def __init__(self, conditions: list[Condition]) -> None:
        if not conditions:
            raise ValueError("the condition pool is empty")
        self.ids = [c.id for c in conditions]
        self._siblings: dict[str, list[str]] = {}
        groups: dict[tuple[str, str], list[str]] = defaultdict(list)
        for c in conditions:
            if c.kind == "above":
                groups[(c.column, c.kind)].append(c.id)
        for c in conditions:
            if c.kind == "above":
                self._siblings[c.id] = [s for s in groups[(c.column, c.kind)] if s != c.id]

    def random(self, rng: np.random.Generator) -> str:
        return self.ids[int(rng.integers(len(self.ids)))]

    def siblings(self, cid: str) -> list[str]:
        return self._siblings.get(cid, [])


def random_leaf(rng: np.random.Generator, exclude: Action | None = None) -> Leaf:
    actions = [a for a in LEAF_ACTIONS if a != exclude]
    weights = np.array([LEAF_WEIGHTS[LEAF_ACTIONS.index(a)] for a in actions])
    return Leaf(actions[int(rng.choice(len(actions), p=weights / weights.sum()))])


def random_tree(rng: np.random.Generator, pool: ConditionPool, max_depth: int, min_depth: int = 1,
                p_split: float = 0.6) -> Tree:
    """Grow a random tree that trades in at least one branch."""
    def grow(level: int) -> Tree:
        if level >= max_depth or (level >= min_depth and rng.random() > p_split):
            return random_leaf(rng)
        return Node(pool.random(rng), grow(level + 1), grow(level + 1))

    for _ in range(100):
        tree = simplify(grow(0))
        if trades_anything(tree) and depth(tree) >= min(min_depth, max_depth):
            return tree
    return Node(pool.random(rng), Leaf(Action.LONG), Leaf(Action.SHORT))


def _paths(tree: Tree, internal: bool | None = None) -> list[str]:
    out = []
    for path, sub in subtrees(tree):
        if internal is None or isinstance(sub, Node) == internal:
            out.append(path)
    return out


def crossover(a: Tree, b: Tree, rng: np.random.Generator, max_depth: int) -> tuple[Tree, Tree]:
    """Swap a random subtree of ``a`` with a random subtree of ``b`` (depth-bounded)."""
    pa, pb = _paths(a), _paths(b)
    for _ in range(10):
        path_a = pa[int(rng.integers(len(pa)))]
        path_b = pb[int(rng.integers(len(pb)))]
        sub_a, sub_b = get_subtree(a, path_a), get_subtree(b, path_b)
        child_a = simplify(replace_subtree(a, path_a, sub_b))
        child_b = simplify(replace_subtree(b, path_b, sub_a))
        if depth(child_a) <= max_depth and depth(child_b) <= max_depth:
            return child_a, child_b
    return a, b


def mutate_condition(tree: Tree, rng: np.random.Generator, pool: ConditionPool) -> Tree:
    paths = _paths(tree, internal=True)
    if not paths:
        return tree
    path = paths[int(rng.integers(len(paths)))]
    node = get_subtree(tree, path)
    assert isinstance(node, Node)
    return replace_subtree(tree, path, Node(pool.random(rng), node.if_true, node.if_false))


def mutate_threshold(tree: Tree, rng: np.random.Generator, pool: ConditionPool) -> Tree:
    candidates = [p for p in _paths(tree, internal=True) if pool.siblings(get_subtree(tree, p).condition)]  # type: ignore[union-attr]
    if not candidates:
        return mutate_condition(tree, rng, pool)
    path = candidates[int(rng.integers(len(candidates)))]
    node = get_subtree(tree, path)
    assert isinstance(node, Node)
    sibs = pool.siblings(node.condition)
    return replace_subtree(tree, path, Node(sibs[int(rng.integers(len(sibs)))], node.if_true, node.if_false))


def mutate_leaf(tree: Tree, rng: np.random.Generator) -> Tree:
    paths = _paths(tree, internal=False)
    path = paths[int(rng.integers(len(paths)))]
    leaf = get_subtree(tree, path)
    assert isinstance(leaf, Leaf)
    return replace_subtree(tree, path, random_leaf(rng, exclude=leaf.action))


def mutate_prune(tree: Tree, rng: np.random.Generator) -> Tree:
    paths = _paths(tree, internal=True)
    if not paths:
        return tree
    path = paths[int(rng.integers(len(paths)))]
    node = get_subtree(tree, path)
    assert isinstance(node, Node)
    keep = node.if_true if rng.random() < 0.5 else node.if_false
    return replace_subtree(tree, path, keep)


def mutate_grow(tree: Tree, rng: np.random.Generator, pool: ConditionPool, max_depth: int) -> Tree:
    paths = [p for p in _paths(tree, internal=False) if len(p) < max_depth]
    if not paths:
        return mutate_condition(tree, rng, pool)
    path = paths[int(rng.integers(len(paths)))]
    leaf = get_subtree(tree, path)
    assert isinstance(leaf, Leaf)
    other = random_leaf(rng, exclude=leaf.action)
    branch = (leaf, other) if rng.random() < 0.5 else (other, leaf)
    return replace_subtree(tree, path, Node(pool.random(rng), *branch))


def mutate(tree: Tree, rng: np.random.Generator, pool: ConditionPool, max_depth: int) -> Tree:
    """Apply one randomly chosen mutation and keep the result valid."""
    for _ in range(10):
        choice = int(rng.integers(5))
        if choice == 0:
            child = mutate_condition(tree, rng, pool)
        elif choice == 1:
            child = mutate_threshold(tree, rng, pool)
        elif choice == 2:
            child = mutate_leaf(tree, rng)
        elif choice == 3:
            child = mutate_prune(tree, rng)
        else:
            child = mutate_grow(tree, rng, pool, max_depth)
        child = simplify(child)
        if child != tree and trades_anything(child) and depth(child) <= max_depth:
            return child
    return tree
