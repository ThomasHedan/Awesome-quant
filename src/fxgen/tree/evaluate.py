"""Vectorised evaluation of a tree on a boolean condition matrix."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from fxgen.tree.node import Leaf, Tree, leaves


def leaf_paths(tree: Tree) -> list[str]:
    return [path for path, _ in leaves(tree)]


def evaluate(tree: Tree, column: Callable[[str], np.ndarray], n: int) -> tuple[np.ndarray, np.ndarray]:
    """Return per-bar ``(action, leaf_index)`` arrays at the decision bar.

    ``column(cid)`` returns the boolean array of a condition (length ``n``).
    ``action`` is int8 in {-1, 0, 1}; ``leaf_index`` indexes ``leaf_paths(tree)``.
    """
    action = np.zeros(n, dtype=np.int8)
    leaf_idx = np.zeros(n, dtype=np.int16)
    counter = [0]

    def walk(node: Tree, mask: np.ndarray) -> None:
        if isinstance(node, Leaf):
            action[mask] = int(node.action)
            leaf_idx[mask] = counter[0]
            counter[0] += 1
            return
        col = column(node.condition)
        walk(node.if_true, mask & col)
        walk(node.if_false, mask & ~col)

    walk(tree, np.ones(n, dtype=bool))
    return action, leaf_idx
