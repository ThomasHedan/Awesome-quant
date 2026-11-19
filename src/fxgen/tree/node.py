"""Tree data structure and structural helpers."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass
from enum import IntEnum


class Action(IntEnum):
    SHORT = -1
    NO_TRADE = 0
    LONG = 1


@dataclass(frozen=True)
class Leaf:
    action: Action


@dataclass(frozen=True)
class Node:
    """Internal node: go to ``if_true`` when ``condition`` holds at the decision bar."""

    condition: str
    if_true: Tree
    if_false: Tree


Tree = Node | Leaf


def depth(tree: Tree) -> int:
    """Number of internal levels (a single leaf has depth 0)."""
    if isinstance(tree, Leaf):
        return 0
    return 1 + max(depth(tree.if_true), depth(tree.if_false))


def n_internal(tree: Tree) -> int:
    if isinstance(tree, Leaf):
        return 0
    return 1 + n_internal(tree.if_true) + n_internal(tree.if_false)


def n_nodes(tree: Tree) -> int:
    """Internal nodes plus leaves."""
    if isinstance(tree, Leaf):
        return 1
    return 1 + n_nodes(tree.if_true) + n_nodes(tree.if_false)


def conditions_used(tree: Tree) -> list[str]:
    if isinstance(tree, Leaf):
        return []
    return [tree.condition, *conditions_used(tree.if_true), *conditions_used(tree.if_false)]


def leaves(tree: Tree, path: str = "") -> Iterator[tuple[str, Leaf]]:
    """Yield ``(path, leaf)`` where path is a string of ``T``/``F`` branch choices."""
    if isinstance(tree, Leaf):
        yield path or "root", tree
        return
    yield from leaves(tree.if_true, path + "T")
    yield from leaves(tree.if_false, path + "F")


def trades_anything(tree: Tree) -> bool:
    return any(leaf.action != Action.NO_TRADE for _, leaf in leaves(tree))


def simplify(tree: Tree, known: dict[str, bool] | None = None) -> Tree:
    """Remove dead branches and nodes whose children are identical leaves.

    A condition already decided on the path from the root is resolved, so the
    tree never tests the same condition twice on one path.
    """
    known = known or {}
    if isinstance(tree, Leaf):
        return tree
    if tree.condition in known:
        return simplify(tree.if_true if known[tree.condition] else tree.if_false, known)
    t = simplify(tree.if_true, {**known, tree.condition: True})
    f = simplify(tree.if_false, {**known, tree.condition: False})
    if isinstance(t, Leaf) and isinstance(f, Leaf) and t.action == f.action:
        return t
    return Node(tree.condition, t, f)


def subtrees(tree: Tree, path: str = "") -> Iterator[tuple[str, Tree]]:
    """Every subtree with its path (``""`` is the root)."""
    yield path, tree
    if isinstance(tree, Node):
        yield from subtrees(tree.if_true, path + "T")
        yield from subtrees(tree.if_false, path + "F")


def get_subtree(tree: Tree, path: str) -> Tree:
    for step in path:
        if not isinstance(tree, Node):
            raise KeyError(path)
        tree = tree.if_true if step == "T" else tree.if_false
    return tree


def replace_subtree(tree: Tree, path: str, new: Tree) -> Tree:
    if not path:
        return new
    if not isinstance(tree, Node):
        raise KeyError(path)
    if path[0] == "T":
        return Node(tree.condition, replace_subtree(tree.if_true, path[1:], new), tree.if_false)
    return Node(tree.condition, tree.if_true, replace_subtree(tree.if_false, path[1:], new))
