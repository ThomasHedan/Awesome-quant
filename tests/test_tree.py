from __future__ import annotations

import numpy as np
import pytest

from fxgen.errors import TreeError
from fxgen.tree.evaluate import evaluate, leaf_paths
from fxgen.tree.node import Action, Leaf, Node, depth, n_internal, n_nodes, replace_subtree, simplify, subtrees
from fxgen.tree.serialize import from_dict, from_json, to_dict, to_json, to_rules, tree_hash

L, S, N = Leaf(Action.LONG), Leaf(Action.SHORT), Leaf(Action.NO_TRADE)
TREE = Node("a", Node("b", L, N), S)


def test_structure_helpers() -> None:
    assert depth(TREE) == 2 and n_internal(TREE) == 2 and n_nodes(TREE) == 5
    assert depth(L) == 0
    assert leaf_paths(TREE) == ["TT", "TF", "F"]
    assert [p for p, _ in subtrees(TREE)] == ["", "T", "TT", "TF", "F"]
    assert replace_subtree(TREE, "F", L) == Node("a", Node("b", L, N), L)


def test_json_roundtrip_and_hash() -> None:
    assert from_json(to_json(TREE)) == TREE
    assert to_dict(TREE)["then"]["if"] == "b"
    assert tree_hash(TREE) == tree_hash(from_dict(to_dict(TREE)))
    assert tree_hash(TREE) != tree_hash(Node("a", Node("b", N, L), S))


@pytest.mark.parametrize(
    "bad",
    [{"leaf": "BUY"}, {"if": "a", "then": {"leaf": "LONG"}}, {"if": 3, "then": {"leaf": "LONG"}, "else": {"leaf": "LONG"}}, []],
)
def test_from_dict_rejects_malformed(bad: object) -> None:
    with pytest.raises(TreeError):
        from_dict(bad)  # type: ignore[arg-type]


def test_rules_text() -> None:
    text = to_rules(TREE, describe=lambda c: c.upper())
    assert text.splitlines()[0] == "IF A:"
    assert "-> LONG  [branch TT]" in text and "ELSE:" in text


def test_simplify_resolves_repeated_condition_and_equal_leaves() -> None:
    assert simplify(Node("a", Node("a", L, S), S)) == Node("a", L, S)
    assert simplify(Node("a", L, L)) == L
    assert simplify(Node("a", Node("b", S, S), L)) == Node("a", S, L)


def test_evaluate_masks() -> None:
    X = np.array([[1, 1], [1, 0], [0, 1], [0, 0]], dtype=bool)
    cols = {"a": X[:, 0], "b": X[:, 1]}
    action, leaf = evaluate(TREE, cols.__getitem__, 4)
    assert action.tolist() == [1, 0, -1, -1]
    assert leaf.tolist() == [0, 1, 2, 2]
    with pytest.raises(KeyError):
        evaluate(Node("zz", L, S), cols.__getitem__, 4)
