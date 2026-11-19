"""JSON (de)serialisation, canonical hashing and readable if/else rules."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable
from typing import Any

from fxgen.errors import TreeError
from fxgen.tree.node import Action, Leaf, Node, Tree


def to_dict(tree: Tree) -> dict[str, Any]:
    if isinstance(tree, Leaf):
        return {"leaf": tree.action.name}
    return {"if": tree.condition, "then": to_dict(tree.if_true), "else": to_dict(tree.if_false)}


def from_dict(data: dict[str, Any]) -> Tree:
    """Parse ``{"if": cond, "then": ..., "else": ...}`` / ``{"leaf": "LONG"}``."""
    if not isinstance(data, dict):
        raise TreeError(f"tree node must be an object, got {type(data).__name__}")
    if "leaf" in data:
        name = str(data["leaf"]).upper()
        if name not in Action.__members__:
            raise TreeError(f"invalid leaf {data['leaf']!r}; use LONG, SHORT or NO_TRADE")
        return Leaf(Action[name])
    missing = {"if", "then", "else"} - set(data)
    if missing:
        raise TreeError(f"internal node is missing keys {sorted(missing)}")
    if not isinstance(data["if"], str) or not data["if"]:
        raise TreeError("node 'if' must be a condition id string")
    return Node(data["if"], from_dict(data["then"]), from_dict(data["else"]))


def to_json(tree: Tree) -> str:
    return json.dumps(to_dict(tree), sort_keys=True, separators=(",", ":"))


def from_json(text: str) -> Tree:
    try:
        return from_dict(json.loads(text))
    except json.JSONDecodeError as exc:
        raise TreeError(f"tree JSON is not valid JSON: {exc}") from exc


def tree_hash(tree: Tree) -> str:
    """Stable 16-hex id of a tree's structure."""
    return hashlib.sha256(to_json(tree).encode()).hexdigest()[:16]


def to_rules(tree: Tree, describe: Callable[[str], str] | None = None, indent: str = "    ") -> str:
    """Readable nested IF / ELSE rules."""
    describe = describe or (lambda cid: cid)
    lines: list[str] = []

    def walk(node: Tree, level: int, path: str) -> None:
        pad = indent * level
        if isinstance(node, Leaf):
            lines.append(f"{pad}-> {node.action.name}  [branch {path or 'root'}]")
            return
        lines.append(f"{pad}IF {describe(node.condition)}:")
        walk(node.if_true, level + 1, path + "T")
        lines.append(f"{pad}ELSE:")
        walk(node.if_false, level + 1, path + "F")

    walk(tree, 0, "")
    return "\n".join(lines)
