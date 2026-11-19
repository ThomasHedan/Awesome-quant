"""Command line: ``fxgen synth | datasets | evolve | holdout | report | pine | runs``."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import yaml

from fxgen.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from fxgen.data.loader import discover_datasets
from fxgen.data.synthetic import write_universe
from fxgen.errors import FxgenError
from fxgen.logging_setup import setup_logging
from fxgen.pipeline import generate_report, run_holdout, run_pipeline, write_pine
from fxgen.storage.db import Store

logger = logging.getLogger("fxgen")


def parse_overrides(items: list[str]) -> dict[str, Any]:
    """``a.b=3`` items -> nested dict (values parsed as YAML)."""
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise FxgenError(f"override {item!r} must look like section.key=value")
        key, raw = item.split("=", 1)
        node = out
        parts = key.split(".")
        for part in parts[:-1]:
            node = node.setdefault(part, {})
        node[parts[-1]] = yaml.safe_load(raw)
    return out


def _print(data: Any) -> None:
    sys.stdout.write(json.dumps(data, indent=2, default=str) + "\n")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="fxgen", description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument("--set", dest="overrides", action="append", default=[], help="override, e.g. ga.generations=10")
    parser.add_argument("-v", "--verbose", action="store_true")
    sub = parser.add_subparsers(dest="command", required=True)

    synth = sub.add_parser("synth", help="write the synthetic demo dataset")
    synth.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "synthetic")
    synth.add_argument("--edge", type=float, default=0.02, help="planted momentum strength (0 = pure noise)")
    synth.add_argument("--seed", type=int, default=42)
    synth.add_argument("--start", default="2015-01-04")
    synth.add_argument("--end", default="2025-01-01")

    sub.add_parser("datasets", help="list data files and their quality")
    sub.add_parser("evolve", help="evolve, validate and select finalists (no holdout)")
    hold = sub.add_parser("holdout", help="evaluate a run's finalists on the holdout (once)")
    hold.add_argument("run_id", nargs="?")
    rep = sub.add_parser("report", help="HTML report of a tree")
    rep.add_argument("tree_id")
    pine = sub.add_parser("pine", help="Pine Script v6 export of a tree")
    pine.add_argument("tree_id")
    sub.add_parser("runs", help="list runs")

    args = parser.parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        cfg = load_config(args.config, parse_overrides(args.overrides))
        store = Store(cfg.resolve(cfg.storage.database))
        if args.command == "synth":
            write_universe(args.out, start=args.start, end=args.end, edge=args.edge, seed=args.seed)
        elif args.command == "datasets":
            _print(discover_datasets(cfg))
        elif args.command == "evolve":
            run_id = run_pipeline(cfg, store)
            _print({"run_id": run_id, "ranking": str(cfg.resolve(cfg.storage.output_dir) / run_id / "ranking.csv")})
        elif args.command == "holdout":
            run_id = args.run_id or store.latest_run()
            if run_id is None:
                raise FxgenError("no finished run; run `fxgen evolve` first")
            _print({"run_id": run_id, "holdout": run_holdout(store, run_id)})
        elif args.command == "report":
            _print({"report": str(generate_report(store, args.tree_id))})
        elif args.command == "pine":
            _print({"pine": str(write_pine(store, args.tree_id))})
        elif args.command == "runs":
            _print(store.list_runs())
    except FxgenError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
