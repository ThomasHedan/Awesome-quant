"""Command line: ``fxgen synth | datasets | evolve | holdout | report | pine | runs``."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path
from typing import Any

import pandas as pd
import yaml

from fxgen.config import DEFAULT_CONFIG_PATH, PROJECT_ROOT, load_config
from fxgen.data.loader import discover_datasets
from fxgen.data.synthetic import write_universe
from fxgen.errors import FxgenError
from fxgen.indicators.base import all_features
from fxgen.indicators.time import CALENDAR_RULES
from fxgen.logging_setup import setup_logging
from fxgen.pipeline import (
    build_and_report_portfolio,
    classify_stored,
    generate_report,
    run_holdout,
    run_pipeline,
    write_pine,
)
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


def indicators_markdown() -> str:
    """Markdown table of every feature and calendar rule, with the conditions each one generates."""
    families = {"trend": "Tendance", "momentum": "Momentum", "volatility": "Volatilité",
                "structure": "Structure de prix et patterns", "volume": "Volume (ignorés sans volume)"}
    features = all_features()
    lines = [
        "# Bibliothèque d'indicateurs",
        "",
        f"{len(features)} indicateurs, calculés sur chaque timeframe (H1, H4, D1), plus {len(CALENDAR_RULES)} "
        "règles calendaires. Généré par `uv run fxgen indicators`.",
        "",
        "Conditions générées par indicateur : `>pXX` (percentiles du train), `>0` (signé), `:rising`, "
        "`:cross_up` / `:cross_down` (croisement de 0), `>N` et `:cross_up@N` / `:cross_down@N` (niveaux). "
        "Les patterns binaires donnent `>0.5` (présent) et `:cross_up@0.5` (apparu dans les 5 dernières bougies).",
        "",
    ]
    for family, title in families.items():
        rows = [f for f in features if f.family == family]
        lines += [f"## {title} ({len(rows)})", "", "| Nom | Définition | Conditions en plus des percentiles |", "|---|---|---|"]
        for f in rows:
            extra = []
            if f.binary:
                extra = ["pattern 0/1 (pas de percentiles)"]
            else:
                if f.signed:
                    extra.append("`>0`")
                if f.crosses:
                    extra.append("croisement de 0")
                if f.levels:
                    extra.append("niveaux " + ", ".join(f"{lv:g}" for lv in f.levels))
            lines.append(f"| `{f.name}` | {f.label} | {', '.join(extra) or '-'} |")
        lines.append("")
    lines += [f"## Calendrier ({len(CALENDAR_RULES)})", "", "| Nom | Définition |", "|---|---|"]
    lines += [f"| `{r.name}` | {r.label} |" for r in CALENDAR_RULES]
    return "\n".join(lines) + "\n"


def lse_download(args: argparse.Namespace, default_pairs: list[str]) -> dict[str, Any]:
    from fxgen.data.lse import LseClient, download_universe, estimate_rows

    client = LseClient(os.environ.get("LSE_API_KEY", ""))
    today = pd.Timestamp.now(tz="UTC").normalize().tz_localize(None)
    start = args.start or str((today - pd.DateOffset(years=10) + pd.Timedelta(days=2)).date())
    end = args.end or str((today - pd.Timedelta(days=1)).date())
    pairs = args.pairs or default_pairs
    if args.estimate:
        return {p: estimate_rows(client, p, start, end) for p in pairs}
    return download_universe(client, pairs, start, end, args.out, args.years_per_chunk)


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

    lse = sub.add_parser("lse-download", help="download forex H1 candles from London Strategic Edge (key in LSE_API_KEY)")
    lse.add_argument("--pairs", nargs="*", help="pairs (default: data.pairs of the config)")
    lse.add_argument("--start", default=None, help="first day (default: 10 years ago, the free plan window)")
    lse.add_argument("--end", default=None, help="last day (default: yesterday)")
    lse.add_argument("--out", type=Path, default=PROJECT_ROOT / "data" / "lse")
    lse.add_argument("--years-per-chunk", type=float, default=5.0)
    lse.add_argument("--estimate", action="store_true", help="only print the rows each pair would read")

    sub.add_parser("datasets", help="list data files and their quality")
    sub.add_parser("evolve", help="evolve, validate and select finalists (no holdout)")
    hold = sub.add_parser("holdout", help="evaluate a run's finalists on the holdout (once)")
    hold.add_argument("run_id", nargs="?")
    rep = sub.add_parser("report", help="HTML report of a tree")
    rep.add_argument("tree_id")
    pine = sub.add_parser("pine", help="Pine Script v6 export of a tree")
    pine.add_argument("tree_id")
    sub.add_parser("runs", help="list runs")
    sub.add_parser("indicators", help="print the indicator library as Markdown")
    cls = sub.add_parser("classify", help="strategy type of a tree (trend following, mean reversion...)")
    cls.add_argument("tree_id")
    pf = sub.add_parser("portfolio", help="aggregate the best decorrelated strategies")
    pf.add_argument("--runs", nargs="*", help="run ids (default: every finished run)")

    args = parser.parse_args(argv)
    setup_logging(logging.DEBUG if args.verbose else logging.INFO)
    try:
        cfg = load_config(args.config, parse_overrides(args.overrides))
        store = Store(cfg.resolve(cfg.storage.database))
        if args.command == "synth":
            write_universe(args.out, start=args.start, end=args.end, edge=args.edge, seed=args.seed)
        elif args.command == "lse-download":
            _print(lse_download(args, cfg.data.pairs))
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
        elif args.command == "classify":
            _print(classify_stored(store, args.tree_id))
        elif args.command == "portfolio":
            out = build_and_report_portfolio(store, args.runs or None, cfg.portfolio)
            _print({k: out[k] for k in ("portfolio_id", "members", "report")} | {"result": out["result"]["splits"]})
        elif args.command == "indicators":
            sys.stdout.write(indicators_markdown())
    except FxgenError as exc:
        logger.error("%s", exc)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
