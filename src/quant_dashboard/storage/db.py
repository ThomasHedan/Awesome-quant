"""SQLite-backed run store.

Every backtest is persisted with its full configuration (source, symbol,
timeframe, window, strategy, params, costs), the computed metrics, and a
deterministic ``config_hash`` so duplicate runs can be detected. Monte Carlo
and optimization runs link to their parent backtest where applicable.

Designed for a single-user local workflow — no concurrent writers, no
migrations. Re-creating the DB from scratch is cheap if the schema ever
needs to change.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from quant_dashboard.engine.metrics import Metrics
from quant_dashboard.engine.montecarlo import MonteCarloResult
from quant_dashboard.engine.optimizer import OptimizationResult
from quant_dashboard.engine.runner import BacktestResult


class StorageError(RuntimeError):
    """User-actionable storage errors."""


_SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    start_ts        TEXT,
    end_ts          TEXT,
    strategy_name   TEXT    NOT NULL,
    params_json     TEXT    NOT NULL,
    costs_json      TEXT    NOT NULL,
    spec_json       TEXT    NOT NULL,
    metrics_json    TEXT    NOT NULL,
    config_hash     TEXT    NOT NULL,
    notes           TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_runs_strategy_symbol
    ON runs(strategy_name, symbol);
CREATE INDEX IF NOT EXISTS idx_runs_hash ON runs(config_hash);

CREATE TABLE IF NOT EXISTS monte_carlo_runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    run_id          INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
    method          TEXT    NOT NULL,
    n_simulations   INTEGER NOT NULL,
    seed            INTEGER NOT NULL,
    goal_return     REAL    NOT NULL,
    ruin_threshold  REAL    NOT NULL,
    summary_json    TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_mc_run_id ON monte_carlo_runs(run_id);

CREATE TABLE IF NOT EXISTS optimizations (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT    NOT NULL,
    source          TEXT    NOT NULL,
    symbol          TEXT    NOT NULL,
    timeframe       TEXT    NOT NULL,
    start_ts        TEXT,
    end_ts          TEXT,
    strategy_name   TEXT    NOT NULL,
    metric          TEXT    NOT NULL,
    grids_json      TEXT    NOT NULL,
    n_combos        INTEGER NOT NULL,
    best_params_json    TEXT NOT NULL,
    best_metric_value   REAL NOT NULL,
    rankings_json   TEXT    NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_opt_strategy_symbol
    ON optimizations(strategy_name, symbol);
"""


def _json_default(o: Any) -> Any:
    if isinstance(o, (pd.Timestamp, datetime)):
        return o.isoformat()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    if is_dataclass(o):
        return asdict(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    if hasattr(o, "item"):
        return o.item()
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def _dumps(obj: Any) -> str:
    return json.dumps(obj, default=_json_default, sort_keys=True)


def _ts_to_iso(ts: Any) -> str | None:
    if ts is None:
        return None
    if isinstance(ts, pd.Timestamp):
        return ts.isoformat()
    return pd.Timestamp(ts).isoformat()


def _config_hash(payload: dict) -> str:
    blob = _dumps(payload).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


# ---------------------------------------------------------------------------


class RunStore:
    """Thin SQLite wrapper. Open one per process; safe for sequential use."""

    def __init__(self, path: str | Path = "runs.db") -> None:
        self.path = Path(path).expanduser()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._init_schema()

    # --- low-level connection management --------------------------------

    @contextmanager
    def _connect(self):
        # Open per-call so the store is safe to use from short-lived dashboard
        # callbacks without worrying about a long-lived connection.
        conn = sqlite3.connect(str(self.path))
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON;")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_schema(self) -> None:
        with self._connect() as conn:
            conn.executescript(_SCHEMA)

    # --- write API -------------------------------------------------------

    def save_backtest(
        self,
        result: BacktestResult,
        metrics: Metrics,
        notes: str = "",
    ) -> int:
        df_idx = result.data.df.index
        start_ts = _ts_to_iso(df_idx[0]) if len(df_idx) else None
        end_ts = _ts_to_iso(df_idx[-1]) if len(df_idx) else None

        params = result.params
        costs = asdict(result.costs)
        spec = asdict(result.data.spec)
        metrics_d = metrics.to_dict()

        config = {
            "source": result.data.source,
            "symbol": result.data.spec.symbol,
            "timeframe": result.data.timeframe,
            "start": start_ts,
            "end": end_ts,
            "strategy": result.strategy_name,
            "params": params,
            "costs": costs,
        }
        h = _config_hash(config)

        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO runs (
                    created_at, source, symbol, timeframe, start_ts, end_ts,
                    strategy_name, params_json, costs_json, spec_json,
                    metrics_json, config_hash, notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    result.data.source,
                    result.data.spec.symbol,
                    result.data.timeframe,
                    start_ts,
                    end_ts,
                    result.strategy_name,
                    _dumps(params),
                    _dumps(costs),
                    _dumps(spec),
                    _dumps(metrics_d),
                    h,
                    notes,
                ),
            )
            return int(cur.lastrowid)

    def save_monte_carlo(self, run_id: int, mc: MonteCarloResult) -> int:
        now = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            row = conn.execute("SELECT id FROM runs WHERE id = ?", (run_id,)).fetchone()
            if row is None:
                raise StorageError(f"no parent backtest run id={run_id}")
            cur = conn.execute(
                """
                INSERT INTO monte_carlo_runs (
                    created_at, run_id, method, n_simulations, seed,
                    goal_return, ruin_threshold, summary_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    run_id,
                    mc.method,
                    mc.n_simulations,
                    mc.seed,
                    mc.goal_return,
                    mc.ruin_threshold,
                    _dumps(mc.to_summary()),
                ),
            )
            return int(cur.lastrowid)

    def save_optimization(
        self,
        opt: OptimizationResult,
        *,
        source: str,
        symbol: str,
        timeframe: str,
        start: Any | None,
        end: Any | None,
        grids: dict[str, Iterable[Any]],
    ) -> int:
        now = datetime.now(timezone.utc).isoformat()
        # rankings_json keeps the full table so the dashboard can rebuild
        # heatmaps without re-running the search.
        rankings_payload = opt.rankings.to_dict(orient="records")
        grids_payload = {k: list(v) for k, v in grids.items()}
        with self._connect() as conn:
            cur = conn.execute(
                """
                INSERT INTO optimizations (
                    created_at, source, symbol, timeframe, start_ts, end_ts,
                    strategy_name, metric, grids_json, n_combos,
                    best_params_json, best_metric_value, rankings_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    now,
                    source,
                    symbol,
                    timeframe,
                    _ts_to_iso(start),
                    _ts_to_iso(end),
                    opt.strategy_name,
                    opt.metric,
                    _dumps(grids_payload),
                    opt.n_combos,
                    _dumps(opt.best_params),
                    float(opt.best_metric_value),
                    _dumps(rankings_payload),
                ),
            )
            return int(cur.lastrowid)

    # --- read API --------------------------------------------------------

    def list_runs(
        self,
        *,
        strategy_name: str | None = None,
        symbol: str | None = None,
        limit: int = 100,
    ) -> pd.DataFrame:
        where, args = [], []
        if strategy_name:
            where.append("strategy_name = ?")
            args.append(strategy_name)
        if symbol:
            where.append("symbol = ?")
            args.append(symbol)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        sql = (
            "SELECT id, created_at, source, symbol, timeframe, start_ts, end_ts, "
            "strategy_name, config_hash, notes FROM runs "
            f"{clause} ORDER BY id DESC LIMIT ?"
        )
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(args)).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_run(self, run_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM runs WHERE id = ?", (run_id,)
            ).fetchone()
            if row is None:
                raise StorageError(f"no run with id={run_id}")
            mc_rows = conn.execute(
                "SELECT * FROM monte_carlo_runs WHERE run_id = ? ORDER BY id DESC",
                (run_id,),
            ).fetchall()
        out = dict(row)
        for key in ("params_json", "costs_json", "spec_json", "metrics_json"):
            out[key.removesuffix("_json")] = json.loads(out.pop(key))
        out["monte_carlo_runs"] = [
            {**dict(r), "summary": json.loads(r["summary_json"])} for r in mc_rows
        ]
        return out

    def list_optimizations(
        self,
        *,
        strategy_name: str | None = None,
        symbol: str | None = None,
        limit: int = 50,
    ) -> pd.DataFrame:
        where, args = [], []
        if strategy_name:
            where.append("strategy_name = ?")
            args.append(strategy_name)
        if symbol:
            where.append("symbol = ?")
            args.append(symbol)
        clause = ("WHERE " + " AND ".join(where)) if where else ""
        sql = (
            "SELECT id, created_at, source, symbol, timeframe, strategy_name, "
            "metric, n_combos, best_metric_value FROM optimizations "
            f"{clause} ORDER BY id DESC LIMIT ?"
        )
        args.append(limit)
        with self._connect() as conn:
            rows = conn.execute(sql, tuple(args)).fetchall()
        return pd.DataFrame([dict(r) for r in rows])

    def get_optimization(self, opt_id: int) -> dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT * FROM optimizations WHERE id = ?", (opt_id,)
            ).fetchone()
            if row is None:
                raise StorageError(f"no optimization with id={opt_id}")
        out = dict(row)
        out["grids"] = json.loads(out.pop("grids_json"))
        out["best_params"] = json.loads(out.pop("best_params_json"))
        out["rankings"] = pd.DataFrame(json.loads(out.pop("rankings_json")))
        return out

    def delete_run(self, run_id: int) -> None:
        with self._connect() as conn:
            conn.execute("DELETE FROM runs WHERE id = ?", (run_id,))
