"""SQLite storage for runs, trees, metrics, trades, robustness, jobs and the trial ledger.

The database is the single source of truth shared by the pipeline (writer,
possibly in a background job process) and the MCP server (reader). WAL mode
lets both work concurrently.
"""

from __future__ import annotations

import json
import math
import sqlite3
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from fxgen.errors import StorageError

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    status TEXT NOT NULL,
    config_json TEXT NOT NULL,
    config_hash TEXT NOT NULL,
    seed INTEGER NOT NULL,
    n_trials INTEGER DEFAULT 0,
    holdout_done INTEGER DEFAULT 0,
    permutation_json TEXT,
    walkforward_json TEXT,
    history_json TEXT,
    error TEXT
);
CREATE TABLE IF NOT EXISTS trees (
    tree_id TEXT PRIMARY KEY,
    tree_json TEXT NOT NULL,
    rules TEXT NOT NULL,
    depth INTEGER NOT NULL,
    n_internal INTEGER NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS run_trees (
    run_id TEXT NOT NULL,
    tree_id TEXT NOT NULL,
    train_fitness REAL,
    fitness_json TEXT,
    is_candidate INTEGER DEFAULT 0,
    is_finalist INTEGER DEFAULT 0,
    survives INTEGER,
    strategy_type TEXT,
    classification_json TEXT,
    PRIMARY KEY (run_id, tree_id)
);
CREATE TABLE IF NOT EXISTS metrics (
    run_id TEXT NOT NULL,
    tree_id TEXT NOT NULL,
    split TEXT NOT NULL,
    pair TEXT NOT NULL,
    sharpe REAL,
    dsr REAL,
    n_trades INTEGER,
    metrics_json TEXT NOT NULL,
    PRIMARY KEY (run_id, tree_id, split, pair)
);
CREATE TABLE IF NOT EXISTS yearly (
    run_id TEXT NOT NULL,
    tree_id TEXT NOT NULL,
    split TEXT NOT NULL,
    year INTEGER NOT NULL,
    n_trades INTEGER,
    total_r REAL,
    return_pct REAL,
    win_rate REAL,
    PRIMARY KEY (run_id, tree_id, split, year)
);
CREATE TABLE IF NOT EXISTS trades (
    run_id TEXT NOT NULL,
    tree_id TEXT NOT NULL,
    split TEXT NOT NULL,
    pair TEXT NOT NULL,
    entry_time TEXT NOT NULL,
    exit_time TEXT NOT NULL,
    direction INTEGER NOT NULL,
    entry_price REAL,
    exit_price REAL,
    bars_held INTEGER,
    net_pips REAL,
    cost_pips REAL,
    r_multiple REAL,
    exit_reason TEXT,
    branch TEXT,
    move_1d REAL,
    move_5d REAL
);
CREATE INDEX IF NOT EXISTS trades_tree ON trades (tree_id, split, pair);
CREATE TABLE IF NOT EXISTS robustness (
    run_id TEXT NOT NULL,
    tree_id TEXT NOT NULL,
    survives INTEGER NOT NULL,
    report_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    PRIMARY KEY (run_id, tree_id)
);
CREATE TABLE IF NOT EXISTS run_conditions (
    run_id TEXT NOT NULL,
    cond_id TEXT NOT NULL,
    family TEXT NOT NULL,
    timeframe TEXT NOT NULL,
    kind TEXT NOT NULL,
    description TEXT NOT NULL,
    activation REAL,
    condition_json TEXT NOT NULL,
    PRIMARY KEY (run_id, cond_id)
);
CREATE TABLE IF NOT EXISTS trials_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    source TEXT NOT NULL,
    n_trials INTEGER NOT NULL,
    note TEXT,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS holdout_access (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id TEXT,
    tree_id TEXT NOT NULL,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS portfolios (
    portfolio_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    settings_json TEXT NOT NULL,
    members_json TEXT NOT NULL,
    result_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS jobs (
    job_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    status TEXT NOT NULL,
    pid INTEGER,
    run_id TEXT,
    overrides_json TEXT,
    progress_json TEXT,
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
"""

DSR_TRIAL_SOURCES = ("ga", "manual")
_ADDED_COLUMNS = [
    ("trades", "move_1d", "REAL"),
    ("trades", "move_5d", "REAL"),
    ("run_trees", "strategy_type", "TEXT"),
    ("run_trees", "classification_json", "TEXT"),
]
SPLIT_ORDER = ("train", "validation", "holdout")


def now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def _clean(value: Any) -> Any:
    """JSON-safe copy (inf/nan -> None, numpy scalars -> python)."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if hasattr(value, "item") and not isinstance(value, (str, bytes)):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _nullable(df: pd.DataFrame, column: str) -> list[float | None]:
    if column not in df:
        return [None] * len(df)
    return [None if pd.isna(v) else float(v) for v in df[column]]


def dumps(value: Any) -> str:
    return json.dumps(_clean(value), sort_keys=True)


class Store:
    """Thin repository over one SQLite file."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)
            for table, column, kind in _ADDED_COLUMNS:
                existing = {r[1] for r in conn.execute(f"PRAGMA table_info({table})")}
                if column not in existing:
                    conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {kind}")

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        try:
            conn = sqlite3.connect(self.path, timeout=30)
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open database {self.path}: {exc}") from exc
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        try:
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise StorageError(f"database error on {self.path}: {exc}") from exc
        finally:
            conn.close()

    def query(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self.connect() as conn:
            return [dict(r) for r in conn.execute(sql, params).fetchall()]

    # runs ---------------------------------------------------------------

    def create_run(self, config: dict[str, Any], config_hash: str, seed: int, run_id: str | None = None) -> str:
        run_id = run_id or f"run_{datetime.now(UTC):%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO runs (run_id, created_at, status, config_json, config_hash, seed) VALUES (?, ?, ?, ?, ?, ?)",
                (run_id, now(), "running", dumps(config), config_hash, seed),
            )
        return run_id

    def update_run(self, run_id: str, **fields: Any) -> None:
        if not fields:
            return
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = [dumps(v) if k.endswith("_json") else v for k, v in fields.items()]
        with self.connect() as conn:
            conn.execute(f"UPDATE runs SET {cols} WHERE run_id = ?", (*values, run_id))

    def get_run(self, run_id: str) -> dict[str, Any]:
        rows = self.query("SELECT * FROM runs WHERE run_id = ?", (run_id,))
        if not rows:
            raise StorageError(f"unknown run {run_id!r}; list runs with get_rankings or the CLI")
        return rows[0]

    def latest_run(self, finished_only: bool = True) -> str | None:
        sql = "SELECT run_id FROM runs" + (" WHERE status = 'done'" if finished_only else "") + " ORDER BY created_at DESC, rowid DESC LIMIT 1"
        rows = self.query(sql)
        return rows[0]["run_id"] if rows else None

    def list_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.query(
            "SELECT run_id, created_at, status, config_hash, n_trials, holdout_done FROM runs ORDER BY created_at DESC, rowid DESC LIMIT ?",
            (limit,),
        )

    # trees --------------------------------------------------------------

    def save_tree(self, tree_id: str, tree_json: dict[str, Any], rules: str, depth: int, n_internal: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR IGNORE INTO trees (tree_id, tree_json, rules, depth, n_internal, created_at) VALUES (?, ?, ?, ?, ?, ?)",
                (tree_id, dumps(tree_json), rules, depth, n_internal, now()),
            )

    def get_tree(self, tree_id: str) -> dict[str, Any]:
        rows = self.query("SELECT * FROM trees WHERE tree_id = ?", (tree_id,))
        if not rows:
            raise StorageError(f"unknown tree {tree_id!r}")
        row = rows[0]
        row["tree_json"] = json.loads(row["tree_json"])
        return row

    def save_run_tree(self, run_id: str, tree_id: str, train_fitness: float, fitness: dict[str, Any], **flags: int) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO run_trees (run_id, tree_id, train_fitness, fitness_json) VALUES (?, ?, ?, ?) "
                "ON CONFLICT(run_id, tree_id) DO UPDATE SET train_fitness = excluded.train_fitness, fitness_json = excluded.fitness_json",
                (run_id, tree_id, train_fitness, dumps(fitness)),
            )
            for key, value in flags.items():
                conn.execute(f"UPDATE run_trees SET {key} = ? WHERE run_id = ? AND tree_id = ?", (value, run_id, tree_id))

    def set_flags(self, run_id: str, tree_id: str, **flags: int) -> None:
        with self.connect() as conn:
            for key, value in flags.items():
                conn.execute(f"UPDATE run_trees SET {key} = ? WHERE run_id = ? AND tree_id = ?", (value, run_id, tree_id))

    def run_of_tree(self, tree_id: str) -> str | None:
        rows = self.query(
            "SELECT rt.run_id FROM run_trees rt JOIN runs r USING (run_id) WHERE rt.tree_id = ? ORDER BY r.created_at DESC LIMIT 1",
            (tree_id,),
        )
        return rows[0]["run_id"] if rows else None

    # metrics, trades ----------------------------------------------------

    def save_backtest(self, run_id: str, tree_id: str, split: str, metrics: dict[str, Any],
                      per_pair: dict[str, dict[str, Any]], trades: pd.DataFrame, yearly: pd.DataFrame,
                      dsr: float | None = None) -> None:
        with self.connect() as conn:
            conn.execute("DELETE FROM metrics WHERE run_id = ? AND tree_id = ? AND split = ?", (run_id, tree_id, split))
            conn.execute("DELETE FROM trades WHERE run_id = ? AND tree_id = ? AND split = ?", (run_id, tree_id, split))
            conn.execute("DELETE FROM yearly WHERE run_id = ? AND tree_id = ? AND split = ?", (run_id, tree_id, split))
            rows = [(run_id, tree_id, split, "ALL", metrics.get("sharpe"), dsr, metrics.get("n_trades"), dumps(metrics))]
            rows += [(run_id, tree_id, split, p, m.get("sharpe"), None, m.get("n_trades"), dumps(m)) for p, m in per_pair.items()]
            conn.executemany("INSERT INTO metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
            if not trades.empty:
                t = trades
                conn.executemany(
                    "INSERT INTO trades VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    zip(
                        [run_id] * len(t), [tree_id] * len(t), [split] * len(t), t["pair"],
                        t["entry_time"].astype(str), t["exit_time"].astype(str), t["direction"].astype(int).tolist(),
                        t["entry_price"].tolist(), t["exit_price"].tolist(), t["bars_held"].astype(int).tolist(),
                        t["net_pips"].tolist(), t["cost_pips"].tolist(), t["r_multiple"].tolist(), t["exit_reason"],
                        t["branch"] if "branch" in t else [None] * len(t),
                        _nullable(t, "move_1d"), _nullable(t, "move_5d"),
                    ),
                )
            if not yearly.empty:
                conn.executemany(
                    "INSERT INTO yearly VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                    [(run_id, tree_id, split, int(r.year), int(r.n_trades), float(r.total_r), float(r.return_pct), float(r.win_rate))
                     for r in yearly.itertuples()],
                )

    def set_dsr(self, run_id: str, tree_id: str, split: str, dsr: float) -> None:
        with self.connect() as conn:
            conn.execute("UPDATE metrics SET dsr = ? WHERE run_id = ? AND tree_id = ? AND split = ? AND pair = 'ALL'",
                         (dsr, run_id, tree_id, split))

    def get_metrics(self, run_id: str, tree_id: str, split: str | None = None, pair: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT split, pair, sharpe, dsr, n_trades, metrics_json FROM metrics WHERE run_id = ? AND tree_id = ?"
        params: list[Any] = [run_id, tree_id]
        if split:
            sql += " AND split = ?"
            params.append(split)
        if pair:
            sql += " AND pair = ?"
            params.append(pair)
        rows = self.query(sql, tuple(params))
        for r in rows:
            r["metrics"] = json.loads(r.pop("metrics_json"))
        return rows

    def get_yearly(self, run_id: str, tree_id: str, split: str | None = None) -> list[dict[str, Any]]:
        sql = "SELECT split, year, n_trades, total_r, return_pct, win_rate FROM yearly WHERE run_id = ? AND tree_id = ?"
        params: list[Any] = [run_id, tree_id]
        if split:
            sql += " AND split = ?"
            params.append(split)
        return self.query(sql + " ORDER BY split, year", tuple(params))

    def get_trades(self, run_id: str, tree_id: str, split: str | None = None, pair: str | None = None,
                   limit: int | None = None, offset: int = 0) -> pd.DataFrame:
        sql = "SELECT * FROM trades WHERE run_id = ? AND tree_id = ?"
        params: list[Any] = [run_id, tree_id]
        if split:
            sql += " AND split = ?"
            params.append(split)
        if pair:
            sql += " AND pair = ?"
            params.append(pair)
        sql += " ORDER BY exit_time, pair"
        if limit is not None:
            sql += " LIMIT ? OFFSET ?"
            params += [limit, offset]
        with self.connect() as conn:
            df = pd.read_sql_query(sql, conn, params=params)
        for col in ("entry_time", "exit_time"):
            df[col] = pd.to_datetime(df[col], utc=True)
        return df

    def count_trades(self, run_id: str, tree_id: str, split: str | None = None, pair: str | None = None) -> int:
        sql = "SELECT COUNT(*) AS n FROM trades WHERE run_id = ? AND tree_id = ?"
        params: list[Any] = [run_id, tree_id]
        if split:
            sql += " AND split = ?"
            params.append(split)
        if pair:
            sql += " AND pair = ?"
            params.append(pair)
        return int(self.query(sql, tuple(params))[0]["n"])

    # robustness, conditions ---------------------------------------------

    def save_robustness(self, run_id: str, tree_id: str, report: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO robustness VALUES (?, ?, ?, ?, ?)",
                (run_id, tree_id, int(bool(report.get("survives"))), dumps(report), now()),
            )
            conn.execute("UPDATE run_trees SET survives = ? WHERE run_id = ? AND tree_id = ?",
                         (int(bool(report.get("survives"))), run_id, tree_id))

    def get_robustness(self, run_id: str, tree_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT report_json FROM robustness WHERE run_id = ? AND tree_id = ?", (run_id, tree_id))
        return json.loads(rows[0]["report_json"]) if rows else None

    def save_conditions(self, run_id: str, rows: list[dict[str, Any]]) -> None:
        with self.connect() as conn:
            conn.executemany(
                "INSERT OR REPLACE INTO run_conditions VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [(run_id, r["id"], r["family"], r["timeframe"], r["kind"], r["description"], r.get("activation"),
                  dumps(r["condition"])) for r in rows],
            )

    def get_conditions(self, run_id: str, family: str | None = None, limit: int = 100, offset: int = 0) -> tuple[list[dict[str, Any]], int]:
        where = "WHERE run_id = ?" + (" AND family = ?" if family else "")
        params: tuple[Any, ...] = (run_id, family) if family else (run_id,)
        total = int(self.query(f"SELECT COUNT(*) AS n FROM run_conditions {where}", params)[0]["n"])
        rows = self.query(
            f"SELECT cond_id, family, timeframe, kind, description, activation FROM run_conditions {where} "
            "ORDER BY family, cond_id LIMIT ? OFFSET ?",
            (*params, limit, offset),
        )
        return rows, total

    def condition_payloads(self, run_id: str) -> list[dict[str, Any]]:
        return [json.loads(r["condition_json"]) for r in self.query(
            "SELECT condition_json FROM run_conditions WHERE run_id = ?", (run_id,))]

    def save_classification(self, run_id: str, tree_id: str, classification: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE run_trees SET strategy_type = ?, classification_json = ? WHERE run_id = ? AND tree_id = ?",
                (classification["label"], dumps(classification), run_id, tree_id),
            )

    def get_classification(self, run_id: str, tree_id: str) -> dict[str, Any] | None:
        rows = self.query("SELECT classification_json FROM run_trees WHERE run_id = ? AND tree_id = ?", (run_id, tree_id))
        return json.loads(rows[0]["classification_json"]) if rows and rows[0]["classification_json"] else None

    def save_portfolio(self, portfolio_id: str, settings: dict[str, Any], members: list[dict[str, Any]],
                       result: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute("INSERT OR REPLACE INTO portfolios VALUES (?, ?, ?, ?, ?)",
                         (portfolio_id, now(), dumps(settings), dumps(members), dumps(result)))

    def get_portfolio(self, portfolio_id: str) -> dict[str, Any]:
        rows = self.query("SELECT * FROM portfolios WHERE portfolio_id = ?", (portfolio_id,))
        if not rows:
            raise StorageError(f"unknown portfolio {portfolio_id!r}")
        row = rows[0]
        return {"portfolio_id": portfolio_id, "created_at": row["created_at"], "settings": json.loads(row["settings_json"]),
                "members": json.loads(row["members_json"]), "result": json.loads(row["result_json"])}

    def list_portfolios(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.query("SELECT portfolio_id, created_at FROM portfolios ORDER BY created_at DESC, rowid DESC LIMIT ?", (limit,))

    # trial ledger and holdout --------------------------------------------

    def log_trials(self, run_id: str | None, source: str, n_trials: int, note: str = "") -> None:
        with self.connect() as conn:
            conn.execute("INSERT INTO trials_log (run_id, source, n_trials, note, created_at) VALUES (?, ?, ?, ?, ?)",
                         (run_id, source, n_trials, note, now()))

    def selection_trials(self) -> int:
        """Trials that count for the deflated Sharpe, excluding holdout accesses."""
        placeholders = ",".join("?" for _ in DSR_TRIAL_SOURCES)
        rows = self.query(f"SELECT COALESCE(SUM(n_trials), 0) AS n FROM trials_log WHERE source IN ({placeholders})",
                          DSR_TRIAL_SOURCES)
        return int(rows[0]["n"])

    def holdout_stage_done(self, run_id: str) -> bool:
        return bool(self.get_run(run_id)["holdout_done"])

    def record_holdout_access(self, run_id: str | None, tree_hash: str, source: str) -> None:
        with self.connect() as conn:
            conn.execute("INSERT INTO holdout_access (run_id, tree_id, source, created_at) VALUES (?, ?, ?, ?)",
                         (run_id, tree_hash, source, now()))

    def holdout_trial_count(self) -> int:
        return int(self.query("SELECT COUNT(*) AS n FROM holdout_access")[0]["n"])

    def total_trials(self) -> int:
        return self.selection_trials() + self.holdout_trial_count()

    # rankings -------------------------------------------------------------

    def ranking_rows(self, run_id: str) -> list[dict[str, Any]]:
        """One row per candidate tree with its key metrics on every split."""
        trees = self.query(
            "SELECT rt.tree_id, rt.train_fitness, rt.is_finalist, rt.survives, rt.strategy_type, t.depth, t.n_internal, t.rules "
            "FROM run_trees rt JOIN trees t USING (tree_id) WHERE rt.run_id = ? AND rt.is_candidate = 1",
            (run_id,),
        )
        metrics = self.query("SELECT tree_id, split, sharpe, dsr, metrics_json FROM metrics WHERE run_id = ? AND pair = 'ALL'", (run_id,))
        by_tree: dict[str, dict[str, Any]] = {}
        for m in metrics:
            data = json.loads(m["metrics_json"])
            by_tree.setdefault(m["tree_id"], {}).update({
                f"{m['split']}_sharpe": m["sharpe"], f"{m['split']}_dsr": m["dsr"],
                f"{m['split']}_n_trades": data.get("n_trades"), f"{m['split']}_profit_factor": data.get("profit_factor"),
                f"{m['split']}_max_dd_pct": data.get("max_dd_pct"),
            })
        robust = {r["tree_id"]: json.loads(r["report_json"]) for r in self.query(
            "SELECT tree_id, report_json FROM robustness WHERE run_id = ?", (run_id,))}
        rows = []
        for t in trees:
            rep = robust.get(t["tree_id"], {})
            rows.append({
                "tree_id": t["tree_id"], "survives": None if t["survives"] is None else bool(t["survives"]),
                "finalist": bool(t["is_finalist"]), "strategy_type": t["strategy_type"],
                "depth": t["depth"], "n_internal": t["n_internal"],
                "train_fitness": t["train_fitness"], **by_tree.get(t["tree_id"], {}),
                "wf_positive_share": rep.get("walkforward", {}).get("positive_share"),
                "sensitivity_worst_ratio": rep.get("sensitivity", {}).get("worst_ratio"),
                "mc_max_dd_pct_95": rep.get("montecarlo", {}).get("max_dd_pct_at_conf"),
                "mc_max_risk_pct": rep.get("montecarlo", {}).get("max_risk_pct_for_limit"),
                "rules": t["rules"],
            })
        return rows

    # jobs -------------------------------------------------------------------

    def create_job(self, kind: str, overrides: dict[str, Any]) -> str:
        job_id = f"job_{uuid.uuid4().hex[:10]}"
        with self.connect() as conn:
            conn.execute(
                "INSERT INTO jobs (job_id, kind, status, overrides_json, created_at, updated_at) VALUES (?, ?, ?, ?, ?, ?)",
                (job_id, kind, "queued", dumps(overrides), now(), now()),
            )
        return job_id

    def update_job(self, job_id: str, **fields: Any) -> None:
        fields["updated_at"] = now()
        cols = ", ".join(f"{k} = ?" for k in fields)
        values = [dumps(v) if k.endswith("_json") else v for k, v in fields.items()]
        with self.connect() as conn:
            conn.execute(f"UPDATE jobs SET {cols} WHERE job_id = ?", (*values, job_id))

    def get_job(self, job_id: str) -> dict[str, Any]:
        rows = self.query("SELECT * FROM jobs WHERE job_id = ?", (job_id,))
        if not rows:
            raise StorageError(f"unknown job {job_id!r}")
        job = rows[0]
        for key in ("overrides_json", "progress_json"):
            job[key.removesuffix("_json")] = json.loads(job.pop(key)) if job.get(key) else None
        return job
