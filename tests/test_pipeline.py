from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fxgen.config import Config
from fxgen.errors import HoldoutLockedError, StorageError
from fxgen.pipeline import generate_report, manual_backtest, pine_source, rerun_robustness, run_holdout, run_pipeline
from fxgen.storage.db import Store
from fxgen.tree.serialize import from_dict

from conftest import make_config


@pytest.fixture(scope="module")
def finished(synthetic_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> tuple[Config, Store, str]:
    db = tmp_path_factory.mktemp("pipe") / "fx.db"
    cfg = make_config(synthetic_dir, db)
    store = Store(db)
    run_id = run_pipeline(cfg, store)
    return cfg, store, run_id


def test_run_is_stored(finished: tuple[Config, Store, str]) -> None:
    cfg, store, run_id = finished
    run = store.get_run(run_id)
    assert run["status"] == "done" and run["n_trials"] > 0 and not run["holdout_done"]
    rows = store.ranking_rows(run_id)
    assert rows and all(r["survives"] is not None for r in rows)
    assert not store.query("SELECT * FROM metrics WHERE run_id = ? AND split = 'holdout'", (run_id,))
    assert store.selection_trials() == run["n_trials"]
    assert (cfg.resolve(cfg.storage.output_dir) / run_id / "ranking.csv").exists()
    conds, total = store.get_conditions(run_id, limit=5)
    assert total > 50 and len(conds) == 5
    tid = rows[0]["tree_id"]
    trades = store.get_trades(run_id, tid, "train")
    assert len(trades) == store.count_trades(run_id, tid, "train") > 0
    assert isinstance(trades["exit_time"].dtype, pd.DatetimeTZDtype)


def test_holdout_requires_finalists_and_runs_once(finished: tuple[Config, Store, str]) -> None:
    cfg, store, run_id = finished
    has_finalists = bool(store.query("SELECT 1 FROM run_trees WHERE run_id = ? AND is_finalist = 1", (run_id,)))
    if not has_finalists:
        with pytest.raises(HoldoutLockedError):
            run_holdout(store, run_id)
        store.set_flags(run_id, store.ranking_rows(run_id)[0]["tree_id"], is_finalist=1)
    before = store.holdout_trial_count()
    summary = run_holdout(store, run_id)
    assert store.holdout_trial_count() == before + len(summary)
    tid, entry = next(iter(summary.items()))
    assert entry["dsr"]["n_trials"] == store.total_trials()
    assert (cfg.resolve(cfg.storage.output_dir) / run_id / "reports" / f"{tid}.html").exists()
    with pytest.raises(HoldoutLockedError):
        run_holdout(store, run_id)


def test_manual_backtest_counts_trials(finished: tuple[Config, Store, str]) -> None:
    _, store, run_id = finished
    tree = from_dict({"if": "D1.dist_ema50>0", "then": {"leaf": "LONG"}, "else": {"leaf": "SHORT"}})
    n0, h0 = store.total_trials(), store.holdout_trial_count()
    out = manual_backtest(store, tree, "validation", ["EURUSD", "GBPUSD"], run_id=run_id)
    assert set(out["per_pair"]) == {"EURUSD", "GBPUSD"}
    assert store.total_trials() == n0 + 1 and store.holdout_trial_count() == h0
    out = manual_backtest(store, tree, "holdout", run_id=run_id)
    assert store.holdout_trial_count() == h0 + 1 and out["dsr"]["n_trials"] == n0 + 2
    assert store.get_metrics(run_id, out["tree_id"], "manual_holdout", "ALL")
    h4 = manual_backtest(store, from_dict({"if": "D1.rsi14>p50", "then": {"leaf": "LONG"}, "else": {"leaf": "NO_TRADE"}}),
                         "train", timeframe="H4", run_id=run_id)
    assert h4["metrics"]["n_trades"] > 0


def test_robustness_report_and_pine(finished: tuple[Config, Store, str]) -> None:
    _, store, run_id = finished
    tid = store.ranking_rows(run_id)[0]["tree_id"]
    report = rerun_robustness(store, tid, run_id)
    assert "walkforward" in report and "permutation" in report["checks"]
    assert pine_source(store, tid, run_id).startswith("//@version=6")
    assert generate_report(store, tid, run_id).exists()


def test_store_errors(tmp_path: Path) -> None:
    store = Store(tmp_path / "x.db")
    with pytest.raises(StorageError):
        store.get_run("nope")
    with pytest.raises(StorageError):
        store.get_tree("abcd")
    assert store.latest_run() is None and store.total_trials() == 0
