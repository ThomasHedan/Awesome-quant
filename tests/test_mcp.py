from __future__ import annotations

import asyncio
import time
from pathlib import Path
from typing import Any

import pytest
import yaml
from mcp.server.fastmcp.exceptions import ToolError

from fxgen.mcp_server import server
from fxgen.pipeline import run_pipeline
from fxgen.storage.db import Store

from conftest import make_config


def call(name: str, args: dict[str, Any] | None = None) -> Any:
    result = asyncio.run(server.mcp.call_tool(name, args or {}))
    return result[1] if isinstance(result, tuple) else result


@pytest.fixture(scope="module")
def env(synthetic_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    root = tmp_path_factory.mktemp("mcp")
    cfg = make_config(synthetic_dir, root / "fx.db")
    config_file = root / "config.yaml"
    config_file.write_text(yaml.safe_dump(cfg.model_dump(mode="json")))
    store = Store(root / "fx.db")
    run_id = run_pipeline(cfg, store)
    mp = pytest.MonkeyPatch()
    mp.setenv("FXGEN_CONFIG", str(config_file))
    tree_id = store.ranking_rows(run_id)[0]["tree_id"]
    yield {"store": store, "run_id": run_id, "tree_id": tree_id}
    mp.undo()


def test_tools_are_registered() -> None:
    names = {t.name for t in asyncio.run(server.mcp.list_tools())}
    assert {"list_datasets", "list_conditions", "get_rankings", "get_tree", "get_metrics", "get_trades",
            "get_robustness", "compare_trees", "run_evolution", "get_job_status", "cancel_job", "backtest_tree",
            "run_robustness", "export_pine", "generate_report", "classify_strategy", "build_portfolio",
            "get_portfolio"} <= names


def test_read_tools(env: dict[str, Any]) -> None:
    tid = env["tree_id"]
    assert len(call("list_datasets")["files"]) == 36
    conds = call("list_conditions", {"family": "momentum", "limit": 5})
    assert len(conds["conditions"]) == 5 and all(c["family"] == "momentum" for c in conds["conditions"])
    ranking = call("get_rankings", {"stage": "validation", "top_n": 3, "sort_by": "sharpe"})
    assert len(ranking["rows"]) <= 3 and ranking["sorted_by"] == "validation_sharpe"
    assert call("get_rankings", {"stage": "holdout"})["note"]
    tree = call("get_tree", {"tree_id": tid})
    assert tree["rules"].startswith("IF") or tree["rules"].startswith("->")
    metrics = call("get_metrics", {"tree_id": tid, "split": "train"})
    assert set(metrics["splits"]) == {"train"} and metrics["splits"]["train"]["portfolio"]["n_trades"] > 0
    trades = call("get_trades", {"tree_id": tid, "limit": 7})
    assert len(trades["trades"]) == 7 and trades["total"] > 7
    robust = call("get_robustness", {"tree_id": tid})
    assert "checks" in robust and robust["run_walkforward_process"] is not None
    other = env["store"].ranking_rows(env["run_id"])[1]["tree_id"]
    cmp = call("compare_trees", {"tree_ids": [tid, other]})
    assert cmp["daily_r_correlation"][tid][tid] == pytest.approx(1.0)


def test_action_tools(env: dict[str, Any]) -> None:
    tid = env["tree_id"]
    assert call("export_pine", {"tree_id": tid})["pine"].startswith("//@version=6")
    assert Path(call("generate_report", {"tree_id": tid})["path"]).exists()
    assert "checks" in call("run_robustness", {"tree_id": tid})
    tree = {"if": "D1.dist_ema200>0", "then": {"leaf": "LONG"}, "else": {"leaf": "SHORT"}}
    out = call("backtest_tree", {"tree_json": tree, "pairs": ["EURUSD"], "split": "validation"})
    assert "warning" not in out and set(out["per_pair"]) == {"EURUSD"}
    held = call("backtest_tree", {"tree_json": tree, "split": "holdout"})
    assert "holdout accessed and logged" in held["warning"]


def test_classify_and_portfolio_tools(env: dict[str, Any]) -> None:
    c = call("classify_strategy", {"tree_id": env["tree_id"]})
    assert c["label"] and "structure" in c
    pf = call("build_portfolio", {"max_strategies": 2, "survivors_only": False, "max_correlation": 0.9})
    assert 1 <= len(pf["members"]) <= 2 and Path(pf["report"]).exists()
    stored = call("get_portfolio", {"portfolio_id": pf["portfolio_id"]})
    assert stored["members"][0]["tree_id"] == pf["members"][0]["tree_id"]
    with pytest.raises(ToolError):
        call("build_portfolio", {"max_correlation": 2})


def test_validation_errors(env: dict[str, Any]) -> None:
    with pytest.raises(ToolError):
        call("get_rankings", {"stage": "test"})
    with pytest.raises(ToolError):
        call("get_rankings", {"top_n": 0})
    with pytest.raises(ToolError, match="TreeError"):
        call("backtest_tree", {"tree_json": {"if": "H4.unknown>p50", "then": {"leaf": "LONG"}, "else": {"leaf": "SHORT"}}})
    with pytest.raises(ToolError, match="invalid leaf"):
        call("backtest_tree", {"tree_json": {"leaf": "BUY"}})
    with pytest.raises(ToolError, match="not allowed"):
        call("run_evolution", {"config_overrides": {"splits": {"holdout_start": "2030-01-01"}}})
    with pytest.raises(ToolError, match="invalid config_overrides"):
        call("run_evolution", {"config_overrides": {"ga": {"population_size": 1}}})
    with pytest.raises(ToolError, match="unknown tree"):
        call("get_tree", {"tree_id": "deadbeef"})


def test_background_job_lifecycle(env: dict[str, Any]) -> None:
    job = call("run_evolution", {"config_overrides": {"ga": {"generations": 2, "population_size": 20},
                                                      "validation": {"permutation": {"n_runs": 0}, "walkforward": {"n_windows": 0}}}})
    job_id = job["job_id"]
    deadline = time.time() + 300
    status = call("get_job_status", {"job_id": job_id})
    while status["status"] in ("running", "queued") and time.time() < deadline:
        time.sleep(2)
        status = call("get_job_status", {"job_id": job_id})
    assert status["status"] == "done", status
    assert status["run_id"] and env["store"].get_run(status["run_id"])["status"] == "done"

    job2 = call("run_evolution", {"config_overrides": {"ga": {"generations": 50, "population_size": 20}}})
    assert call("cancel_job", {"job_id": job2["job_id"]})["status"] == "cancel_requested"
    deadline = time.time() + 300
    status = call("get_job_status", {"job_id": job2["job_id"]})
    while status["status"] == "cancel_requested" and time.time() < deadline:
        time.sleep(2)
        status = call("get_job_status", {"job_id": job2["job_id"]})
    assert status["status"] == "cancelled", status
