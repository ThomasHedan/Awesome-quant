"""Background jobs: a detached worker process runs the pipeline and reports to SQLite.

The MCP server only launches, polls and cancels; it never blocks on a run.
Cancellation is cooperative: the worker checks the job status between GA
generations and between pipeline stages.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

from fxgen.config import load_config
from fxgen.logging_setup import setup_logging
from fxgen.pipeline import Cancelled, run_pipeline
from fxgen.storage.db import Store

logger = logging.getLogger(__name__)

ACTIVE = ("queued", "running", "cancel_requested")


def launch_evolution(store: Store, config_path: Path, overrides: dict[str, Any], log_dir: Path) -> str:
    """Validate the config, create the job row and start the worker. Returns the job id."""
    load_config(config_path, overrides)
    job_id = store.create_job("evolution", overrides)
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = (log_dir / f"{job_id}.log").open("w")
    proc = subprocess.Popen(
        [sys.executable, "-m", "fxgen.jobs", "--db", str(store.path), "--job", job_id, "--config", str(config_path)],
        stdout=subprocess.DEVNULL,
        stderr=log_file,
        start_new_session=True,
    )
    store.update_job(job_id, pid=proc.pid, status="running")
    return job_id


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    try:
        os.waitpid(pid, os.WNOHANG)
    except ChildProcessError:
        pass
    try:
        with open(f"/proc/{pid}/stat") as fh:
            return fh.read().split()[2] != "Z"
    except OSError:
        return True


def job_status(store: Store, job_id: str) -> dict[str, Any]:
    """Job row, marking it failed if its worker died without reporting."""
    job = store.get_job(job_id)
    if job["status"] in ACTIVE and job["pid"] and not _alive(job["pid"]):
        store.update_job(job_id, status="failed", error="worker process exited unexpectedly; see its log file")
        job = store.get_job(job_id)
    return job


def cancel(store: Store, job_id: str) -> dict[str, Any]:
    job = job_status(store, job_id)
    if job["status"] in ("queued", "running"):
        store.update_job(job_id, status="cancel_requested")
    return store.get_job(job_id)


def work(db: Path, job_id: str, config_path: Path) -> None:
    store = Store(db)
    job = store.get_job(job_id)
    overrides = job["overrides"] or {}
    cfg = load_config(config_path, overrides)

    def progress(stage: str, info: dict[str, Any]) -> None:
        fields: dict[str, Any] = {"progress_json": {"stage": stage, **info}}
        if "run_id" in info:
            fields["run_id"] = info["run_id"]
        store.update_job(job_id, **fields)

    def should_stop() -> bool:
        return store.get_job(job_id)["status"] == "cancel_requested"

    try:
        run_id = run_pipeline(cfg, store, progress=progress, should_stop=should_stop)
        store.update_job(job_id, status="done", run_id=run_id)
    except Cancelled as exc:
        store.update_job(job_id, status="cancelled", error=str(exc))
    except Exception as exc:
        logger.exception("job %s failed", job_id)
        store.update_job(job_id, status="failed", error=f"{type(exc).__name__}: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="fxgen background worker")
    parser.add_argument("--db", required=True, type=Path)
    parser.add_argument("--job", required=True)
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()
    setup_logging()
    work(args.db, args.job, args.config)


if __name__ == "__main__":
    main()
