from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from fxgen.config import Config, load_config
from fxgen.data.synthetic import write_universe
from fxgen.workspace import Workspace, build_workspace

SMALL_PAIRS = ["EURUSD", "GBPUSD", "USDJPY", "AUDUSD"]


def make_config(data_dir: Path, db_path: Path, **overrides: object) -> Config:
    base: dict[str, object] = {
        "data": {"directory": str(data_dir), "pairs": SMALL_PAIRS},
        "splits": {
            "train_start": "2016-01-01",
            "validation_start": "2018-07-01",
            "holdout_start": "2019-07-01",
            "end": "2019-12-31",
        },
        "ga": {"population_size": 30, "generations": 3, "n_jobs": 1, "elite_size": 2},
        "fitness": {"min_trades_total": 20, "min_trades_per_pair": 2, "n_subperiods": 2, "min_positive_subperiods": 1},
        "validation": {
            "walkforward": {"n_windows": 2, "train_years": 1, "test_years": 0.5, "generations": 2, "population_size": 20},
            "permutation": {"n_runs": 2, "generations": 2, "population_size": 20},
            "montecarlo": {"n_sims": 100},
            "n_finalists": 3,
        },
        "storage": {"database": str(db_path), "output_dir": str(db_path.parent / "out")},
    }
    for key, value in overrides.items():
        base[key] = value
    return load_config(overrides=base)


@pytest.fixture(scope="session")
def synthetic_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    directory = tmp_path_factory.mktemp("synthetic")
    write_universe(directory, start="2016-01-03", end="2020-01-01", edge=0.03, seed=7)
    return directory


@pytest.fixture
def cfg(synthetic_dir: Path, tmp_path: Path) -> Config:
    return make_config(synthetic_dir, tmp_path / "fxgen.db")


@pytest.fixture(scope="session")
def session_cfg(synthetic_dir: Path, tmp_path_factory: pytest.TempPathFactory) -> Config:
    return make_config(synthetic_dir, tmp_path_factory.mktemp("db") / "fxgen.db")


@pytest.fixture(scope="session")
def ws(session_cfg: Config) -> Workspace:
    return build_workspace(session_cfg)


def hourly_frame(n: int = 200, start: str = "2020-01-06 00:00") -> pd.DataFrame:
    index = pd.date_range(start, periods=n, freq="1h", tz="UTC", name="time")
    close = pd.Series(range(n), index=index, dtype=float) * 0.0001 + 1.1
    return pd.DataFrame(
        {"open": close.shift(1).fillna(1.1), "high": close + 0.0005, "low": close - 0.0005, "close": close, "volume": 100.0}
    )
