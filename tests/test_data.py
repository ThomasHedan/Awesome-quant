from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxgen.config import Config, DataConfig
from fxgen.data.loader import discover_datasets, load_csv, load_pair
from fxgen.data.mtf import align_to_execution
from fxgen.data.splits import split_period, subperiods, walkforward_windows
from fxgen.data.resample import resample
from fxgen.data.synthetic import generate_universe
from fxgen.data.validation import count_gaps
from fxgen.errors import DataError, HoldoutLockedError

from conftest import hourly_frame


def _data_cfg(**kwargs: object) -> DataConfig:
    return DataConfig.model_validate({"pairs": ["EURUSD"], **kwargs})


def test_load_csv_custom_columns_and_timezone(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    pd.DataFrame(
        {
            "Date": ["2020-01-06 10:00", "2020-01-06 11:00", "2020-01-06 11:00", "2020-01-06 12:00"],
            "O": [1.0, 1.1, 1.1, 1.2], "H": [1.2, 1.2, 1.2, 1.3], "L": [0.9, 1.0, 1.0, 1.1], "C": [1.1, 1.15, 1.16, 1.25],
        }
    ).to_csv(path, index=False)
    cfg = _data_cfg(
        columns={"time": "Date", "open": "O", "high": "H", "low": "L", "close": "C", "volume": None},
        source_timezone="Europe/Paris",
    )
    df, quality = load_csv(path, "EURUSD", "H1", cfg)
    assert str(df.index.tz) == "UTC"
    assert df.index[0] == pd.Timestamp("2020-01-06 09:00", tz="UTC")
    assert quality.duplicates_removed == 1
    assert df.loc["2020-01-06 10:00", "close"].item() == 1.16
    assert (df["volume"] == 0).all()
    assert list(df.dtypes.unique()) == [np.float64]


def test_load_csv_close_label_shifts_to_open_time(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    pd.DataFrame({"time": ["2020-01-06 04:00"], "open": [1], "high": [1], "low": [1], "close": [1], "volume": [1]}).to_csv(path, index=False)
    df, _ = load_csv(path, "EURUSD", "H4", _data_cfg(timestamp_label="close"))
    assert df.index[0] == pd.Timestamp("2020-01-06 00:00", tz="UTC")


def test_load_csv_repairs_ohlc_and_drops_nan(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    pd.DataFrame(
        {"time": ["2020-01-06 00:00", "2020-01-06 01:00"], "open": [1.0, None], "high": [0.95, 1.0],
         "low": [0.9, 1.0], "close": [1.05, 1.0], "volume": [1, 1]}
    ).to_csv(path, index=False)
    df, quality = load_csv(path, "EURUSD", "H1", _data_cfg())
    assert quality.nan_rows_removed == 1 and quality.ohlc_repaired == 1
    assert df["high"].iloc[0] == 1.05


def test_load_csv_errors(tmp_path: Path) -> None:
    with pytest.raises(DataError, match="not found"):
        load_csv(tmp_path / "missing.csv", "EURUSD", "H1", _data_cfg())
    path = tmp_path / "bad.csv"
    pd.DataFrame({"time": ["2020-01-06"], "open": [1]}).to_csv(path, index=False)
    with pytest.raises(DataError, match="missing columns"):
        load_csv(path, "EURUSD", "H1", _data_cfg())


def test_gaps_ignore_weekend() -> None:
    idx = pd.DatetimeIndex(
        ["2020-01-10 20:00", "2020-01-10 21:00", "2020-01-12 22:00", "2020-01-12 23:00", "2020-01-13 03:00"], tz="UTC"
    )
    n, max_gap = count_gaps(idx, "H1")
    assert n == 1 and max_gap == 4.0


def test_load_pair_and_discover(cfg: Config) -> None:
    pair = load_pair(cfg, "EURUSD")
    assert set(pair.frames) == {"H1", "H4", "D1"}
    assert all(q.n_gaps == 0 for q in pair.quality.values())
    datasets = discover_datasets(cfg)
    assert len(datasets) == 12 * 3
    assert {d["timeframe"] for d in datasets} == {"H1", "H4", "D1"}


def test_mtf_alignment_uses_only_closed_bars() -> None:
    h1 = hourly_frame(48)
    h4 = resample(h1, "H4")
    aligned = align_to_execution(h4[["close"]], "H4", h1.index, "H1")
    for t in h1.index:
        decision_time = t + pd.Timedelta(hours=1)
        closed = h4[h4.index + pd.Timedelta(hours=4) <= decision_time]
        expected = closed["close"].iloc[-1] if len(closed) else np.nan
        got = aligned.loc[t, "close"]
        assert (np.isnan(expected) and np.isnan(got)) or got == expected


def test_mtf_h4_bar_becomes_visible_on_its_last_h1_bar() -> None:
    h1 = hourly_frame(48, start="2020-01-06 02:00")
    h4 = resample(h1, "H4")
    aligned = align_to_execution(h4[["close"]], "H4", h1.index, "H1")
    first_h4 = h4.index[0]
    last_h1_of_first_h4 = first_h4 + pd.Timedelta(hours=3)
    assert np.isnan(aligned.loc[last_h1_of_first_h4 - pd.Timedelta(hours=1), "close"])
    assert aligned.loc[last_h1_of_first_h4, "close"] == h1.loc[last_h1_of_first_h4, "close"]


def test_mtf_alignment_is_causal_under_future_corruption() -> None:
    h1 = hourly_frame(200)
    cut = h1.index[120]
    corrupted = h1.copy()
    corrupted.loc[corrupted.index > cut, ["open", "high", "low", "close"]] *= 3.0
    for tf in ("H4", "D1"):
        clean = align_to_execution(resample(h1, tf), tf, h1.index, "H1")
        dirty = align_to_execution(resample(corrupted, tf), tf, h1.index, "H1")
        pd.testing.assert_frame_equal(clean.loc[:cut], dirty.loc[:cut])


def test_splits_and_holdout_lock(cfg: Config) -> None:
    train = split_period(cfg.splits, "train")
    val = split_period(cfg.splits, "validation")
    assert train.end == val.start
    with pytest.raises(HoldoutLockedError):
        split_period(cfg.splits, "holdout")
    hold = split_period(cfg.splits, "holdout", allow_holdout=True)
    assert hold.start == val.end


def test_walkforward_windows_never_touch_holdout(cfg: Config) -> None:
    windows = walkforward_windows(cfg.splits, cfg.validation.walkforward)
    holdout_start = split_period(cfg.splits, "holdout", allow_holdout=True).start
    assert len(windows) == 2
    for is_p, oos_p in windows:
        assert is_p.end == oos_p.start <= holdout_start
    assert windows[-1][1].end == holdout_start
    assert len(subperiods(windows[0][0], 3)) == 3


def test_synthetic_is_deterministic_and_valid() -> None:
    a = generate_universe(start="2020-01-05", end="2020-03-01", pairs={"EURUSD": 1.1}, seed=1)
    b = generate_universe(start="2020-01-05", end="2020-03-01", pairs={"EURUSD": 1.1}, seed=1)
    pd.testing.assert_frame_equal(a["EURUSD"]["H1"], b["EURUSD"]["H1"])
    h1 = a["EURUSD"]["H1"]
    assert (h1["high"] >= h1[["open", "close"]].max(axis=1)).all()
    assert (h1["low"] <= h1[["open", "close"]].min(axis=1)).all()
    assert not (h1.index.day_of_week == 5).any()
    d1 = a["EURUSD"]["D1"]
    assert (d1.index.hour == 22).all()
