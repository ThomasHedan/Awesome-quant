from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxgen.conditions.builder import generate_candidates, parse_condition_id
from fxgen.conditions.redundancy import correlation_matrix, filter_conditions
from fxgen.conditions.thresholds import fit_thresholds
from fxgen.data.loader import PairData
from fxgen.data.splits import Period
from fxgen.data.synthetic import generate_universe
from fxgen.errors import TreeError
from fxgen.indicators import core
from fxgen.indicators.base import FAMILIES, all_features
from fxgen.market import build_pair_frame, condition_column

FEATURES = all_features()


@pytest.fixture(scope="module")
def universe() -> dict[str, dict[str, pd.DataFrame]]:
    return generate_universe(start="2019-01-06", end="2020-07-01", pairs={"EURUSD": 1.1, "USDJPY": 110.0}, seed=3)


@pytest.fixture(scope="module")
def frames(universe: dict[str, dict[str, pd.DataFrame]]):  # type: ignore[no-untyped-def]
    return {p: build_pair_frame(PairData(p, f), FEATURES, "H1", ["H4", "D1"]) for p, f in universe.items()}


TRAIN = Period(pd.Timestamp("2019-01-01", tz="UTC"), pd.Timestamp("2020-01-01", tz="UTC"))


def test_library_size_and_families() -> None:
    assert len(FEATURES) * 3 >= 100
    assert {f.family for f in FEATURES} <= set(FAMILIES)
    assert all(f.pine for f in FEATURES)


def test_rma_matches_pine_seed() -> None:
    x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    out = core.rma(x, 3)
    assert np.isnan(out.iloc[1])
    assert out.iloc[2] == pytest.approx(2.0)
    assert out.iloc[3] == pytest.approx(2.0 + (4.0 - 2.0) / 3)


def test_percentrank_definition() -> None:
    x = pd.Series([1.0, 3.0, 2.0, 5.0, 4.0])
    out = core.percentrank(x, 3)
    assert out.iloc[3] == pytest.approx(100.0)
    assert out.iloc[4] == pytest.approx(100 * 2 / 3)


def test_linreg_slope_matches_polyfit() -> None:
    x = pd.Series(np.random.default_rng(0).standard_normal(30).cumsum())
    out = core.linreg_slope(x, 10)
    assert out.iloc[-1] == pytest.approx(np.polyfit(np.arange(10), x.iloc[-10:], 1)[0])


def test_bars_since_highest() -> None:
    x = pd.Series([1.0, 5.0, 2.0, 3.0, 1.0])
    assert core.bars_since_highest(x, 4).iloc[-1] == 3
    assert core.bars_since_lowest(x, 3).iloc[-1] == 0


def test_rsi_bounds() -> None:
    x = pd.Series(np.random.default_rng(1).standard_normal(500).cumsum() + 100)
    r = core.rsi(x, 14).dropna()
    assert ((r >= 0) & (r <= 100)).all()


def test_no_lookahead_in_any_feature_or_state(universe: dict[str, dict[str, pd.DataFrame]]) -> None:
    """Corrupting every bar after ``cut`` (on all timeframes) must not change anything up to ``cut``."""
    frames = universe["EURUSD"]
    cut = frames["H1"].index[6000]
    clean = build_pair_frame(PairData("EURUSD", frames), FEATURES, "H1", ["H4", "D1"])
    corrupted: dict[str, pd.DataFrame] = {}
    for tf, df in frames.items():
        bad = df.copy()
        bar_close = bad.index + {"H1": pd.Timedelta(hours=1), "H4": pd.Timedelta(hours=4), "D1": pd.Timedelta(days=1)}[tf]
        future = bar_close > cut + pd.Timedelta(hours=1)
        rng = np.random.default_rng(9)
        noise = rng.uniform(0.5, 1.5, size=(int(future.sum()), 1))
        bad.loc[future, ["open", "high", "low", "close"]] = bad.loc[future, ["open", "high", "low", "close"]].to_numpy() * noise
        bad.loc[future, "volume"] = bad.loc[future, "volume"] * 7
        corrupted[tf] = bad
    dirty = build_pair_frame(PairData("EURUSD", corrupted), FEATURES, "H1", ["H4", "D1"])
    pd.testing.assert_frame_equal(clean.features.loc[:cut], dirty.features.loc[:cut])
    pd.testing.assert_frame_equal(clean.states.loc[:cut], dirty.states.loc[:cut])
    assert not clean.features.loc[cut + pd.Timedelta(days=3):].equals(dirty.features.loc[cut + pd.Timedelta(days=3):])


def test_candidate_ids_parse_back() -> None:
    candidates = generate_candidates(FEATURES, ["H1", "H4", "D1"], [25, 50, 75])
    assert len({c.id for c in candidates}) == len(candidates)
    for cond in candidates:
        parsed = parse_condition_id(cond.id, FEATURES)
        assert (parsed.kind, parsed.feature, parsed.timeframe, parsed.percentile) == (
            cond.kind, cond.feature, cond.timeframe, cond.percentile
        )
    assert parse_condition_id("H4.rsi14>70", FEATURES).threshold == 70.0
    with pytest.raises(TreeError):
        parse_condition_id("H4.nope>p50", FEATURES)
    with pytest.raises(TreeError):
        parse_condition_id("garbage", FEATURES)


def test_thresholds_use_train_only(frames, universe) -> None:  # type: ignore[no-untyped-def]
    cands = [parse_condition_id("H4.rsi14>p75", FEATURES), parse_condition_id("D1.dist_ema50>p25", FEATURES)]
    fitted = fit_thresholds(frames, cands, TRAIN)
    shifted = {}
    for pair, fr in frames.items():
        f = fr.features.copy()
        f.loc[f.index >= TRAIN.end] += 1000.0
        shifted[pair] = type(fr)(fr.pair, fr.execution_tf, fr.ohlc, fr.atr, f, fr.states)
    refit = fit_thresholds(shifted, cands, TRAIN)
    assert [c.threshold for c in fitted] == [c.threshold for c in refit]
    values = np.concatenate([fr.features["H4.rsi14"][TRAIN.mask(fr.features.index)].dropna() for fr in frames.values()])
    assert fitted[0].threshold == pytest.approx(np.percentile(values, 75))


def test_condition_column_matches_threshold(frames) -> None:  # type: ignore[no-untyped-def]
    cond = fit_thresholds(frames, [parse_condition_id("H1.rsi14>p50", FEATURES)], TRAIN)[0]
    col = condition_column(frames["EURUSD"], cond)
    values = frames["EURUSD"].features["H1.rsi14"].to_numpy()
    assert (col == (np.nan_to_num(values, nan=-np.inf) > cond.threshold)).all()
    assert 0.4 < col[TRAIN.mask(frames["EURUSD"].features.index).to_numpy()].mean() < 0.6


def test_correlation_matrix_phi() -> None:
    a = np.array([1, 0, 1, 0, 1, 1], dtype=bool)
    X = np.column_stack([a, a, ~a, np.ones(6, dtype=bool)])
    c = correlation_matrix(X)
    assert c[0, 1] == pytest.approx(1.0) and c[0, 2] == pytest.approx(-1.0) and c[0, 3] == 0


def test_redundancy_removes_duplicates_and_inactive(frames) -> None:  # type: ignore[no-untyped-def]
    ids = ["H1.rsi14>p50", "H1.rsi14>p25", "H1.linreg_slope20>0", "H1.linreg_slope20>p50", "H1.stoch14>p90", "H1.cal:day_monday"]
    cands = fit_thresholds(frames, [parse_condition_id(i, FEATURES) for i in ids], TRAIN)
    report = filter_conditions(frames, cands, TRAIN, threshold=0.9, min_rate=0.05, max_rate=0.95)
    kept = {c.id for c in report.kept}
    assert len(kept & {"H1.linreg_slope20>0", "H1.linreg_slope20>p50"}) == 1
    assert "H1.rsi14>p50" in kept and "H1.cal:day_monday" in kept
    assert "H1.stoch14>p90" in report.dropped_inactive
