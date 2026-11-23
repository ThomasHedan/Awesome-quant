from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from fxgen.backtest.engine import ExitRules, rollover_units, simulate_pair
from fxgen.backtest.metrics import compute_metrics, daily_r, max_drawdown, sharpe
from fxgen.backtest.runner import run_backtest
from fxgen.config import Config, CostsConfig
from fxgen.data.loader import load_universe
from fxgen.data.splits import Period
from fxgen.market import PairSplit
from fxgen.tree.node import Action, Leaf, Node
from fxgen.workspace import build_workspace

NO_COST = CostsConfig(commission_pips_round_turn=0, default_spread_pips=0, swap_pips_long=0, swap_pips_short=0)
RULES = ExitRules(stop_atr=1.0, take_profit_atr=2.0, max_holding_bars=None, exit_on_leaf_change=False)


def _split(close: list[float], high: list[float] | None = None, low: list[float] | None = None,
           open_: list[float] | None = None, atr: float = 0.0010) -> PairSplit:
    n = len(close)
    idx = pd.date_range("2020-01-06 00:00", periods=n, freq="1h", tz="UTC")
    c = np.array(close)
    o = np.array(open_) if open_ else np.concatenate([[c[0]], c[:-1]])
    h = np.array(high) if high else np.maximum(o, c)
    lo = np.array(low) if low else np.minimum(o, c)
    return PairSplit("EURUSD", idx, o, h, lo, c, np.full(n, atr), np.zeros((n, 0), dtype=bool))


def _signal(n: int, at: dict[int, int]) -> tuple[np.ndarray, np.ndarray]:
    action = np.zeros(n, dtype=np.int8)
    leaf = np.zeros(n, dtype=np.int16)
    for i, a in at.items():
        action[i] = a
        leaf[i] = 1
    return action, leaf


def test_signal_at_close_fills_at_next_open() -> None:
    data = _split([1.1000, 1.1001, 1.1002, 1.1003, 1.1004], open_=[1.1, 1.1, 1.1005, 1.1003, 1.1004])
    action, leaf = _signal(5, {1: 1})
    trades = simulate_pair(data, action, leaf, RULES, NO_COST)
    assert len(trades) == 1
    assert trades["entry_time"].iloc[0] == data.index[2]
    assert trades["entry_price"].iloc[0] == pytest.approx(1.1005)
    assert trades["exit_reason"].iloc[0] == "end"


def test_stop_target_and_gap_fill() -> None:
    close = [1.1, 1.1, 1.1, 1.1, 1.1]
    data = _split(close, high=[1.1, 1.1, 1.1, 1.1025, 1.1], low=[1.1, 1.1, 1.1, 1.0995, 1.1])
    action, leaf = _signal(5, {1: 1})
    t = simulate_pair(data, action, leaf, RULES, NO_COST)
    assert t["exit_reason"].iloc[0] == "target" and t["exit_price"].iloc[0] == pytest.approx(1.1020)
    assert t["r_multiple"].iloc[0] == pytest.approx(2.0)

    data = _split(close, high=[1.1, 1.1, 1.1, 1.1025, 1.1], low=[1.1, 1.1, 1.1, 1.0985, 1.1])
    t = simulate_pair(data, action, leaf, RULES, NO_COST)
    assert t["exit_reason"].iloc[0] == "stop" and t["r_multiple"].iloc[0] == pytest.approx(-1.0)

    data = _split([1.1, 1.1, 1.1, 1.0970, 1.0970], open_=[1.1, 1.1, 1.1, 1.0975, 1.097])
    t = simulate_pair(data, action, leaf, RULES, NO_COST)
    assert t["exit_reason"].iloc[0] == "stop" and t["exit_price"].iloc[0] == pytest.approx(1.0975)


def test_short_trade_and_time_exit() -> None:
    data = _split([1.1] * 10)
    action, leaf = _signal(10, {i: -1 for i in range(1, 9)})
    rules = ExitRules(1.0, None, 3, False)
    t = simulate_pair(data, action, leaf, rules, NO_COST)
    assert t["exit_reason"].iloc[0] == "time" and t["bars_held"].iloc[0] == 3
    assert len(t) == 1, "same branch must not re-enter after a time exit"
    assert (t["direction"] == -1).all()


def test_leaf_change_and_reversal_exits() -> None:
    data = _split([1.1] * 8)
    action = np.array([0, 1, 1, 1, -1, -1, 0, 0], dtype=np.int8)
    leaf = np.array([0, 1, 1, 2, 3, 3, 0, 0], dtype=np.int16)
    t = simulate_pair(data, action, leaf, ExitRules(1.0, None, None, True), NO_COST)
    assert t["exit_reason"].tolist() == ["signal", "signal", "signal", "signal"][: len(t)]
    assert t["direction"].tolist() == [1, 1, -1]
    t2 = simulate_pair(data, action, leaf, ExitRules(1.0, None, None, False), NO_COST)
    assert t2["direction"].tolist() == [1, -1]


def test_costs_and_swap() -> None:
    idx = pd.date_range("2020-01-07 20:00", periods=30, freq="1h", tz="UTC")
    units = rollover_units(idx)
    assert units[idx.get_loc(pd.Timestamp("2020-01-07 23:00", tz="UTC"))] - units[0] == 1
    assert units[-1] - units[idx.get_loc(pd.Timestamp("2020-01-08 21:00", tz="UTC"))] == 3

    data = _split([1.1] * 4)
    action, leaf = _signal(4, {1: 1})
    costs = CostsConfig(commission_pips_round_turn=0.5, spread_pips={"EURUSD": 1.0}, swap_pips_long=-1, swap_pips_short=0)
    t = simulate_pair(data, action, leaf, RULES, costs)
    assert t["cost_pips"].iloc[0] == pytest.approx(1.5)
    assert t["net_pips"].iloc[0] == pytest.approx(-1.5)


def test_metrics_basic() -> None:
    period = Period(pd.Timestamp("2020-01-01", tz="UTC"), pd.Timestamp("2021-01-01", tz="UTC"))
    trades = pd.DataFrame(
        {"entry_time": pd.to_datetime(["2020-02-01", "2020-02-02", "2020-02-07"], utc=True),
         "exit_time": pd.to_datetime(["2020-02-03", "2020-02-04", "2020-02-08"], utc=True),
         "r_multiple": [2.0, -1.0, 1.0], "bars_held": [5, 5, 5], "direction": [1, -1, 1]}
    )
    m = compute_metrics(trades, period, risk_pct=0.5)
    assert m["n_trades"] == 3 and m["profit_factor"] == pytest.approx(3.0)
    assert m["max_dd_r"] == pytest.approx(1.0) and m["max_dd_pct"] == pytest.approx(0.5)
    assert m["return_pct"] == pytest.approx(1.0)
    assert m["max_concurrent"] == 2
    d = daily_r(trades, period)
    assert d.loc[pd.Timestamp("2020-02-10", tz="UTC")] == 1.0, "Saturday exit rolls to Monday"
    assert max_drawdown(np.array([1, -2, 1, -3])) == pytest.approx(4)
    assert sharpe(pd.Series([0.0, 0.0])) == 0.0
    assert compute_metrics(trades.iloc[:0], period, 0.5)["n_trades"] == 0


def test_backtest_is_causal_under_future_corruption(cfg: Config) -> None:
    """Trades that close before ``cut`` are identical when every bar after ``cut`` is corrupted."""
    universe = load_universe(cfg, ["EURUSD", "USDJPY"])
    ws = build_workspace(cfg, universe)
    tree = Node(ws.conditions[0].id, Node(ws.conditions[5].id, Leaf(Action.LONG), Leaf(Action.SHORT)), Leaf(Action.NO_TRADE))
    period = ws.period("train+validation")
    base = run_backtest(tree, ws.tree_dataset(tree, period), ws.rules, cfg.costs, 0.5)
    cut = pd.Timestamp("2018-06-01", tz="UTC")
    for data in universe.values():
        for tf, df in data.frames.items():
            future = df.index >= cut
            df.loc[future, ["open", "high", "low", "close"]] *= np.random.default_rng(1).uniform(0.9, 1.1, (int(future.sum()), 1))
    ws2 = build_workspace(cfg, universe)
    ws2.conditions, ws2._extra = ws.conditions, {}
    dirty = run_backtest(tree, ws2.tree_dataset(tree, period), ws2.rules, cfg.costs, 0.5)
    early = lambda r: r.trades[r.trades["exit_time"] < cut - pd.Timedelta(hours=1)].reset_index(drop=True)  # noqa: E731
    assert len(early(base)) > 10
    pd.testing.assert_frame_equal(early(base), early(dirty))


def test_run_backtest_aggregates_pairs(cfg: Config) -> None:
    ws = build_workspace(cfg)
    tree = Node(ws.conditions[0].id, Leaf(Action.LONG), Leaf(Action.SHORT))
    result = run_backtest(tree, ws.tree_dataset(tree, ws.train), ws.rules, cfg.costs, 0.5)
    assert set(result.per_pair) == set(cfg.data.pairs)
    assert result.metrics["n_trades"] == sum(m["n_trades"] for m in result.per_pair.values())
    assert set(result.trades["branch"]) <= {"T", "F"}
