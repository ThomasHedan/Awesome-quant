from __future__ import annotations

import hashlib
import json
import urllib.parse
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fxgen.data.loader import load_csv
from fxgen.data.lse import LseClient, Response, download_pair, download_universe, lse_symbol, normalise_candles
from fxgen.config import DataConfig
from fxgen.errors import DataError


class FakeLse:
    """Mimics the documented /v3/export answers."""

    def __init__(self, max_days: int = 400, earliest: str | None = None, busy_once: bool = False,
                 rows_quota: int | None = None, corrupt: bool = False) -> None:
        self.max_days, self.earliest, self.busy_once = max_days, earliest, busy_once
        self.rows_quota, self.corrupt = rows_quota, corrupt
        self.calls: list[dict[str, str]] = []

    def __call__(self, url: str, headers: dict[str, str]) -> Response:
        assert headers["X-API-Key"] == "k"
        q = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(url).query))
        self.calls.append(q)
        if self.busy_once:
            self.busy_once = False
            return self._err(429, {"code": "connections-exhausted", "message": "One download at a time."}, {"retry-after": "3"})
        start, end = pd.Timestamp(q["from"]), pd.Timestamp(q["to"])
        if self.earliest and start < pd.Timestamp(self.earliest):
            return self._err(403, {"code": "history-limit", "earliest": self.earliest})
        if (end - start).days > self.max_days:
            return self._err(413, {"code": "export-too-wide", "message": "Choose shorter dates."})
        if self.rows_quota is not None:
            if self.rows_quota <= 0:
                return self._err(429, {"code": "rows-exhausted", "message": "daily rows used"}, {"retry-after": "40000"})
            self.rows_quota -= 1
        idx = pd.date_range(start, end + pd.Timedelta(days=1), freq="1h", inclusive="left")
        idx = idx[idx.day_of_week < 5]
        close = 1.1 + np.cumsum(np.full(len(idx), 0.0001))
        csv = pd.DataFrame({"time": idx.strftime("%Y-%m-%d %H:%M:%S"), "open": close, "high": close + 0.0005,
                            "low": close - 0.0005, "close": close, "volume": 0, "trades": 120}).to_csv(index=False).encode()
        digest = hashlib.sha256(csv).hexdigest()
        return Response(200, {"x-checksum-sha256": "0" * 64 if self.corrupt else digest, "x-export-rows": str(len(idx) * 60)}, csv)

    @staticmethod
    def _err(status: int, body: dict[str, object], headers: dict[str, str] | None = None) -> Response:
        return Response(status, headers or {}, json.dumps(body).encode())


def _client(fake: FakeLse, sleeps: list[float] | None = None) -> LseClient:
    return LseClient("k", fetch=fake, sleep=(sleeps.append if sleeps is not None else lambda s: None))


def test_symbol_mapping() -> None:
    assert lse_symbol("eurusd") == "EUR_USD" and lse_symbol("GBP/JPY") == "GBP_JPY"
    with pytest.raises(DataError):
        lse_symbol("EURUSDX")
    with pytest.raises(DataError, match="LSE_API_KEY"):
        LseClient("")


def test_download_splits_waits_and_writes_fxgen_files(tmp_path: Path) -> None:
    fake, sleeps = FakeLse(max_days=400, busy_once=True), []
    out = download_pair(_client(fake, sleeps), "EURUSD", "2020-01-01", "2021-12-31", tmp_path, years_per_chunk=2)
    assert sleeps == [4.0]
    assert any((pd.Timestamp(c["to"]) - pd.Timestamp(c["from"])).days > 400 for c in fake.calls)
    assert all(c["symbol"] == "EUR_USD" and c["timeframe"] == "1h" and c["format"] == "csv" for c in fake.calls)
    assert out["bars"]["H1"] > 10_000 and set(out["bars"]) == {"H1", "H4", "D1"}
    df, quality = load_csv(tmp_path / "EURUSD_H1.csv", "EURUSD", "H1", DataConfig(pairs=["EURUSD"]))
    assert quality.duplicates_removed == 0 and quality.n_gaps == 0
    assert (df["volume"] == 120).all(), "trades replace an empty volume column"
    assert (tmp_path / "EURUSD_D1.csv").exists() and (tmp_path / "EURUSD_H4.csv").exists()


def test_quota_stops_and_resumes(tmp_path: Path) -> None:
    fake = FakeLse(max_days=400, rows_quota=1)
    result = download_universe(_client(fake), ["EURUSD", "GBPUSD"], "2020-01-01", "2021-12-31", tmp_path, 1)
    assert result["remaining"] == ["EURUSD", "GBPUSD"] and "tomorrow" in result["stopped"]
    assert len(list((tmp_path / "raw" / "EURUSD").glob("*.csv"))) == 1
    fake.rows_quota = 100
    n_before = len(fake.calls)
    result = download_universe(_client(fake), ["EURUSD", "GBPUSD"], "2020-01-01", "2021-12-31", tmp_path, 1)
    assert result["remaining"] == [] and len(result["done"]) == 2
    resumed = fake.calls[n_before:]
    assert not any(c["symbol"] == "EUR_USD" and c["from"] == "2020-01-01" for c in resumed), "finished chunks are not downloaded again"


def test_history_limit_and_checksum(tmp_path: Path) -> None:
    out = download_pair(_client(FakeLse(earliest="2020-06-30")), "USDJPY", "2020-01-01", "2020-12-31", tmp_path, 1)
    assert out["start"].startswith("2020-07-01")
    with pytest.raises(DataError, match="checksum"):
        download_pair(_client(FakeLse(corrupt=True)), "AUDUSD", "2020-01-01", "2020-03-31", tmp_path, 1)


def test_normalise_rejects_unknown_columns() -> None:
    with pytest.raises(DataError, match="unexpected LSE columns"):
        normalise_candles(pd.DataFrame({"t": [1], "o": [1]}))
