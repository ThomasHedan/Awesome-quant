"""Downloader for London Strategic Edge (LSE Data 3.0) forex candles.

The API is ``GET https://download.londonstrategicedge.com/v3/<call>`` with the
key in the ``X-API-Key`` header (free key from the Databank page). Only H1
candles are downloaded; H4 and D1 are rebuilt locally with the 22:00 UTC daily
roll, which saves quota and keeps every timeframe consistent.

Free-plan limits that shape this module (from the API documentation):

* an export may read at most 2,500,000 stored rows; forex H1 candles are
  folded from 1-minute bars, so one year of one pair reads about 375,000 rows
  and exports are split into chunks of a few years (``413 export-too-wide``
  halves the chunk);
* 12,500,000 rows a day (``429 rows-exhausted``), one download at a time
  (``429 connections-exhausted``), 60 calls a minute, a 10-year history window
  (``403 history-limit``).

Every chunk is written to ``<out>/raw/<PAIR>/`` once its SHA-256 matches, so a
download interrupted by a quota resumes where it stopped.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from fxgen.data.resample import resample
from fxgen.errors import DataError

logger = logging.getLogger(__name__)

BASE_URL = "https://download.londonstrategicedge.com"
MAX_WAIT_SECONDS = 3700


class QuotaExhausted(DataError):
    """The plan's allowance is used up; run the same command again later to resume."""


@dataclass
class Response:
    status: int
    headers: dict[str, str]
    body: bytes

    def json(self) -> dict[str, object]:
        try:
            return json.loads(self.body.decode())
        except (ValueError, UnicodeDecodeError):
            return {}


Fetch = Callable[[str, dict[str, str]], Response]


def urllib_fetch(url: str, headers: dict[str, str]) -> Response:
    request = urllib.request.Request(url, headers=headers)
    try:
        with urllib.request.urlopen(request, timeout=600) as resp:
            return Response(resp.status, {k.lower(): v for k, v in resp.headers.items()}, resp.read())
    except urllib.error.HTTPError as exc:
        return Response(exc.code, {k.lower(): v for k, v in exc.headers.items()}, exc.read())


def lse_symbol(pair: str) -> str:
    """``EURUSD`` -> ``EUR_USD``."""
    pair = pair.upper().replace("/", "").replace("_", "")
    if len(pair) != 6:
        raise DataError(f"{pair!r} is not a 6-letter FX pair")
    return f"{pair[:3]}_{pair[3:]}"


class LseClient:
    def __init__(self, api_key: str, fetch: Fetch = urllib_fetch, base_url: str = BASE_URL,
                 sleep: Callable[[float], None] = time.sleep) -> None:
        if not api_key:
            raise DataError("no LSE API key: get a free key on londonstrategicedge.com/data and set LSE_API_KEY")
        self._key = api_key
        self._fetch = fetch
        self._base = base_url.rstrip("/")
        self._sleep = sleep

    def get(self, call: str, params: dict[str, str]) -> Response:
        """GET a /v3 call, waiting out short rate limits. Raises on quotas and auth errors."""
        url = f"{self._base}/v3/{call}?{urllib.parse.urlencode(params)}"
        while True:
            resp = self._fetch(url, {"X-API-Key": self._key, "Accept": "*/*", "User-Agent": "fxgen"})
            if resp.status == 200:
                return resp
            body = resp.json()
            code = str(body.get("code", ""))
            message = str(body.get("message", resp.body[:200]))
            if resp.status == 401:
                raise DataError(f"LSE rejected the API key ({code}): check LSE_API_KEY")
            retry_after = resp.headers.get("retry-after") or body.get("retry_after")
            if resp.status in (429, 503) and code != "rows-exhausted":
                wait = float(retry_after) if retry_after is not None else 30.0
                if wait > MAX_WAIT_SECONDS:
                    raise QuotaExhausted(f"LSE {code}: {message}. Run the same command again later to resume.")
                logger.info("LSE %s, waiting %.0f s", code or resp.status, wait)
                self._sleep(wait + 1)
                continue
            if code == "rows-exhausted":
                raise QuotaExhausted(f"LSE daily row allowance used up ({message}). Run the same command again "
                                     "tomorrow (reset at 00:00 UTC) to resume.")
            return resp


def _chunks(start: pd.Timestamp, end: pd.Timestamp, years: float) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    out = []
    step = pd.DateOffset(days=round(365.25 * years))
    current = start
    while current <= end:
        stop = min(current + step - pd.Timedelta(days=1), end)
        out.append((current, stop))
        current = stop + pd.Timedelta(days=1)
    return out


def _download_chunk(client: LseClient, symbol: str, start: pd.Timestamp, end: pd.Timestamp, raw_dir: Path) -> list[Path]:
    path = raw_dir / f"{start.date()}_{end.date()}.csv"
    if path.exists():
        return [path]
    resp = client.get("export", {"market": "forex", "symbol": symbol, "timeframe": "1h", "from": str(start.date()),
                                 "to": str(end.date()), "format": "csv"})
    if resp.status == 200:
        expected = resp.headers.get("x-checksum-sha256")
        if expected and hashlib.sha256(resp.body).hexdigest() != expected.lower():
            raise DataError(f"{symbol} {start.date()}..{end.date()}: checksum mismatch, the file was truncated; run again")
        tmp = path.with_suffix(".part")
        tmp.write_bytes(resp.body)
        tmp.replace(path)
        logger.info("%s %s..%s: %s rows", symbol, start.date(), end.date(), resp.headers.get("x-export-rows", "?"))
        return [path]
    body = resp.json()
    code = body.get("code")
    if code == "export-too-wide" and end > start:
        middle = start + (end - start) / 2
        middle = pd.Timestamp(middle.date())
        logger.info("%s %s..%s too wide for one export, splitting", symbol, start.date(), end.date())
        return (_download_chunk(client, symbol, start, middle, raw_dir)
                + _download_chunk(client, symbol, middle + pd.Timedelta(days=1), end, raw_dir))
    if code == "history-limit" and body.get("earliest"):
        earliest = pd.Timestamp(str(body["earliest"])[:10]) + pd.Timedelta(days=1)
        if earliest > end:
            logger.warning("%s %s..%s is entirely before the plan's history window", symbol, start.date(), end.date())
            return []
        logger.warning("%s: the plan's history starts at %s, skipping earlier dates", symbol, earliest.date())
        return _download_chunk(client, symbol, earliest, end, raw_dir)
    raise DataError(f"LSE export {symbol} {start.date()}..{end.date()} failed: {resp.status} {code} {body.get('message', '')}")


def normalise_candles(raw: pd.DataFrame) -> pd.DataFrame:
    """LSE candle CSV -> fxgen OHLCV frame (UTC open time index, float columns)."""
    missing = {"time", "open", "high", "low", "close"} - set(raw.columns)
    if missing:
        raise DataError(f"unexpected LSE columns {list(raw.columns)}; missing {sorted(missing)}")
    df = raw.copy()
    df["time"] = pd.to_datetime(df["time"], utc=True)
    out = pd.DataFrame({c: pd.to_numeric(df[c], errors="coerce") for c in ("open", "high", "low", "close")})
    out["volume"] = 0.0
    for col in ("volume", "trades"):
        if col in df:
            values = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
            if values.sum() > 0:
                out["volume"] = values.to_numpy()
                break
    out.index = pd.DatetimeIndex(df["time"]).rename("time")
    out = out[~out.index.duplicated(keep="last")].sort_index()
    return out.dropna(subset=["open", "high", "low", "close"])


def download_pair(client: LseClient, pair: str, start: str, end: str, out_dir: Path,
                  years_per_chunk: float = 5.0, file_pattern: str = "{pair}_{timeframe}.csv") -> dict[str, object]:
    """Download one pair's H1 candles and write H1, H4 and D1 CSV files for fxgen."""
    symbol = lse_symbol(pair)
    raw_dir = out_dir / "raw" / pair.upper()
    raw_dir.mkdir(parents=True, exist_ok=True)
    files: list[Path] = []
    for a, b in _chunks(pd.Timestamp(start), pd.Timestamp(end), years_per_chunk):
        files += _download_chunk(client, symbol, a, b, raw_dir)
    if not files:
        raise DataError(f"{pair}: no data downloaded")
    h1 = normalise_candles(pd.concat([pd.read_csv(f) for f in sorted(files)], ignore_index=True))
    written = {}
    for tf, frame in {"H1": h1, "H4": resample(h1, "H4"), "D1": resample(h1, "D1")}.items():
        path = out_dir / file_pattern.format(pair=pair.upper(), timeframe=tf)
        out = frame.copy()
        out.index = out.index.strftime("%Y-%m-%d %H:%M:%S")
        out.to_csv(path, index_label="time", float_format="%.6f")
        written[tf] = len(frame)
    return {"pair": pair.upper(), "bars": written, "start": str(h1.index[0]), "end": str(h1.index[-1])}


def download_universe(client: LseClient, pairs: list[str], start: str, end: str, out_dir: Path,
                      years_per_chunk: float = 5.0) -> dict[str, object]:
    """Download every pair; stops cleanly on an exhausted quota and reports what is left."""
    done, remaining = [], list(pairs)
    for pair in pairs:
        try:
            done.append(download_pair(client, pair, start, end, out_dir, years_per_chunk))
            remaining.remove(pair)
        except QuotaExhausted as exc:
            logger.warning("%s", exc)
            return {"done": done, "remaining": remaining, "stopped": str(exc)}
    return {"done": done, "remaining": [], "stopped": None}


def estimate_rows(client: LseClient, pair: str, start: str, end: str) -> dict[str, object]:
    """Rows an H1 export would read (no download, no row allowance used)."""
    resp = client.get("estimate", {"market": "forex", "symbol": lse_symbol(pair), "timeframe": "1h",
                                   "from": start, "to": end, "format": "csv"})
    return resp.json()
