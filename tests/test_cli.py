from __future__ import annotations

import json
from pathlib import Path

import pytest

from fxgen.cli import main, parse_overrides
from fxgen.errors import FxgenError


def test_parse_overrides() -> None:
    assert parse_overrides(["ga.generations=5", "tree.max_depth=4", "data.pairs=[EURUSD, GBPUSD]"]) == {
        "ga": {"generations": 5}, "tree": {"max_depth": 4}, "data": {"pairs": ["EURUSD", "GBPUSD"]}}
    with pytest.raises(FxgenError):
        parse_overrides(["nonsense"])


def test_synth_and_datasets(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    out = tmp_path / "d"
    assert main(["synth", "--out", str(out), "--start", "2020-01-05", "--end", "2020-03-01"]) == 0
    assert len(list(out.glob("*.csv"))) == 36
    assert main(["--set", f"data.directory={out}", "--set", f"storage.database={tmp_path / 'x.db'}", "datasets"]) == 0
    rows = json.loads(capsys.readouterr().out)
    assert len(rows) == 36 and rows[0]["n_gaps"] == 0


def test_cli_reports_errors(tmp_path: Path) -> None:
    assert main(["--set", f"storage.database={tmp_path / 'x.db'}", "holdout"]) == 1


def test_indicators_doc_is_up_to_date(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["indicators"]) == 0
    generated = capsys.readouterr().out
    doc = Path(__file__).resolve().parents[1] / "docs" / "INDICATORS.md"
    assert generated == doc.read_text(), "run `uv run fxgen indicators > docs/INDICATORS.md`"
