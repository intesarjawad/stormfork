from pathlib import Path

import pytest

from stormfork.cli import main


def test_missing_towns_has_one_next_step_without_traceback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
):
    monkeypatch.setattr(
        "sys.argv",
        [
            "stormfork",
            "scores",
            "collect",
            "--output",
            str(tmp_path),
            "--towns",
            str(tmp_path / "missing.json"),
        ],
    )
    with pytest.raises(SystemExit) as raised:
        main()
    assert raised.value.code == 1
    lines = capsys.readouterr().err.splitlines()
    assert len(lines) == 2
    assert lines[0].startswith("error: ")
    assert "missing.json" in lines[0]
    assert lines[1].startswith("next: ")
