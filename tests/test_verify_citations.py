import json
from pathlib import Path

import pytest

from scripts.verify_citations import check_doi, find_dois, main

DOI = "10.1002/2015GL063525"


def handle(code: int) -> bytes:
    return json.dumps({"responseCode": code, "handle": DOI}).encode()


def test_find_dois_in_links_and_prose_without_trailing_punctuation():
    text = (
        "See [Selz and Craig](https://doi.org/10.1002/2015GL063525), "
        "also doi:10.1175/2009BAMS2755.1. Again 10.1002/2015GL063525; "
        "a version 3.10.1 and an address 10.0.0.1/path are not DOIs."
    )
    assert find_dois(text) == ["10.1002/2015GL063525", "10.1175/2009BAMS2755.1"]


def test_a_doi_is_valid_only_with_response_code_one():
    assert check_doi(DOI, lambda _: handle(1)) is None
    assert check_doi(DOI, lambda _: handle(100)) == "responseCode 100, expected 1"
    assert check_doi(DOI, lambda _: b"{}") == "responseCode None, expected 1"
    assert "cannot read" in (check_doi(DOI, lambda _: b"not json") or "")


def test_a_network_error_is_a_failure_not_a_pass():
    def refuse(_: str) -> bytes:
        msg = "offline"
        raise OSError(msg)

    assert "offline" in (check_doi(DOI, refuse) or "")


def test_main_checks_each_doi_once_and_reports_failures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    (tmp_path / "a.md").write_text(f"{DOI} and 10.1000/missing\n")
    (tmp_path / "b.md").write_text(f"{DOI}\n")
    asked: list[str] = []

    def fetch(doi: str) -> bytes:
        asked.append(doi)
        return handle(1 if doi == DOI else 100)

    assert main([str(tmp_path)], fetch) == 1
    captured = capsys.readouterr()
    assert asked == [DOI, "10.1000/missing"]
    assert "a.md: 10.1000/missing: responseCode 100, expected 1" in captured.err
    assert "2 DOIs checked, 1 failed" in captured.out


def test_main_passes_when_every_doi_resolves(tmp_path: Path):
    (tmp_path / "a.md").write_text(f"[x](https://doi.org/{DOI})\n")
    assert main([str(tmp_path)], lambda _: handle(1)) == 0
