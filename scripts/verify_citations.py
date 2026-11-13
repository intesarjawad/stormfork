"""Check that every DOI cited in docs/ and the README resolves at the DOI handle API.

A DOI is valid when https://doi.org/api/handles/<doi> answers responseCode 1.
The check never follows the redirect to the publisher: some publishers refuse
scripted requests with 403, which would fail a valid citation. It is a
maintainer check that needs network access to doi.org.
"""

import argparse
import http.client
import re
import sys
import urllib.parse
from collections.abc import Callable, Iterable, Sequence
from http import HTTPStatus
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

HANDLE_HOST = "doi.org"
HANDLE_PATH = "/api/handles/"
RESPONSE_OK = 1
TIMEOUT_S = 20
DOI = re.compile(r"\b10\.\d{4,9}/[^\s\"'<>\[\]()]+")
TRAILING = ".,;:"
DEFAULT_PATHS = (Path("docs"), Path("README.md"))
Json = TypeAdapter(JsonValue)
Fetch = Callable[[str], bytes]


def find_dois(text: str) -> list[str]:
    """Every DOI in the text, in order of first appearance, without trailing punctuation."""
    found = (match.rstrip(TRAILING) for match in DOI.findall(text))
    return list(dict.fromkeys(found))


def document_paths(paths: Iterable[Path]) -> list[Path]:
    """Markdown files under the given files and directories, sorted."""
    files: list[Path] = []
    for path in paths:
        files.extend(sorted(path.rglob("*.md")) if path.is_dir() else [path])
    return files


def fetch_handle(doi: str) -> bytes:
    """Ask the handle API about a DOI; a 404 body still carries its responseCode.

    A plain connection never follows a redirect, so the publisher is not contacted.

    Raises:
        OSError: The API cannot be reached or answers with another status.
    """
    connection = http.client.HTTPSConnection(HANDLE_HOST, timeout=TIMEOUT_S)
    try:
        connection.request("GET", HANDLE_PATH + urllib.parse.quote(doi, safe="/"))
        response = connection.getresponse()
        if response.status not in (HTTPStatus.OK, HTTPStatus.NOT_FOUND):
            msg = f"handle API answered HTTP {response.status}"
            raise OSError(msg)
        return response.read()
    finally:
        connection.close()


def check_doi(doi: str, fetch: Fetch = fetch_handle) -> str | None:
    """None when the handle API reports responseCode 1, otherwise the reason it fails."""
    try:
        body = Json.validate_json(fetch(doi))
    except (OSError, ValueError) as error:
        return f"cannot read the handle record ({error})"
    code = body.get("responseCode") if isinstance(body, dict) else None
    return None if code == RESPONSE_OK else f"responseCode {code}, expected {RESPONSE_OK}"


def main(argv: Sequence[str] | None = None, fetch: Fetch = fetch_handle) -> int:
    """Check every DOI and print one line for each failure; exit 1 if any fails."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="*", type=Path, default=list(DEFAULT_PATHS))
    arguments = parser.parse_args(argv)
    cited: dict[str, Path] = {}
    for path in document_paths(arguments.paths):
        for doi in find_dois(path.read_text(encoding="utf-8")):
            cited.setdefault(doi, path)
    failures = {doi: check_doi(doi, fetch) for doi in cited}
    for doi, reason in failures.items():
        if reason is not None:
            print(f"{cited[doi]}: {doi}: {reason}", file=sys.stderr)
    print(f"{len(cited)} DOIs checked, {sum(r is not None for r in failures.values())} failed")
    return 1 if any(reason is not None for reason in failures.values()) else 0


if __name__ == "__main__":
    raise SystemExit(main())
