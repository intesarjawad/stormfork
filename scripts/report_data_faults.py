"""Report quarantined scoreboard cells in one open data-fault issue.

The workflow runs this after rebuilding scores.json. A quarantined cell is a
data fault, not evidence, so it is reported once and the page links to the
same label search. With no quarantined cells the script makes no GitHub call.
It needs network access and an authenticated GitHub CLI, as in the workflow.
"""

import argparse
import json
import subprocess
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

LABEL = "data-fault"
TITLE = "Scoreboard data fault: degenerate cells"
LIST_ARGUMENTS = ("issue", "list", "--label", LABEL, "--state", "open", "--json", "number")
ObjectDict = TypeAdapter(dict[str, object])
ObjectList = TypeAdapter(list[object])


def quarantined_cells(document: object) -> list[dict[str, object]]:
    """Return the cells marked quarantined, accepting a board or a bare list."""
    if isinstance(document, dict):
        cells: object = ObjectDict.validate_python(document).get("cells")
    elif isinstance(document, list):
        cells = ObjectList.validate_python(document)
    else:
        msg = "scores.json is not an object or a list"
        raise TypeError(msg)
    try:
        items = ObjectList.validate_python(cells)
    except ValidationError as error:
        msg = "scores.json has no cells list"
        raise TypeError(msg) from error
    result: list[dict[str, object]] = []
    for cell in items:
        if not isinstance(cell, dict):
            continue
        record = ObjectDict.validate_python(cell)
        if record.get("quarantined") is True:
            result.append(record)
    return result


def cell_line(cell: dict[str, object]) -> str:
    """One markdown line naming the cell and the values that made it degenerate."""
    return (
        f"- {cell.get('town')} {cell.get('variable')} lead {cell.get('lead_days')} d "
        f"({cell.get('model')}): mean difference {cell.get('mean_difference')}, "
        f"block sd {cell.get('block_sd')}, {cell.get('blocks')} blocks"
    )


def body_text(cells: Sequence[dict[str, object]], generated_at: object) -> str:
    """The issue body for today's quarantined cells."""
    return "\n".join(
        [
            f"Quarantined cells in the scoreboard generated at {generated_at}.",
            "",
            (
                "A cell is quarantined when its weekly block means have zero spread around a "
                "non-zero mean, which is a data fault rather than evidence."
            ),
            "",
            *[cell_line(cell) for cell in cells],
            "",
        ]
    )


def run_gh(arguments: Sequence[str]) -> str:
    """Run one gh command and return its standard output."""
    result = subprocess.run(  # noqa: S603
        ["gh", *arguments],  # noqa: S607
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout


def open_issue_numbers() -> list[int]:
    """The numbers of the open data-fault issues, oldest first as gh returns them."""
    records = json.loads(run_gh(LIST_ARGUMENTS))
    return [int(record["number"]) for record in records]


def report(path: Path) -> None:
    """Print the quarantined cells and comment on or create the single open issue."""
    document: object = json.loads(path.read_text(encoding="utf-8"))
    cells = quarantined_cells(document)
    if not cells:
        return
    for cell in cells:
        print(cell_line(cell))
    generated_at = (
        ObjectDict.validate_python(document).get("generated_at")
        if isinstance(document, dict)
        else None
    )
    with tempfile.TemporaryDirectory() as directory:
        body_path = Path(directory) / "body.md"
        body_path.write_text(body_text(cells, generated_at), encoding="utf-8")
        numbers = open_issue_numbers()
        if numbers:
            run_gh(("issue", "comment", str(numbers[0]), "--body-file", str(body_path)))
        else:
            run_gh(
                (
                    "issue",
                    "create",
                    "--label",
                    LABEL,
                    "--title",
                    TITLE,
                    "--body-file",
                    str(body_path),
                )
            )


def main(argv: Sequence[str] | None = None) -> int:
    """Report the quarantined cells in one scores.json file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scores", type=Path, help="the rebuilt scores.json")
    arguments = parser.parse_args(argv)
    try:
        report(arguments.scores)
    except (OSError, TypeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
