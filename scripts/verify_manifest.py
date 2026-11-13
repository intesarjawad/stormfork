"""Validate every experiment's reproductions.json and hold added entries to the PR author.

A reproduction entry's `by` is whatever the contributor typed. Given the pull
request author and the base commit, an entry that is not already on the base
must name that author, and so must the `run_by` of a new or changed manifest.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from pydantic import JsonValue, TypeAdapter, ValidationError

from stormfork.export.reproduction import Reproductions

EXPERIMENTS_DIR = Path("site/public/experiments")
LEDGER = "reproductions.json"
MANIFEST = "manifest.json"
Json = TypeAdapter(JsonValue)


def git(*arguments: str) -> subprocess.CompletedProcess[str]:
    """Run git in the current directory and keep its output."""
    return subprocess.run(  # noqa: S603
        ["git", *arguments],  # noqa: S607
        capture_output=True,
        text=True,
        check=False,
    )


def read_at_base(base: str, path: Path) -> JsonValue | None:
    """Read a JSON file as it is at the base commit, or None when the base lacks it.

    The commit is read directly, so a shallow checkout needs no merge base.

    Raises:
        ValueError: The base is not a commit or the file at the base is not JSON.
    """
    if git("rev-parse", "--verify", f"{base}^{{commit}}").returncode:
        msg = f"{base} is not a commit; fetch the base branch first"
        raise ValueError(msg)
    shown = git("show", f"{base}:./{Path(os.path.relpath(path)).as_posix()}")
    return Json.validate_json(shown.stdout) if shown.returncode == 0 else None


def entries_of(value: JsonValue | None) -> list[JsonValue]:
    """The entries of a ledger document, or none for an absent or malformed one."""
    return list(value) if isinstance(value, list) else []


def added_entries(current: list[JsonValue], base: list[JsonValue]) -> list[JsonValue]:
    """Entries in current beyond those on the base, matched one for one."""
    remaining = list(base)
    added: list[JsonValue] = []
    for entry in current:
        if entry in remaining:
            remaining.remove(entry)
        else:
            added.append(entry)
    return added


def author_of(entry: JsonValue, key: str) -> str | None:
    """The text under key in a JSON object, or None."""
    value = entry.get(key) if isinstance(entry, dict) else None
    return value if isinstance(value, str) else None


def is_author(name: str | None, author: str) -> bool:
    """GitHub logins are case-insensitive."""
    return name is not None and name.casefold() == author.casefold()


def check_ledgers(experiments: Path) -> list[str]:
    """Validate each reproductions.json with the manifest's own models."""
    problems: list[str] = []
    for path in sorted(experiments.glob(f"*/{LEDGER}")):
        try:
            Reproductions.validate_json(path.read_bytes())
        except ValidationError as error:
            problems.append(f"{path}: {error}")
    return problems


def check_authors(experiments: Path, author: str, base: str) -> list[str]:
    """Report added ledger entries and new run_by values that name someone else."""
    problems: list[str] = []
    for ledger in sorted(experiments.glob(f"*/{LEDGER}")):
        current = entries_of(Json.validate_json(ledger.read_bytes()))
        added = added_entries(current, entries_of(read_at_base(base, ledger)))
        problems.extend(
            f"{ledger}: added entry is by {author_of(entry, 'by')!r}, not the author {author!r}"
            for entry in added
            if not is_author(author_of(entry, "by"), author)
        )
    for manifest in sorted(experiments.glob(f"*/{MANIFEST}")):
        run_by = author_of(Json.validate_json(manifest.read_bytes()), "run_by")
        before = read_at_base(base, manifest)
        changed = run_by is not None and run_by != author_of(before, "run_by")
        if changed and not is_author(run_by, author):
            problems.append(f"{manifest}: run_by is {run_by!r}, not the author {author!r}")
    return problems


def main(argv: list[str] | None = None) -> int:
    """Validate the ledgers and, given a pull request, who added what."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiments", type=Path, default=EXPERIMENTS_DIR)
    parser.add_argument("--pr-author", help="GitHub login of the pull request author")
    parser.add_argument("--base", help="base commit or ref to compare with")
    args = parser.parse_args(argv)
    if (args.pr_author is None) != (args.base is None):
        parser.error("--pr-author and --base must be supplied together")
    problems = check_ledgers(args.experiments)
    if args.pr_author is not None and not problems:
        try:
            problems = check_authors(args.experiments, args.pr_author, args.base)
        except (ValueError, json.JSONDecodeError) as error:
            problems = [str(error)]
    for problem in problems:
        print(problem, file=sys.stderr)
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
