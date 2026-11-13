"""Print the experiment schema from its model, or check that a document prints the same one.

The recipes guide shows the schema of stormfork.scenarios.Scenario. The check
keeps that copy from drifting: it parses the JSON block after the marker and
compares it with the model's schema, ignoring formatting.
"""

import argparse
import json
import re
import sys
from pathlib import Path

from pydantic import JsonValue, TypeAdapter

from stormfork.scenarios import Scenario

MARKER = "<!-- recipe-schema -->"
SCHEMA_BLOCK = re.compile(re.escape(MARKER) + r"\n```json\n(.*?)\n```", re.DOTALL)
Json = TypeAdapter(JsonValue)


def schema_text() -> str:
    """The model's JSON Schema, indented as the docs print it."""
    return json.dumps(Scenario.model_json_schema(), indent=2)


def check_document(path: Path) -> None:
    """Require the schema block in a document to equal the model's schema.

    Raises:
        ValueError: The block is missing, is not JSON or differs from the model.
        OSError: The document cannot be read.
    """
    found = SCHEMA_BLOCK.search(path.read_text(encoding="utf-8"))
    if found is None:
        msg = f"{path} has no json block after {MARKER}"
        raise ValueError(msg)
    if Json.validate_json(found.group(1)) != Json.validate_json(schema_text()):
        msg = f"{path}: the schema block differs from the model; paste the output of {__file__}"
        raise ValueError(msg)


def main(argv: list[str] | None = None) -> int:
    """Print the schema, or with --check verify a document and exit 1 if it differs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", type=Path, metavar="DOC", help="document holding the schema")
    arguments = parser.parse_args(argv)
    if arguments.check is None:
        print(schema_text())
        return 0
    try:
        check_document(arguments.check)
    except (ValueError, OSError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
