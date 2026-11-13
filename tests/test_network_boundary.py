"""Every connection the package opens passes the offline switch.

The download layer is the one place that opens a connection. A library that
opens its own, such as the ERA5 store, is called only after the switch is
checked in the same function. Maintainer scripts may go online, and say so.
"""

import ast
from collections.abc import Iterator
from pathlib import Path

import pytest

REPOSITORY = Path(__file__).parent.parent
PACKAGE = REPOSITORY / "stormfork"
SCRIPTS = REPOSITORY / "scripts"
NETWORK_MODULES = {"requests", "httpx", "fsspec", "gcsfs", "s3fs", "aiohttp", "urllib3", "socket"}
CONNECTING_CALLS = {"urlopen", "build_opener", "open_zarr", "HTTPConnection", "HTTPSConnection"}
NETWORK_PROGRAMS = {"curl", "wget", "gh"}
GUARD = "require_online"
# The layer itself opens the connection that every other request goes through.
LAYER = {("stormfork/downloads.py", "open_https", "build_opener")}
# Libraries that open their own connection, each preceded by the guard.
GUARDED = {("stormfork/cli_forecast.py", "initial_conditions", "open_zarr")}


def called_name(call: ast.Call) -> str | None:
    function = call.func
    if isinstance(function, ast.Attribute):
        return function.attr
    if isinstance(function, ast.Name):
        return function.id
    return None


def runs_network_program(call: ast.Call) -> bool:
    function = call.func
    if not (
        isinstance(function, ast.Attribute)
        and isinstance(function.value, ast.Name)
        and function.value.id == "subprocess"
        and call.args
    ):
        return False
    command = call.args[0]
    return (
        isinstance(command, ast.List)
        and bool(command.elts)
        and isinstance(command.elts[0], ast.Constant)
        and command.elts[0].value in NETWORK_PROGRAMS
    )


def imported_modules(tree: ast.Module) -> Iterator[str]:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            yield from (alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            yield node.module.split(".")[0]


def network_calls(tree: ast.Module) -> Iterator[tuple[str, str, set[str]]]:
    """Yield each connecting call with its enclosing function and that function's calls."""
    for function in ast.walk(tree):
        if not isinstance(function, ast.FunctionDef):
            continue
        calls = [node for node in ast.walk(function) if isinstance(node, ast.Call)]
        names = {name for call in calls if (name := called_name(call))}
        for call in calls:
            if runs_network_program(call):
                yield function.name, "subprocess", names
            elif (name := called_name(call)) in CONNECTING_CALLS:
                yield function.name, name, names


def source_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.py"))


@pytest.mark.parametrize("path", source_files(PACKAGE), ids=lambda path: path.name)
def test_the_package_connects_only_through_the_download_layer(path: Path):
    relative = path.relative_to(REPOSITORY).as_posix()
    tree = ast.parse(path.read_text(encoding="utf-8"))
    assert not NETWORK_MODULES & set(imported_modules(tree))
    for function, call, names in network_calls(tree):
        site = (relative, function, call)
        assert site in LAYER or (site in GUARDED and GUARD in names), site


def is_online_script(tree: ast.Module) -> bool:
    if NETWORK_MODULES & set(imported_modules(tree)) or any(True for _ in network_calls(tree)):
        return True
    return any(
        isinstance(node, ast.alias)
        and node.name in {"fetch_bytes", "fetch_lock", "download_experiment"}
        for node in ast.walk(tree)
    )


@pytest.mark.parametrize("path", source_files(SCRIPTS), ids=lambda path: path.name)
def test_a_script_that_goes_online_says_so_in_its_help(path: Path):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if is_online_script(tree):
        assert "needs network access" in (ast.get_docstring(tree) or ""), path.name


def test_the_check_finds_a_direct_connection():
    tree = ast.parse("def get(url):\n    return urllib.request.urlopen(url)\n")
    assert [call for _, call, _ in network_calls(tree)] == ["urlopen"]
    tree = ast.parse("def issue():\n    subprocess.run(['gh', 'issue', 'list'])\n")
    assert [call for _, call, _ in network_calls(tree)] == ["subprocess"]
