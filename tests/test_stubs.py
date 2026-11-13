import ast
import importlib
import inspect
from importlib.util import find_spec
from pathlib import Path

import pytest


def test_fiddle_stub_matches_pinned_signature():
    if find_spec("weathernext") is None:
        pytest.skip("model dependency group is absent")
    module = importlib.import_module("weathernext.utils.fiddle_config_io")
    path = Path("typings/weathernext/utils/fiddle_config_io.pyi")
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    assert [node.name for node in functions] == ["get_fiddle_config_by_name"]
    stub = functions[0]
    signature = inspect.signature(getattr(module, stub.name))
    assert [arg.arg for arg in stub.args.args] == list(signature.parameters)
    assert all(
        parameter.default is inspect.Parameter.empty for parameter in signature.parameters.values()
    )


@pytest.mark.parametrize(
    ("module_name", "stub_path"),
    [
        ("haiku", "typings/haiku/__init__.pyi"),
        ("jax", "typings/jax/__init__.pyi"),
        ("jax.random", "typings/jax/random.pyi"),
        ("xarray", "typings/xarray/__init__.pyi"),
        ("weathernext.utils.checkpoint", "typings/weathernext/utils/checkpoint.pyi"),
        ("weathernext.utils.data_utils", "typings/weathernext/utils/data_utils.pyi"),
        ("weathernext.weathernext2.fgn", "typings/weathernext/weathernext2/fgn.pyi"),
    ],
)
def test_smoke_stub_parameters_match_pinned_source(module_name: str, stub_path: str):
    if find_spec("weathernext") is None:
        pytest.skip("model dependency group is absent")
    module = importlib.import_module(module_name)
    tree = ast.parse(Path(stub_path).read_text())
    for node in tree.body:
        if not isinstance(node, ast.FunctionDef):
            continue
        actual = inspect.signature(getattr(module, node.name))
        arguments = node.args.posonlyargs + node.args.args + node.args.kwonlyargs
        names = [argument.arg for argument in arguments]
        if node.args.kwarg:
            names.append(node.args.kwarg.arg)
        assert names == [name for name in actual.parameters if name in names]
        for argument in node.args.kwonlyargs:
            assert actual.parameters[argument.arg].kind == inspect.Parameter.KEYWORD_ONLY
        for name, parameter in actual.parameters.items():
            if parameter.default is inspect.Parameter.empty and parameter.kind not in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                assert name in names


@pytest.mark.parametrize("class_name", ["PredictorConfig", "CheckPoint"])
def test_smoke_dataclass_stubs_match_pinned_source(class_name: str):
    if find_spec("weathernext") is None:
        pytest.skip("model dependency group is absent")
    module = importlib.import_module("weathernext.weathernext2.fgn")
    tree = ast.parse(Path("typings/weathernext/weathernext2/fgn.pyi").read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name
    )
    fields = [
        node.target.id
        for node in cls.body
        if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name)
    ]
    assert fields == list(inspect.signature(getattr(module, class_name)).parameters)


@pytest.mark.parametrize("name", ["read", "validate"])
def test_notebook_stub_parameters_match_installed_source(name: str):
    nbformat = importlib.import_module("nbformat")
    tree = ast.parse(Path("typings/nbformat/__init__.pyi").read_text())
    stub = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name
    )
    actual = list(inspect.signature(getattr(nbformat, name)).parameters)
    names = [argument.arg for argument in stub.args.args]
    assert names == [parameter for parameter in actual if parameter in names]
