import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
import xarray as xr
import yaml

from scripts.recipes_schema import MARKER, main, schema_text
from stormfork.cli_model import resolve_recipe
from stormfork.grid import great_circle_km
from stormfork.scenarios import load_scenarios
from tests.test_cli import write_netcdf

DOC = Path(__file__).parent.parent / "docs" / "recipes.md"
YAML_BLOCK = re.compile(r"```yaml\n(.*?)```", re.DOTALL)
PYTHON_BLOCK = re.compile(r"```python\n(.*?)```", re.DOTALL)
EARTH_RADIUS_KM = 6371.0088


def worked_recipes() -> list[str]:
    return YAML_BLOCK.findall(DOC.read_text(encoding="utf-8"))


def test_the_schema_in_the_guide_equals_the_models(capsys: pytest.CaptureFixture[str]):
    assert main(["--check", str(DOC)]) == 0
    assert capsys.readouterr().err == ""


def test_a_drifted_schema_or_a_missing_block_fails_the_check(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    drifted = json.loads(schema_text())
    drifted["properties"]["seeds"] = {"type": "string"}
    document = tmp_path / "recipes.md"
    document.write_text(f"{MARKER}\n```json\n{json.dumps(drifted, indent=2)}\n```\n")
    assert main(["--check", str(document)]) == 1
    assert "differs from the model" in capsys.readouterr().err
    document.write_text("no schema here\n")
    assert main(["--check", str(document)]) == 1
    assert "has no json block" in capsys.readouterr().err
    assert main(["--check", str(tmp_path / "absent.md")]) == 1


def test_every_worked_recipe_loads_with_the_scenario_loader(tmp_path: Path):
    blocks = worked_recipes()
    assert len(blocks) == 5
    for block in blocks:
        (tmp_path / f"{yaml.safe_load(block)['id']}.yaml").write_text(block)
    loaded = {scenario.id: scenario for scenario in load_scenarios(tmp_path)}
    assert set(loaded) == {
        "my_relocation",
        "my_scaling",
        "my_warm_gulf",
        "my_noise",
        "my_scaling_sweep",
    }
    assert {s.edit.type for s in loaded.values() if s.edit} == {
        "relocate",
        "scale_intensity",
        "sst_anomaly",
        "noise",
    }
    assert loaded["my_scaling_sweep"].kind == "sweep"
    for scenario in loaded.values():
        assert scenario.source == "hres_sample"
        assert scenario.title


def test_a_worked_recipe_resolves_as_a_rerun_target(tmp_path: Path):
    recipe = tmp_path / "my.yaml"
    recipe.write_text(worked_recipes()[0])
    assert resolve_recipe(str(recipe)).id == "my_relocation"


def test_the_python_example_relocates_a_low_on_a_quarter_degree_grid(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    (example,) = PYTHON_BLOCK.findall(DOC.read_text(encoding="utf-8"))
    assert len(example.splitlines()) == 15
    lat = np.arange(10, 40.25, 0.25)
    lon = np.arange(0, 360, 0.25)
    distance = great_circle_km(25.0, 280.0, lat[:, None], lon)
    write_netcdf(
        xr.Dataset(
            {"mslp": (("lat", "lon"), 101000 - 6000 * np.exp(-0.5 * (distance / 130) ** 2))},
            coords={"lat": lat, "lon": lon},
        ),
        tmp_path / "grid.nc",
    )
    monkeypatch.chdir(tmp_path)
    exec(example, {})  # noqa: S102
    before, after = (line.split() for line in capsys.readouterr().out.splitlines())
    assert before == ["before", "25.0", "280.0"]
    expected_lat = 25.0 + math.degrees(240.0 / EARTH_RADIUS_KM)
    expected_lon = 280.0 + math.degrees(180.0 / (EARTH_RADIUS_KM * math.cos(math.radians(25.0))))
    assert after[0] == "after"
    assert float(after[1]) == pytest.approx(expected_lat, abs=0.25)
    assert float(after[2]) == pytest.approx(expected_lon, abs=0.25)
    assert (tmp_path / "moved.nc").is_file()
