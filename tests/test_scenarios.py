import csv
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import pytest
import yaml
from pydantic import ValidationError

from stormfork.cli_data import taper_from_csv
from stormfork.errors import UserError
from stormfork.scenarios import Scenario, load_scenarios
from tests.test_cli import invoke

FIXTURES = Path(__file__).parent / "fixtures"
IBTRACS = FIXTURES / "ibtracs_scenarios.csv"


def at(text: str) -> datetime:
    return datetime.fromisoformat(text)


IDS = {
    "sandy_east_300km",
    "sandy_sweep",
    "katrina_scale_0p7",
    "milton_noise_1pct",
    "milton_gulf_minus2k",
    "milton_south_75km",
    "hindcast_milton_2024",
    "hindcast_helene_2024",
    "hindcast_melissa_2025",
}


@pytest.fixture(scope="module")
def scenarios() -> dict[str, Scenario]:
    return {scenario.id: scenario for scenario in load_scenarios()}


def valid_fork() -> dict[str, Any]:
    return {
        "id": "example",
        "kind": "fork",
        "storm": "2005236N23285",
        "init": "2005-08-27T00:00Z",
        "source": "era5",
        "edit": {"type": "scale_intensity", "params": {"factor": 0.7}},
        "taper": {"centre": [24.6, -83.3], "radius_km": 400},
        "seeds": {"baseline": 0, "control": 1000},
    }


def test_list_prints_every_id_and_init(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    invoke(monkeypatch, "scenarios", "list")
    lines = capsys.readouterr().out.splitlines()
    rows = [line.split() for line in lines]
    assert len(rows) == 15
    assert {row[0] for row in rows} == IDS
    assert len({row[3] for row in rows}) == 12
    assert ["sandy_east_300km", "fork", "era5", "2012-10-26T00:00Z"] in rows
    assert ["milton_noise_1pct", "fork", "hres_sample", "2024-10-07T00:00Z"] in rows
    assert max(map(len, lines)) <= 80


def test_every_experiment_fixes_the_seeds(scenarios: dict[str, Scenario]):
    assert set(scenarios) == IDS
    for scenario in scenarios.values():
        assert (scenario.seeds.baseline, scenario.seeds.control) == (0, 1000)


def test_experiment_definitions(scenarios: dict[str, Scenario]):
    def single(identifier: str) -> datetime:
        (init,) = scenarios[identifier].init
        return init

    assert single("sandy_east_300km") == at("2012-10-26T00:00")
    assert single("sandy_sweep") == at("2012-10-26T00:00")
    assert single("katrina_scale_0p7") == at("2005-08-27T00:00")
    for identifier in ("milton_noise_1pct", "milton_gulf_minus2k", "milton_south_75km"):
        assert single(identifier) == at("2024-10-07T00:00")
        assert scenarios[identifier].source == "hres_sample"
    for identifier in ("sandy_east_300km", "sandy_sweep", "katrina_scale_0p7"):
        assert scenarios[identifier].source == "era5"

    east = scenarios["sandy_east_300km"].edit
    assert east is not None
    assert (east.type, east.params["east_km"]) == ("relocate", 300)
    sweep = scenarios["sandy_sweep"].sweep
    assert sweep is not None
    assert (sweep.param, sweep.values) == ("east_km", [-300, -150, 0, 150, 300, 450, 600])
    scale = scenarios["katrina_scale_0p7"].edit
    assert scale is not None
    assert (scale.type, scale.params) == ("scale_intensity", {"factor": 0.7})
    noise = scenarios["milton_noise_1pct"].edit
    assert noise is not None
    assert (noise.type, noise.params["amplitude"]) == ("noise", 0.01)
    sst = scenarios["milton_gulf_minus2k"].edit
    assert sst is not None
    assert sst.type == "sst_anomaly"
    assert sst.params == {"centre": [25.5, -86.0], "sigma_km": 400, "delta_k": -2}
    south = scenarios["milton_south_75km"].edit
    assert south is not None
    assert (south.type, south.params["north_km"]) == ("relocate", -75)


def test_hindcast_inits_are_listed(scenarios: dict[str, Scenario]):
    assert scenarios["hindcast_milton_2024"].init == (
        at("2024-10-04T00:00"),
        at("2024-10-06T00:00"),
        at("2024-10-08T00:00"),
    )
    assert scenarios["hindcast_helene_2024"].init == (
        at("2024-09-21T00:00"),
        at("2024-09-23T00:00"),
        at("2024-09-25T00:00"),
    )
    assert scenarios["hindcast_melissa_2025"].init == (
        at("2025-10-22T12:00"),
        at("2025-10-24T12:00"),
        at("2025-10-26T12:00"),
    )
    for identifier in ("hindcast_milton_2024", "hindcast_helene_2024", "hindcast_melissa_2025"):
        assert scenarios[identifier].edit is None


def test_era5_inits_precede_the_store_end(scenarios: dict[str, Scenario]):
    attributes = json.loads((FIXTURES / "era5_store_attrs.json").read_text())["attributes"]
    stop = at(attributes["valid_time_stop"])
    era5 = [s for s in scenarios.values() if s.source == "era5"]
    assert len(era5) == 6
    for scenario in era5:
        assert all(init < stop for init in scenario.init), scenario.id


def first_landfalls() -> dict[str, datetime]:
    """Earliest USA landfall record ("L") per storm in the retained IBTrACS rows."""
    landfalls: dict[str, datetime] = {}
    with IBTRACS.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if row["USA_RECORD"].strip() != "L":
                continue
            time = at(row["ISO_TIME"])
            landfalls[row["SID"]] = min(time, landfalls.get(row["SID"], time))
    return landfalls


def test_retained_landfalls():
    assert first_landfalls() == {
        "2024279N21265": at("2024-10-10T00:30"),
        "2024268N17278": at("2024-09-27T03:10"),
        "2025294N14290": at("2025-10-28T17:25"),
    }


@pytest.mark.parametrize(
    "identifier", ["hindcast_milton_2024", "hindcast_helene_2024", "hindcast_melissa_2025"]
)
def test_hindcast_inits_follow_the_landfall_rule(scenarios: dict[str, Scenario], identifier: str):
    scenario = scenarios[identifier]
    landfall = first_landfalls()[scenario.storm]
    expected: list[datetime] = []
    for days in (6, 4, 2):
        earlier = landfall - timedelta(days=days)
        expected.append(earlier.replace(hour=earlier.hour // 12 * 12, minute=0, second=0))
    assert list(scenario.init) == expected


@pytest.mark.parametrize(
    ("identifier", "centre", "radius_km"),
    [
        ("sandy_east_300km", [24.8, -75.9], 833.4),
        ("sandy_sweep", [24.8, -75.9], 833.4),
        ("katrina_scale_0p7", [24.6, -83.3], 400),
        ("milton_south_75km", [22.5, -93.4], 400),
    ],
)
def test_taper_matches_the_retained_row(
    scenarios: dict[str, Scenario], identifier: str, centre: list[float], radius_km: float
):
    scenario = scenarios[identifier]
    assert scenario.taper is not None
    assert list(scenario.taper.centre) == centre
    assert scenario.taper.radius_km == pytest.approx(radius_km)
    derived = taper_from_csv(IBTRACS, scenario.storm, scenario.init[0].isoformat())
    assert (derived.centre_lat, derived.centre_lon) == tuple(scenario.taper.centre)
    assert derived.radius_km == pytest.approx(scenario.taper.radius_km)


@pytest.mark.parametrize("identifier", ["milton_noise_1pct", "milton_gulf_minus2k"])
def test_only_relocation_and_scaling_have_a_taper(scenarios: dict[str, Scenario], identifier: str):
    assert scenarios[identifier].taper is None
    assert not any(s.taper for s in scenarios.values() if s.kind == "hindcast")


def test_naive_and_aware_inits_agree():
    aware = Scenario.model_validate({**valid_fork(), "init": "2005-08-27T02:00+02:00"})
    naive = Scenario.model_validate({**valid_fork(), "init": "2005-08-27T00:00:00"})
    assert aware.init == naive.init == (at("2005-08-27T00:00"),)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        ({"init": "2005-08-27T03:00Z"}, "six-hour"),
        ({"init": "2005-08-27T00:30Z"}, "six-hour"),
        ({"init": ["2005-08-27T00:00Z", "2005-08-27T00:00Z"]}, "distinct"),
        ({"init": ["2005-08-27T00:00Z", "2005-08-28T00:00Z"]}, "exactly one init"),
        ({"source": "gfs"}, "source"),
        ({"taper": None}, "need a taper"),
        ({"kind": "hindcast"}, "no edit"),
        ({"kind": "sweep"}, "sweep block"),
        ({"edit": {"type": "teleport", "params": {}}}, "needs an edit"),
        ({"edit": {"type": "scale_intensity", "params": {"east_km": 1}}}, "invalid"),
        ({"edit": {"type": "noise", "params": {"amplitude": 0.01, "seed": 0}}}, "need a taper"),
        ({"seeds": {"baseline": 0, "control": 1}}, "control seed"),
        ({"seeds": None}, "seeds"),
        ({"sweep": {"param": "factor", "values": [0.5, 1.0]}}, "sweep block"),
        ({"extra": 1}, "extra"),
        ({"id": "Bad Id"}, "id"),
    ],
)
def test_invalid_experiments(change: dict[str, Any], message: str):
    definition = valid_fork()
    for key, value in change.items():
        if value is None:
            del definition[key]
        else:
            definition[key] = value
    with pytest.raises(ValidationError, match=message):
        Scenario.model_validate(definition)


def test_sweep_cannot_fix_its_own_parameter():
    definition = {
        **valid_fork(),
        "kind": "sweep",
        "edit": {"type": "scale_intensity", "params": {"factor": 0.7}},
        "sweep": {"param": "factor", "values": [0.5, 1.0]},
    }
    with pytest.raises(ValidationError, match="invalid"):
        Scenario.model_validate(definition)


@pytest.mark.parametrize(
    ("sweep", "message"),
    [
        ({"param": "factor", "values": [0.5, 0.75]}, "must include 1, the null fork"),
        ({"param": "factor", "values": [0.5, 1.0, 0.5]}, "distinct"),
    ],
)
def test_a_sweep_has_distinct_values_and_its_null_fork(sweep: dict[str, Any], message: str):
    definition: dict[str, Any] = {
        **valid_fork(),
        "kind": "sweep",
        "edit": {"type": "scale_intensity", "params": {}},
        "sweep": sweep,
    }
    with pytest.raises(ValidationError, match=message):
        Scenario.model_validate(definition)


def test_a_sweep_over_a_parameter_without_a_null_value_is_refused():
    definition = {
        **valid_fork(),
        "kind": "sweep",
        "edit": {"type": "noise", "params": {"amplitude": 0.01}},
        "sweep": {"param": "seed", "values": [0, 1]},
    }
    del definition["taper"]
    with pytest.raises(ValidationError, match="seed has no null value"):
        Scenario.model_validate(definition)


def test_file_name_must_match_id(tmp_path: Path):
    (tmp_path / "other.yaml").write_text(yaml.safe_dump(valid_fork()))
    with pytest.raises(ValueError, match="must match the file name"):
        load_scenarios(tmp_path)
    (tmp_path / "other.yaml").rename(tmp_path / "example.yaml")
    assert [s.id for s in load_scenarios(tmp_path)] == ["example"]


def test_missing_directory_is_a_user_error(tmp_path: Path):
    with pytest.raises(UserError, match="no experiment files"):
        load_scenarios(tmp_path / "absent")
