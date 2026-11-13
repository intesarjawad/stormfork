from datetime import UTC, datetime
from pathlib import Path

import pytest

from stormfork.downloads import HRES_SAMPLE
from stormfork.export.manifest import Diagnostics, ModelInfo, RuntimeInfo
from stormfork.export.metadata import UNPUBLISHED_LIMIT, experiment_metadata
from stormfork.export.quantise import field_metadata
from stormfork.scenarios import Scenario, load_scenarios
from stormfork.track import StormStart

EXPERIMENTS = Path(__file__).parent.parent / "experiments"
MODEL = ModelInfo(name="fixture", checkpoint="fixture.npz", md5="0" * 32, code_version="v")
RUNTIME = RuntimeInfo(kind="cpu", device="cpu")
CREATED = datetime(2026, 11, 9, 12, tzinfo=UTC)
MILTON_START = StormStart((22.5, -93.4), datetime.fromisoformat("2024-10-05T12:00"))


def published(experiment: str) -> Scenario:
    return next(item for item in load_scenarios(EXPERIMENTS) if item.id == experiment)


def describe(
    scenario: Scenario,
    weights_license: str = "Weights\n  CC BY 4.0.",
    *,
    init: datetime | None = None,
    storm: StormStart = MILTON_START,
) -> dict[str, object]:
    return experiment_metadata(
        scenario,
        init=init or scenario.init[0],
        model=MODEL,
        weights_license=weights_license,
        runtime=RUNTIME,
        storm=storm,
        members=1,
        steps=8,
        fields=field_metadata(),
        diagnostics=Diagnostics(adjustment_ratio=1.2, bounds_check="passed"),
        created=CREATED,
    ).model_dump(mode="json")


def test_a_published_fork_carries_its_recipe_and_how_it_ran():
    scenario = published("milton_south_75km")
    metadata = describe(scenario)
    assert metadata["schema_version"] == 2
    assert metadata["id"] == "milton_south_75km"
    assert metadata["parent_id"] is None
    assert metadata["title"] == scenario.title
    assert metadata["limits"] == scenario.limits
    assert metadata["init"] == {
        "source": "hres_sample",
        "time": "2024-10-07T00:00:00",
        "storm": {
            "ibtracs_sid": "2024279N21265",
            "lat": 22.5,
            "lon": -93.4,
            "first_record_time": "2024-10-05T12:00:00",
        },
    }
    assert metadata["perturbation"] == {
        "type": "relocate",
        "params": {"east_km": 0, "north_km": -75},
    }
    assert metadata["seeds"] == {"baseline": 0, "control": 1000}
    assert (metadata["members"], metadata["steps"]) == (1, 8)
    assert metadata["attribution"] == ["Weights CC BY 4.0.", HRES_SAMPLE.credit]
    assert metadata["diagnostics"] == {
        "adjustment_ratio": 1.2,
        "core_land_fraction": None,
        "bounds_check": "passed",
        "intensity_drift_24h_hpa": None,
        "null_fork_max_track_diff_km": None,
    }


def test_each_hindcast_init_is_its_own_experiment_and_may_precede_genesis():
    scenario = published("hindcast_milton_2024")
    before_genesis = StormStart(None, datetime.fromisoformat("2024-10-05T12:00"))
    metadata = describe(scenario, init=scenario.init[0], storm=before_genesis)
    assert metadata["id"] == "hindcast_milton_2024-2024100400"
    assert metadata["parent_id"] == "hindcast_milton_2024"
    assert metadata["init"] == {
        "source": "era5",
        "time": "2024-10-04T00:00:00",
        "storm": {
            "ibtracs_sid": "2024279N21265",
            "lat": None,
            "lon": None,
            "first_record_time": "2024-10-05T12:00:00",
        },
    }
    assert describe(scenario, init=scenario.init[2])["id"] == "hindcast_milton_2024-2024100800"


def test_a_recipe_without_title_or_limits_says_it_is_unpublished():
    scenario = published("milton_noise_1pct").model_copy(update={"title": None, "limits": []})
    metadata = describe(scenario)
    assert metadata["title"] == "milton_noise_1pct"
    assert metadata["limits"] == [UNPUBLISHED_LIMIT]


@pytest.mark.parametrize(
    ("source", "sources"),
    [
        ("era5", ["Contains modified Copernicus Climate Change Service information 2026."]),
        (
            "ifs_open_hybrid",
            [
                "Contains modified Copernicus Climate Change Service information 2026.",
                "ECMWF open data, CC BY 4.0.",
            ],
        ),
    ],
)
def test_reanalysis_sources_are_credited_between_the_weights_and_the_sample(
    source: str, sources: list[str]
):
    scenario = published("milton_noise_1pct").model_copy(update={"source": source})
    assert describe(scenario)["attribution"] == ["Weights CC BY 4.0.", *sources, HRES_SAMPLE.credit]
