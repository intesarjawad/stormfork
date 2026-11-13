"""Describe a forecast run as the metadata.json that export reads.

Everything that needs the clock, the weights file or the device is passed in,
so the description of a run is fixed by its arguments.
"""

from datetime import datetime

from stormfork.downloads import HRES_SAMPLE
from stormfork.export.divergence import EvidenceModel, Nonnegative, Probability
from stormfork.export.manifest import (
    SCHEMA_VERSION,
    Diagnostics,
    ExperimentMetadata,
    InitSource,
    ModelInfo,
    PerturbationInfo,
    RuntimeInfo,
    StormInfo,
    Text,
    hindcast_init_id,
)
from stormfork.export.quantise import FieldMetadata
from stormfork.scenarios import Scenario
from stormfork.track import StormStart

NO_EDIT = PerturbationInfo(type="none", params={})
UNPUBLISHED_LIMIT = (
    "This recipe is not one of the published experiments, so no limits are stated for it."
)


class SweepStop(EvidenceModel):
    """One sweep value other than the null fork, as stops.json records it.

    run is the stop's directory in the run directory and param the swept edit
    parameter. A stop whose edited input failed the edit check is not rolled
    out: refused holds the check's message and the diagnostics stay null.
    """

    run: Text
    param: Text
    value: float
    adjustment_ratio: Nonnegative | None = None
    core_land_fraction: Probability | None = None
    refused: Text | None = None


def source_attribution(source: InitSource, year: int) -> list[str]:
    """Credit the reanalysis and forecast data an initial-condition source reads.

    ERA5 asks for the Copernicus sentence with the year of use; the hybrid also
    reads ECMWF's open data. Every source takes at least its terrain and land
    mask from the HRES sample, so every source credits ECMWF's HRES data.
    """
    copernicus = f"Contains modified Copernicus Climate Change Service information {year}."
    reanalysis = {
        "era5": [copernicus],
        "ifs_open_hybrid": [copernicus, "ECMWF open data, CC BY 4.0."],
    }.get(source, [])
    return [*reanalysis, HRES_SAMPLE.credit]


def experiment_metadata(  # noqa: PLR0913
    scenario: Scenario,
    *,
    init: datetime,
    model: ModelInfo,
    weights_license: str,
    runtime: RuntimeInfo,
    storm: StormStart,
    members: int,
    steps: int,
    fields: list[FieldMetadata],
    diagnostics: Diagnostics,
    created: datetime,
) -> ExperimentMetadata:
    """Describe one run of a fork, sweep or hindcast init from its recipe and how it was run.

    init is the one the run started from, and storm the target storm then. Each
    init of a hindcast is its own experiment, named by hindcast_init_id, and has
    no edit, recorded as type "none". A recipe without a title or limits takes
    its id and a sentence saying it is unpublished. diagnostics holds what the
    edit and the gridded first step give; export fills in what the tracks give.

    Raises:
        ValueError: The init is not one of the recipe's.
    """
    if init not in scenario.init:
        msg = f"{scenario.id} has no init {init.isoformat()}"
        raise ValueError(msg)
    lat, lon = storm.position or (None, None)
    hindcast = scenario.kind == "hindcast"
    return ExperimentMetadata.model_validate(
        {
            "schema_version": SCHEMA_VERSION,
            "id": hindcast_init_id(scenario.id, init) if hindcast else scenario.id,
            "parent_id": scenario.id if hindcast else None,
            "title": scenario.title or scenario.id,
            "kind": scenario.kind,
            "model": model,
            "runtime": runtime,
            "init": {
                "source": scenario.source,
                "time": init,
                "storm": StormInfo(
                    ibtracs_sid=scenario.storm,
                    lat=lat,
                    lon=lon,
                    first_record_time=storm.first_record_time,
                ),
            },
            "perturbation": scenario.edit or NO_EDIT,
            "seeds": scenario.seeds,
            "members": members,
            "steps": steps,
            "fields": fields,
            "diagnostics": diagnostics,
            "limits": scenario.limits or [UNPUBLISHED_LIMIT],
            "attribution": [
                " ".join(weights_license.split()),
                *source_attribution(scenario.source, created.year),
            ],
            "created": created,
        }
    )
