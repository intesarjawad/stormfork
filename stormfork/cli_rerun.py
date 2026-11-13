"""Rerun an experiment, then compare it with the published run or export it.

The forecast stage is the only part that needs the model, so it sits behind
run_forecasts; the comparison and export reuse the verify and export commands.
Offline, a published id is compared only with a copy already in the checkout.
"""

import os
from argparse import Namespace
from pathlib import Path

import numpy as np

from stormfork import cli_verify
from stormfork.cli_data import load_dataset
from stormfork.cli_export import TRACKS_FILE, export_run_directory
from stormfork.cli_model import input_file, is_recipe, resolve_recipe
from stormfork.downloads import HRES_SAMPLE, IBTRACS, OFFLINE_ENV, WEIGHTS, is_offline
from stormfork.export.divergence import Divergence
from stormfork.run.forks import fork_recipes, moved_taper
from stormfork.scenarios import Scenario
from stormfork.track import track

DIVERGENCE_FILE = "divergence.json"


def run_forecasts(scenario: Scenario, arguments: Namespace, out: Path) -> None:
    """Write each run's forecast, atlases and tracks.csv, and the run's metadata.json.

    A fork, and each sweep stop, is tracked from the storm's relocated
    position when its edit moves it.

    Raises:
        UserError: An input file is neither passed nor available from the cache.
    """
    sample = input_file(arguments, HRES_SAMPLE)
    weights = input_file(arguments, WEIGHTS)
    storms = load_dataset(input_file(arguments, IBTRACS))
    from stormfork import cli_forecast  # noqa: PLC0415 - loads the model stack

    init = scenario.init[0]
    model_input = cli_forecast.initial_conditions(scenario, init, sample)
    runs = cli_forecast.forecast_run_directory(
        scenario, init, model_input, weights, storms, arguments, out
    )
    recipes = fork_recipes(scenario)
    for run in runs:
        moved = moved_taper(recipes[run]) if run in recipes else None
        position = (moved.centre_lat, moved.centre_lon) if moved else None
        tracks = track(
            load_dataset(out / run / cli_forecast.FORECAST_FILE),
            storms,
            np.datetime64(init),
            target_track_id=scenario.storm,
            moved_initial_position=position,
        )
        tracks.to_csv(out / run / TRACKS_FILE, index=False)


def rerun_target(arguments: Namespace) -> str:
    """Rerun a published id or a recipe file into the output directory.

    A published id is compared with its published run through verify
    --reproduction; a recipe file is only exported, since nothing exists to
    compare it with, and the fork's and control's separation at its last step
    is reported. Offline, without the published run in the checkout, the
    export is kept and the output says it was not compared and how to compare it.
    """
    scenario = resolve_recipe(arguments.target)
    out = arguments.out or Path(scenario.id)
    run_forecasts(scenario, arguments, out)
    exported = export_run_directory(out, out)
    if is_recipe(arguments.target):
        divergence = Divergence.model_validate_json((out / DIVERGENCE_FILE).read_bytes())
        last = max(divergence.steps, key=lambda row: row.step)
        return "\n".join([exported, *cli_verify.separation_lines(last)])
    published = cli_verify.PUBLISHED_DIR / scenario.id
    if not published.is_dir() and is_offline(os.environ):
        reason = f"{published} is absent and {OFFLINE_ENV} is set"
        command = f"stormfork verify {scenario.id} --reproduction {out} --by {arguments.by}"
        return "\n".join(
            [
                exported,
                f"not compared with the published {scenario.id}: {reason}",
                f"compare later: {command}",
            ]
        )
    return cli_verify.verify_target(scenario.id, reproduction=out, by=arguments.by)
