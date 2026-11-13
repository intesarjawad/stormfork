"""Commands that run the forecast model and the checks that precede them.

The model stack is imported only when a command needs it, so every other
command works without the model dependency group.
"""

from argparse import Namespace
from importlib.util import find_spec
from pathlib import Path

import yaml

from stormfork.cli_data import parse_time
from stormfork.downloads import Source, resolve
from stormfork.errors import UserError
from stormfork.export.atlas import FRAME_COUNT
from stormfork.scenarios import EXPERIMENTS_DIR, Scenario, load_scenarios

MODEL_PACKAGE = "weathernext"
RECIPE_SUFFIXES = (".yaml", ".yml")
MAX_STEPS = FRAME_COUNT - 1


def require_model_stack(command: str) -> None:
    """Fail with the install step when the model dependency group is absent.

    Raises:
        UserError: The model package cannot be found.
    """
    if find_spec(MODEL_PACKAGE) is None:
        msg = f"{command} needs the model stack ({MODEL_PACKAGE}), which is not installed"
        raise UserError(msg, "run uv sync --group model and retry", 2)


def known_scenario(experiment: str, directory: Path = EXPERIMENTS_DIR) -> Scenario:
    """Find a published experiment by id.

    Raises:
        UserError: No experiment has this id.
    """
    for scenario in load_scenarios(directory):
        if scenario.id == experiment:
            return scenario
    msg = f'unknown experiment "{experiment}"'
    raise UserError(msg, "pick an id from stormfork scenarios list", 2)


def is_recipe(target: str) -> bool:
    """Tell a recipe file from a published experiment id."""
    path = Path(target)
    return path.suffix in RECIPE_SUFFIXES or path.is_file()


def resolve_recipe(target: str, directory: Path = EXPERIMENTS_DIR) -> Scenario:
    """Read a recipe file, or find a published experiment by id.

    Raises:
        UserError: The file is not YAML or no experiment has this id.
        ValueError: The recipe does not describe a runnable experiment.
    """
    if not is_recipe(target):
        return known_scenario(target, directory)
    path = Path(target)
    try:
        document = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as error:
        msg = f'"{path}" is not valid YAML ({error})'
        raise UserError(msg, "fix the recipe file and retry", 2) from error
    return Scenario.model_validate(document)


def input_file(arguments: Namespace, source: Source) -> Path:
    """Return the local file passed for an input, else its cached copy, downloading it if absent.

    Raises:
        UserError: The flag names no file, or the input is neither cached nor downloadable.
    """
    path: Path | None = getattr(arguments, source.flag)
    if path is None:
        return resolve(source)
    if not path.is_file():
        msg = f'{source.title} file "{path}" does not exist'
        raise UserError(msg, f"pass an existing file to --{source.flag}", 2)
    return path


def check_fork(scenario: Scenario, command: str) -> None:
    """Refuse the kinds whose run directory has no baseline, fork and control.

    Raises:
        UserError: The experiment is a hindcast or a sweep.
    """
    if scenario.kind != "fork":
        msg = f"{scenario.id} is a {scenario.kind}, and {command} rolls out only forks"
        raise UserError(msg, "pick a fork from stormfork scenarios list", 2)


def check_counts(arguments: Namespace) -> None:
    """Require at least one member and between one step and as many as an atlas holds.

    Raises:
        UserError: A count is not positive, or there are more steps than atlas tiles.
    """
    if arguments.members < 1 or arguments.steps < 1:
        msg = "--members and --steps must be at least 1"
        raise UserError(msg, "supply positive counts", 2)
    if arguments.steps > MAX_STEPS:
        msg = f"an atlas holds the analysis and at most {MAX_STEPS} six-hour steps"
        raise UserError(msg, f"pass --steps between 1 and {MAX_STEPS}", 2)


def check_run(arguments: Namespace) -> None:
    """Check the model stack, the experiment and the counts before a forecast run.

    Raises:
        UserError: The experiment is unknown, or a count is out of range.
    """
    require_model_stack("run")
    known_scenario(arguments.id)
    check_counts(arguments)


def check_ic_build(arguments: Namespace) -> None:
    """Check the model stack, the experiment id and its init times before building.

    Raises:
        UserError: The init time is not one of the experiment's inits.
    """
    require_model_stack("ic build")
    scenario = known_scenario(arguments.id)
    if parse_time(arguments.init) not in scenario.init:
        msg = f"{scenario.id} has no init {arguments.init}"
        raise UserError(msg, "use an init time listed by stormfork scenarios list", 2)


def check_rerun(arguments: Namespace) -> None:
    """Check the model stack, the recipe, its initial-condition source and the reproducer.

    Raises:
        UserError: The source is not hres_sample, the experiment is not a fork, a
            count is not positive or a published experiment is rerun without --by.
    """
    require_model_stack("rerun")
    scenario = resolve_recipe(arguments.target)
    if scenario.source != "hres_sample":
        msg = f"{scenario.id} starts from {scenario.source}, which rerun cannot fetch"
        raise UserError(msg, "open the rerun notebook for this recipe", 2)
    check_fork(scenario, "rerun")
    check_counts(arguments)
    if arguments.by is None and not is_recipe(arguments.target):
        msg = "rerunning a published experiment needs --by"
        raise UserError(msg, "pass your GitHub login to --by", 2)


def check_forecast_command(arguments: Namespace) -> None:
    """Run the checks for whichever forecast command was requested, if any."""
    check = {"run": check_run, "ic": check_ic_build, "rerun": check_rerun}.get(arguments.command)
    if check is not None:
        check(arguments)
