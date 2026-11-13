"""Plain-text entry points for local weather experiment operations."""

import argparse
import csv
import sys
import textwrap
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Never

import numpy as np

from stormfork.cli_data import load_dataset, parse_time, taper_from_csv
from stormfork.cli_export import export_release
from stormfork.cli_model import check_forecast_command
from stormfork.cli_perturb import perturb
from stormfork.cli_rerun import rerun_target
from stormfork.cli_verify import verify_target
from stormfork.downloads import CACHE_ENV, OFFLINE_ENV, fetch_all
from stormfork.errors import UserError
from stormfork.scenarios import EXPERIMENTS_DIR, list_lines, load_scenarios
from stormfork.scores.build import build_from_files, write_scoreboard
from stormfork.scores.fetch import (
    RunNotReadyError,
    backfill_plan,
    collect_backfill,
    collect_wn2,
    probe_wn2_previous_runs,
)
from stormfork.scores.inputs import fetch_inputs
from stormfork.scores.metrics import WINDOW_DAYS
from stormfork.scores.towns import load_towns
from stormfork.track import track
from stormfork.verify.benchmark import benchmark_case, resolve_case

OUTPUT_WIDTH = 80
CACHED = "(default: cached download)"


class HelpFormatter(argparse.HelpFormatter):
    """Keep help independent of terminal width and COLUMNS."""

    def __init__(self, prog: str) -> None:
        super().__init__(prog, width=OUTPUT_WIDTH, max_help_position=24)


class Parser(argparse.ArgumentParser):
    """Route argparse failures through the same contract as command failures."""

    def _get_formatter(self) -> argparse.HelpFormatter:
        return HelpFormatter(self.prog)

    def error(self, message: str) -> Never:
        """Report a usage failure with one command-specific next step."""
        raise UserError(message, f"run {self.prog} --help", 2)


def configure_perturb(edit: argparse.ArgumentParser) -> None:
    """Describe edit parameters in their native units."""
    edit.add_argument("dataset", type=Path, help="local NetCDF file or Zarr directory")
    edit.add_argument(
        "--dry-run", action="store_true", required=True, help="compute without writing"
    )
    edit.add_argument(
        "--operation",
        required=True,
        choices=("relocate", "scale_intensity", "sst_anomaly", "noise"),
        metavar="EDIT",
        help="relocate, scale_intensity, sst_anomaly, or noise",
    )
    edit.add_argument("--centre-lat", type=float, required=True, help="taper latitude in degrees")
    edit.add_argument("--centre-lon", type=float, required=True, help="taper longitude in degrees")
    edit.add_argument("--radius-km", type=float, required=True, help="taper radius in km")
    edit.add_argument("--east-km", type=float, default=0, help="relocation east in km (default: 0)")
    edit.add_argument(
        "--north-km", type=float, default=0, help="relocation north in km (default: 0)"
    )
    edit.add_argument("--factor", type=float, help="intensity multiplier in [0.5, 1.5]")
    edit.add_argument("--sigma-km", type=float, help="SST Gaussian width in km")
    edit.add_argument("--delta-k", type=float, help="SST peak change in [-3, 3] K")
    edit.add_argument("--amplitude", type=float, help="noise amplitude in [0, 0.05]")
    edit.add_argument("--seed", type=int, help="nonnegative noise seed")


def configure_forecast_commands(add_parser: Callable[..., argparse.ArgumentParser]) -> None:
    """Describe the commands that run the model and publish its output."""
    run = add_parser("run", help="run baseline, fork and control forecasts")
    run.add_argument("id", help="experiment id")
    run.add_argument("--runtime", choices=("tpu", "gpu", "cpu"), default="cpu")
    run.add_argument("--out", type=Path, help="run directory with input.nc (default: ./<id>/)")
    run.add_argument("--weights", type=Path, help=f"local model weights {CACHED}")
    run.add_argument("--ibtracs", type=Path, help=f"local IBTrACS NetCDF {CACHED}")
    run.add_argument("--members", type=int, default=8, help="ensemble members (default: 8)")
    run.add_argument("--steps", type=int, default=40, help="six-hour steps (default: 40)")

    initial = add_parser("ic", help="initial conditions")
    build = initial.add_subparsers(dest="ic_command", required=True).add_parser(
        "build", help="build the model input for one init time"
    )
    build.add_argument("id", help="experiment id")
    build.add_argument("--init", required=True, help="ISO init time (UTC)")
    build.add_argument("--out", type=Path, required=True, help="destination input file")
    build.add_argument("--sample", type=Path, help=f"local HRES sample {CACHED}")

    export = add_parser(
        "export", help="write manifest, tracks, divergence and dist/atlases/<id>.zip"
    )
    export.add_argument("run_dir", type=Path, help="run directory with metadata and tracks")
    export.add_argument("--out", type=Path, required=True, help="destination directory")
    export.add_argument("--ibtracs", type=Path, help="local IBTrACS CSV (a hindcast needs it)")

    rerun = add_parser("rerun", help="reproduce a published experiment")
    rerun.add_argument("target", help="experiment id or recipe.yaml")
    rerun.add_argument("--members", type=int, default=1, help="ensemble members (default: 1)")
    rerun.add_argument("--steps", type=int, default=8, help="six-hour steps (default: 8)")
    rerun.add_argument("--runtime", choices=("cpu", "gpu", "tpu"), default="cpu")
    rerun.add_argument("--out", type=Path, help="destination directory (default: ./<id>/)")
    rerun.add_argument("--sample", type=Path, help=f"local HRES sample {CACHED}")
    rerun.add_argument("--weights", type=Path, help=f"local model weights {CACHED}")
    rerun.add_argument("--ibtracs", type=Path, help=f"local IBTrACS NetCDF {CACHED}")
    rerun.add_argument(
        "--by", help="GitHub user running it; required for a published id, optional for a recipe"
    )


def configure_scores(add_parser: Callable[..., argparse.ArgumentParser]) -> None:
    """Describe the public score collection commands."""
    collect = add_parser("collect", help="collect the newest complete WN2 run")
    collect.add_argument("--output", type=Path, required=True)
    collect.add_argument("--towns", type=Path, default=Path("data/towns.json"))
    fetch = add_parser("fetch", help="fetch observations and IFS forecasts")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--scoring-day", help="ISO date (default: yesterday UTC)")
    fetch.add_argument("--towns", type=Path, default=Path("data/towns.json"))
    backfill = add_parser("backfill", help="collect the next Previous Runs and observation chunk")
    backfill.add_argument("--output", type=Path, required=True)
    backfill.add_argument("--since", required=True, help="first ISO date to archive")
    backfill.add_argument("--until", help="last ISO date (default: today UTC)")
    backfill.add_argument("--towns", type=Path, default=Path("data/towns.json"))
    probe = add_parser("probe", help="report whether WN2 Previous Runs holds this week's values")
    probe.add_argument("--towns", type=Path, default=Path("data/towns.json"))
    build = add_parser("build", help="rebuild scores.json from local inputs")
    build.add_argument("--wn2", type=Path, help="stored WN2 parquet directory")
    build.add_argument("--observations", type=Path, help="observation CSV")
    build.add_argument("--ifs", type=Path, help="IFS forecast CSV")
    build.add_argument("--output", type=Path, help="destination scores.json")
    build.add_argument("--scoring-day", help="ISO date (default: yesterday UTC)")
    build.add_argument("--lead-days", default="0", help="comma-separated lead days (default: 0)")
    build.add_argument("--since", help="ISO date to plan a Previous Runs backfill")
    build.add_argument("--until", help="last ISO date of a backfill plan (default: today UTC)")
    build.add_argument("--dry-run", action="store_true", help="print the backfill plan only")
    build.add_argument("--towns", type=Path, default=Path("data/towns.json"))
    build.add_argument(
        "--wn2-previous-runs-available",
        choices=("true", "false"),
        help="record the weekly WN2 Previous Runs probe",
    )


def build_parser() -> argparse.ArgumentParser:
    """Register every command."""
    parser = Parser(prog="stormfork", description="Local weather experiment tools")
    parser.add_argument("--debug", action="store_true", help="show full failures and tracebacks")
    commands = parser.add_subparsers(dest="command", required=True, metavar="COMMAND")
    scenarios = commands.add_parser("scenarios", help="storm taper placement")
    scenario_commands = scenarios.add_subparsers(dest="scenario_command", required=True)
    listing = scenario_commands.add_parser("list", help="list published experiments by init time")
    listing.add_argument(
        "--dir", type=Path, default=EXPERIMENTS_DIR, help="experiment files (default: experiments)"
    )
    taper = scenario_commands.add_parser("taper", help="print a taper block from an IBTrACS row")
    taper.add_argument("id", help="IBTrACS storm SID")
    taper.add_argument("--ibtracs", type=Path, required=True, help="local IBTrACS CSV")
    taper.add_argument("--init", required=True, help="ISO init time (UTC)")

    configure_perturb(commands.add_parser("perturb", help="validate an edit and print statistics"))

    tracker = commands.add_parser("track", help="track storms in a supplied dataset")
    tracker.add_argument("dataset", type=Path, help="local NetCDF file or Zarr directory")
    tracker.add_argument("--ibtracs", type=Path, required=True, help="local IBTrACS NetCDF or Zarr")
    tracker.add_argument("--init", required=True, help="ISO init time (UTC)")
    tracker.add_argument("--out", type=Path, required=True, help="destination track CSV")
    tracker.add_argument(
        "--target-track-id", help="IBTrACS SID of the storm to keep (and to move for relocation)"
    )
    tracker.add_argument("--moved-lat", type=float, help="relocated latitude in degrees")
    tracker.add_argument("--moved-lon", type=float, help="relocated longitude in degrees")

    verify = commands.add_parser("verify", help="check a published experiment's evidence")
    verify.add_argument(
        "target",
        nargs="?",
        help="experiment id, manifest.json or directory (default: the hero)",
    )
    verify.add_argument(
        "--reproduction", type=Path, help="exported run directory to compare with the published run"
    )
    verify.add_argument("--by", help="GitHub user who reproduced the run (with --reproduction)")

    configure_forecast_commands(commands.add_parser)
    commands.add_parser(
        "fetch",
        help="download the weights, HRES sample and IBTrACS once",
        description=(
            f"Downloads go to ${CACHE_ENV}, else $XDG_CACHE_HOME/stormfork or "
            f"~/.cache/stormfork. With {OFFLINE_ENV}=1, every command uses only the cache."
        ),
    )

    benchmark = commands.add_parser("benchmark", help="compare a hindcast with NHC aids")
    benchmark.add_argument("case", help="hindcast case directory or id")
    benchmark.add_argument("--out", type=Path, help="destination benchmark JSON")

    scores = commands.add_parser("scores", help="public forecast collection")
    configure_scores(scores.add_subparsers(dest="scores_command", required=True).add_parser)
    return parser


def scenario_output(arguments: argparse.Namespace) -> str:
    """List the published experiments or print a taper block."""
    if arguments.scenario_command == "list":
        return list_lines(load_scenarios(arguments.dir))
    taper = taper_from_csv(arguments.ibtracs, arguments.id, arguments.init)
    return (
        f"taper:\n  centre_lat: {taper.centre_lat}\n"
        f"  centre_lon: {taper.centre_lon}\n  radius_km: {taper.radius_km}"
    )


def track_output(arguments: argparse.Namespace) -> str:
    """Track storms in a supplied dataset and write the track table."""
    position = None
    if (arguments.moved_lat is None) != (arguments.moved_lon is None):
        msg = "--moved-lat and --moved-lon must be supplied together"
        raise UserError(
            msg,
            "supply both relocation coordinates or omit both",
            2,
        )
    if arguments.moved_lat is not None:
        position = (arguments.moved_lat, arguments.moved_lon)
    tracks = track(
        load_dataset(arguments.dataset),
        load_dataset(arguments.ibtracs),
        np.datetime64(parse_time(arguments.init)),
        target_track_id=arguments.target_track_id,
        moved_initial_position=position,
    )
    tracks.to_csv(arguments.out, index=False)
    return f"{len(tracks)} track samples written to {arguments.out}"


def verify_output(arguments: argparse.Namespace) -> str:
    """Verify a published experiment, or compare a rerun with it."""
    return verify_target(arguments.target, reproduction=arguments.reproduction, by=arguments.by)


def export_output(arguments: argparse.Namespace) -> str:
    """Write the evidence files for a run directory and its release zip."""
    return export_release(arguments.run_dir, arguments.out, arguments.ibtracs)


def scoring_day(value: str | None) -> datetime:
    """Use the supplied date, or yesterday UTC when the workflow omits it."""
    if value is None:
        today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
        return today - timedelta(days=1)
    return datetime.fromisoformat(value).replace(tzinfo=UTC)


def parse_lead_days(value: str) -> list[int]:
    """Read the comma-separated lead days the board should score."""
    return [int(part) for part in value.split(",") if part]


def fetch_output(arguments: argparse.Namespace) -> str:
    """Fetch the observations and IFS forecasts for the scoring window."""
    day = scoring_day(arguments.scoring_day)
    observations, _ifs = fetch_inputs(
        load_towns(arguments.towns),
        start=day - timedelta(days=WINDOW_DAYS),
        end=day,
        output=arguments.output,
    )
    return f"score inputs written to {observations.parent}"


def archive_bounds(arguments: argparse.Namespace) -> tuple[datetime, datetime]:
    """Use midnight UTC for both ends so chunk files land on whole days."""
    since = scoring_day(arguments.since)
    if arguments.until is not None:
        return since, scoring_day(arguments.until)
    today = datetime.now(UTC).replace(hour=0, minute=0, second=0, microsecond=0)
    return since, today


def backfill_plan_output(arguments: argparse.Namespace) -> str:
    """Print the weighted calls each day of the archive plan would spend."""
    since, until = archive_bounds(arguments)
    days = backfill_plan(load_towns(arguments.towns), since=since, until=until)
    return "\n".join(f"{day.start} to {day.end}: {day.calls:.1f} weighted calls" for day in days)


def backfill_output(arguments: argparse.Namespace) -> str:
    """Collect the next chunk that is not on disk yet."""
    since, until = archive_bounds(arguments)
    day = collect_backfill(load_towns(arguments.towns), arguments.output, since=since, until=until)
    if day is None:
        return "no Previous Runs chunk is waiting"
    return f"{day.start} to {day.end}: {day.calls:.1f} weighted calls written to {arguments.output}"


def probe_output(arguments: argparse.Namespace) -> str:
    """Print whether WN2's Previous Runs archive holds a value for the last week."""
    available = probe_wn2_previous_runs(load_towns(arguments.towns)[0], now=datetime.now(UTC))
    return "true" if available else "false"


def build_output(arguments: argparse.Namespace) -> str:
    """Rebuild the scoreboard document from the three local inputs, or plan a backfill."""
    if arguments.since is not None:
        if not arguments.dry_run:
            msg = "--since only plans a backfill and needs --dry-run"
            raise UserError(msg, "add --dry-run to print the plan", 2)
        return backfill_plan_output(arguments)
    missing = [
        name
        for name in ("wn2", "observations", "ifs", "output")
        if getattr(arguments, name) is None
    ]
    if missing:
        listed = ", ".join(f"--{name}" for name in missing)
        msg = f"the following arguments are required: {listed}"
        raise UserError(msg, "run stormfork scores build --help", 2)
    available = (
        None
        if arguments.wn2_previous_runs_available is None
        else arguments.wn2_previous_runs_available == "true"
    )
    board = build_from_files(
        wn2=arguments.wn2,
        observations=arguments.observations,
        ifs=arguments.ifs,
        scoring_day=scoring_day(arguments.scoring_day),
        lead_days=parse_lead_days(arguments.lead_days),
        wn2_previous_runs_available=available,
    )
    write_scoreboard(board, arguments.output)
    return f"scoreboard written to {arguments.output}"


def scores_output(arguments: argparse.Namespace) -> str:
    """Run the selected public collection command."""
    if arguments.scores_command == "collect":
        try:
            return str(
                collect_wn2(load_towns(arguments.towns), arguments.output, now=datetime.now(UTC))
            )
        except RunNotReadyError as error:
            raise UserError(str(error), error.next_step, 3) from error
    if arguments.scores_command == "fetch":
        return fetch_output(arguments)
    if arguments.scores_command == "backfill":
        return backfill_output(arguments)
    if arguments.scores_command == "probe":
        return probe_output(arguments)
    return build_output(arguments)


def dispatch(arguments: argparse.Namespace) -> str:
    """Dispatch local operations without loading forecast model dependencies."""
    local = {
        "scenarios": scenario_output,
        "perturb": perturb,
        "track": track_output,
        "verify": verify_output,
        "export": export_output,
    }.get(arguments.command)
    if local is not None:
        return local(arguments)
    if arguments.command == "fetch":
        return fetch_all()
    check_forecast_command(arguments)
    if arguments.command == "rerun":
        return rerun_target(arguments)
    if arguments.command == "scores":
        return scores_output(arguments)
    from stormfork import cli_forecast  # noqa: PLC0415 - loads the model stack

    if arguments.command == "run":
        return cli_forecast.run_output(arguments)
    return cli_forecast.ic_build_output(arguments)


def error_record(prefix: str, message: str) -> str:
    """Keep two physical error lines even for unbounded paths and schema errors.

    I abbreviate the middle of oversized records to retain their beginning and
    offending value at the end. --debug retains the complete exception. Full
    arbitrary values, exactly two lines and an 80-column limit cannot coexist.
    """
    record = f"{prefix}: {' '.join(message.split())}"
    if len(record) <= OUTPUT_WIDTH:
        return record
    half = (OUTPUT_WIDTH - len(" ... ")) // 2
    return f"{record[:half]} ... {record[-(OUTPUT_WIDTH - half - len(' ... ')) :]}"


def main() -> None:
    """Run a local operation or report one actionable failure without a traceback."""
    try:
        arguments = build_parser().parse_args()
        if arguments.command == "benchmark":
            destination = benchmark_case(resolve_case(arguments.case), arguments.out)
            output = f"benchmark written to {destination}"
        else:
            output = dispatch(arguments)
    except (UserError, ValueError, TypeError, KeyError, OSError, ImportError, csv.Error) as error:
        if "--debug" in sys.argv[1:]:
            raise
        if isinstance(error, UserError):
            failure = error
        elif isinstance(error, ImportError):
            failure = UserError(str(error), "run uv sync --group model and retry", 2)
        else:
            failure = UserError(str(error), "check the reported data or path and retry", 1)
        print(error_record("error", failure.what), file=sys.stderr)
        print(error_record("next", failure.next_step), file=sys.stderr)
        raise SystemExit(failure.exit_code) from None
    for line in output.splitlines():
        # A JSON record must stay on one line to be pasted into a ledger.
        print(
            line
            if line.startswith("{")
            else textwrap.fill(line, width=OUTPUT_WIDTH, replace_whitespace=False)
        )
