"""Apply one explicitly configured edit without writing a forecast state."""

from argparse import Namespace

import numpy as np

from stormfork.cli_data import load_dataset
from stormfork.errors import UserError
from stormfork.grid import EARTH_RADIUS_KM
from stormfork.perturb import Taper, noise, relocate, scale_intensity, sst_anomaly, validate
from stormfork.perturb.validate import dry_run_statistics


def moved_centre(taper: Taper, east_km: float, north_km: float) -> tuple[float, float]:
    """Move a taper centre by a relocation, with the translation the public edit API uses."""
    latitude = taper.centre_lat + float(np.rad2deg(north_km / EARTH_RADIUS_KM))
    longitude = taper.centre_lon + float(
        np.rad2deg(east_km / (EARTH_RADIUS_KM * np.cos(np.deg2rad(taper.centre_lat))))
    )
    return latitude, longitude


def perturb(arguments: Namespace) -> str:
    """Validate one edit and report native-unit statistics without file writes."""
    original = load_dataset(arguments.dataset)
    taper = Taper(arguments.centre_lat, arguments.centre_lon, arguments.radius_km)
    moved_taper = None
    match arguments.operation:
        case "relocate":
            edited = relocate(
                original, taper, east_km=arguments.east_km, north_km=arguments.north_km
            )
            latitude, longitude = moved_centre(taper, arguments.east_km, arguments.north_km)
            moved_taper = Taper(latitude, longitude, taper.radius_km)
        case "scale_intensity":
            if arguments.factor is None:
                msg = "scale_intensity requires --factor"
                raise UserError(msg, "supply --factor in [0.5, 1.5]", 2)
            edited = scale_intensity(original, taper, factor=arguments.factor)
        case "sst_anomaly":
            if arguments.sigma_km is None or arguments.delta_k is None:
                msg = "sst_anomaly requires --sigma-km and --delta-k"
                raise UserError(
                    msg,
                    "supply a width in km and a temperature change in [-3, 3] K",
                    2,
                )
            edited = sst_anomaly(
                original,
                centre=(taper.centre_lat, taper.centre_lon),
                sigma_km=arguments.sigma_km,
                delta_k=arguments.delta_k,
            )
        case "noise":
            if arguments.amplitude is None or arguments.seed is None:
                msg = "noise requires --amplitude and --seed"
                raise UserError(
                    msg,
                    "supply an amplitude in [0, 0.05] and a nonnegative integer seed",
                    2,
                )
            edited = noise(original, amplitude=arguments.amplitude, seed=arguments.seed)
        case _:
            msg = f'unknown operation "{arguments.operation}"'
            raise UserError(msg, "use perturb --help", 2)
    validate(original, edited, moved_taper=moved_taper)
    return dry_run_statistics(original, edited, moved_taper or taper)
