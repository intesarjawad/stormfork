from collections.abc import Callable
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

STATION_IDS = (
    "KEYW",
    "KMIA",
    "KRSW",
    "KTPA",
    "KTLH",
    "KPNS",
    "KMOB",
    "KGPT",
    "KMSY",
    "KLCH",
    "KGLS",
    "KHOU",
    "KCRP",
    "KBRO",
    "KJAX",
    "KSAV",
    "KCHS",
    "KILM",
    "KHSE",
    "KORF",
    "KACY",
    "KJFK",
    "KBOS",
    "TJSJ",
)
IEM_NETWORKS = (
    "FL_ASOS",
    "AL_ASOS",
    "MS_ASOS",
    "LA_ASOS",
    "TX_ASOS",
    "GA_ASOS",
    "SC_ASOS",
    "NC_ASOS",
    "VA_ASOS",
    "NJ_ASOS",
    "NY_ASOS",
    "MA_ASOS",
    "PR_ASOS",
)


class Town(BaseModel):
    """A coastal ASOS station with WGS84 coordinates from IEM."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    icao: str = Field(pattern=r"^[A-Z]{4}$")
    name: str = Field(min_length=1)
    lat: float = Field(ge=-90, le=90, allow_inf_nan=False)
    lon: float = Field(ge=-180, le=180, allow_inf_nan=False)


class StationGeometry(BaseModel):
    """IEM GeoJSON point in longitude, latitude order."""

    coordinates: tuple[float, float]


class StationProperties(BaseModel):
    """The station's display name from IEM metadata."""

    sname: str


class StationFeature(BaseModel):
    """The station metadata used to construct the checked-in town set."""

    id: str
    properties: StationProperties
    geometry: StationGeometry


class StationNetwork(BaseModel):
    """IEM's GeoJSON station collection."""

    features: list[StationFeature]


def fetch_towns(request: Callable[[str], bytes]) -> tuple[Town, ...]:
    """Fetch the planned stations once from IEM's public network metadata."""
    stations: dict[str, Town] = {}
    for network in IEM_NETWORKS:
        url = f"https://mesonet.agron.iastate.edu/geojson/network/{network}.geojson"
        for feature in StationNetwork.model_validate_json(request(url)).features:
            icao = f"K{feature.id}" if network != "PR_ASOS" else feature.id
            if icao in STATION_IDS:
                lon, lat = feature.geometry.coordinates
                stations[icao] = Town(icao=icao, name=feature.properties.sname, lat=lat, lon=lon)
    missing = set(STATION_IDS) - stations.keys()
    if missing:
        msg = f"IEM metadata is missing planned stations: {sorted(missing)}"
        raise ValueError(msg)
    return tuple(stations[icao] for icao in STATION_IDS)


def load_towns(path: Path) -> tuple[Town, ...]:
    """Read the fixed scoreboard station set, rejecting omissions and duplicates."""
    towns = TypeAdapter(tuple[Town, ...]).validate_json(path.read_bytes())
    if tuple(town.icao for town in towns) != STATION_IDS:
        msg = "towns must contain the 24 planned ASOS stations in the documented order"
        raise ValueError(msg)
    return towns
