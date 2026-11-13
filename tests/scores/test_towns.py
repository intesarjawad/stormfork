from pathlib import Path

from stormfork.scores.towns import STATION_IDS, load_towns


def test_checked_in_towns_match_planned_station_set():
    towns = load_towns(Path("data/towns.json"))
    assert len(towns) == len({town.icao for town in towns}) == 24
    assert tuple(town.icao for town in towns) == STATION_IDS
    miami = next(town for town in towns if town.icao == "KMIA")
    assert miami.lat == 25.788
    assert miami.lon == -80.3169
