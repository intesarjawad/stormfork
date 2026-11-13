import csv
import io
import json

import numpy as np
import pandas as pd
import pytest
from pydantic import TypeAdapter, ValidationError

from stormfork.export.manifest import TrackPoint, track_exports
from stormfork.track.match import TRACK_COLUMNS
from tests.test_divergence import synthetic_tracks


def test_geojson_coordinate_order_wrap_and_full_round_trip():
    tracks = synthetic_tracks(members=2)
    tracks["lon"] = np.tile([359, 0, 181, -181, 540], 2)
    other = tracks.copy()
    other["track_id"] = "second"
    files = track_exports(
        {"baseline": pd.concat([other, tracks]).sample(frac=1, random_state=1), "fork": tracks}
    )
    adapter = TypeAdapter(list[TrackPoint])
    canonical = adapter.validate_json(files["tracks.json"])
    geojson = json.loads(files["tracks.geojson"])
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 6
    restored: list[TrackPoint] = []
    for feature in geojson["features"]:
        assert feature["geometry"]["type"] == "LineString"
        properties = feature["properties"]
        coordinates = feature["geometry"]["coordinates"]
        assert len(coordinates) == properties["sample_count"]
        assert [coordinate[0] for coordinate in coordinates] == [-1, 0, -179, 179, -180]
        assert all(-180 <= lon <= 180 and -90 <= lat <= 90 for lon, lat in coordinates)
        for index, (lon, lat) in enumerate(coordinates):
            restored.append(
                TrackPoint(
                    run=properties["run"],
                    member=properties["member"],
                    track_id=properties["track_id"],
                    step=properties["steps"][index],
                    valid_time=properties["valid_times"][index],
                    lat=lat,
                    lon=lon,
                    mslp_pa=properties["mslp_pa"][index],
                    wind_ms=properties["wind_ms"][index],
                )
            )
    assert restored == canonical
    first = canonical[0]
    assert first.lat == tracks.iloc[0].lat
    assert first.lon == -1


@pytest.mark.parametrize("track_id", ['a,"quoted" storm', 123])
def test_csv_round_trips_every_tracker_column_plus_run(track_id: str | int):
    tracks = synthetic_tracks(members=2)
    tracks["track_id"] = track_id
    tracks["valid_time"] += pd.Timedelta(microseconds=123456)
    tracks["lat"] += 0.123456789012345
    files = track_exports({"baseline": tracks})
    reader = csv.DictReader(io.StringIO(files["tracks.csv"].decode()))
    assert reader.fieldnames == [*TRACK_COLUMNS, "run"]
    adapter = TypeAdapter(list[TrackPoint])
    assert adapter.validate_python(list(reader)) == adapter.validate_json(files["tracks.json"])
    assert adapter.validate_json(files["tracks.json"])[0].valid_time == tracks.iloc[0].valid_time


def test_single_sample_retains_a_valid_linestring_without_inventing_a_sample():
    tracks = synthetic_tracks(members=1).iloc[:1]
    files = track_exports({"baseline": tracks})
    feature = json.loads(files["tracks.geojson"])["features"][0]
    assert feature["geometry"]["coordinates"] == [[-60, 20], [-60, 20]]
    assert feature["properties"]["sample_count"] == 1
    assert len(feature["properties"]["valid_times"]) == 1
    assert len(json.loads(files["tracks.json"])) == 1


def test_empty_track_formats_are_valid_and_deterministic():
    files = track_exports({"baseline": synthetic_tracks().iloc[:0]})
    assert json.loads(files["tracks.json"]) == []
    assert json.loads(files["tracks.geojson"])["features"] == []
    assert list(csv.DictReader(io.StringIO(files["tracks.csv"].decode()))) == []
    tracks = synthetic_tracks()
    assert track_exports({"baseline": tracks}) == track_exports(
        {"baseline": tracks.sample(frac=1, random_state=2)}
    )


def test_duplicate_or_invalid_samples_cannot_be_published():
    tracks = synthetic_tracks()
    with pytest.raises(ValueError, match="one sample"):
        track_exports({"baseline": pd.concat([tracks, tracks])})
    tracks.loc[0, "lat"] = 91
    with pytest.raises(ValidationError):
        track_exports({"baseline": tracks})
