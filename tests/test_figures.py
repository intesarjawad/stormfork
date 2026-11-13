import math
from pathlib import Path

import numpy as np
import pytest

from scripts.figures import load_published, main, relocation_section, taper_profile
from tests.test_manifest import example_export

RADIUS_KM = 900.0


def cosine_weight(distance_km: float) -> float:
    transition = min(max(2 * distance_km / RADIUS_KM - 1, 0), 1)
    return (1 + math.cos(math.pi * transition)) / 2


def test_taper_profile_is_flat_inside_half_the_radius_and_zero_at_the_edge():
    distance, weight = taper_profile(RADIUS_KM)
    assert distance[0] == 0
    assert np.all(weight[distance <= RADIUS_KM / 2] == 1)
    assert np.all(weight[distance >= RADIUS_KM] == 0)
    for point in (600.0, 675.0, 800.0):
        nearest = int(np.argmin(np.abs(distance - point)))
        assert weight[nearest] == pytest.approx(cosine_weight(distance[nearest]), abs=1e-6)


def test_relocation_section_moves_the_low_east_by_the_requested_distance():
    lon, before, after = relocation_section()
    earth_radius_km = 6371.0088
    expected_shift = math.degrees(300.0 / (earth_radius_km * math.cos(math.radians(25.0))))
    assert lon[np.argmin(before)] == pytest.approx(280.0, abs=0.25)
    assert lon[np.argmin(after)] - lon[np.argmin(before)] == pytest.approx(expected_shift, abs=0.25)


def test_without_published_experiments_two_figures_are_written_and_two_skipped(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    assert main(["--out", str(tmp_path / "out"), "--experiments", str(tmp_path / "none")]) == 0
    assert sorted(path.name for path in (tmp_path / "out").iterdir()) == [
        "relocation_section.png",
        "taper_profile.png",
    ]
    error = capsys.readouterr().err
    assert "adjustment_ratio.png skipped" in error
    assert "fork_control.png skipped" in error


def test_a_published_experiment_adds_the_other_two_figures(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
):
    experiment = tmp_path / "experiments" / "synthetic_comparison"
    experiment.mkdir(parents=True)
    example_export(experiment, adjustment_ratio=1.25)
    (tmp_path / "experiments" / "partial").mkdir()
    (tmp_path / "experiments" / "partial" / "manifest.json").write_text('{"id": "partial"}')
    out = tmp_path / "out"
    assert main(["--out", str(out), "--experiments", str(tmp_path / "experiments")]) == 0
    assert {path.name for path in out.iterdir()} >= {"adjustment_ratio.png", "fork_control.png"}
    assert "skipped partial" in capsys.readouterr().err
    assert [manifest.id for manifest, _ in load_published(tmp_path / "experiments")] == [
        "synthetic_comparison"
    ]
