import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from stormfork.grid import LAT, LEVELS_HPA, LON
from stormfork.ic.era5 import ATMOSPHERE, SURFACE
from stormfork.ic.ifs_open import (
    METHODS,
    Method,
    best_coarsening,
    build_hybrid,
    build_with_method,
    coarsening_line,
    coarsening_score,
    decision_line,
    fetch_ifs_store,
    hybrid_is_source,
    hybrid_store,
    ifs_url,
    rms_difference,
    sample_std,
    select_ranges,
    to_store,
)
from stormfork.ic.layout import SampleStatics, load_sample_statics

INIT = datetime(2024, 10, 7, tzinfo=UTC)
SAMPLE_PATH = Path("data/model/hres_2024-10-07_01.nc")
ARCO_STORE = "gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3"
LEVELS = (1000, 925, 850, 700, 600, 500, 400, 300, 250, 200, 150, 100, 50)
LAT_45 = 135  # Index of 45 degrees on the ascending one-degree grid.
PRESSURE_SHORT = ("t", "gh", "u", "v", "w", "q")
SURFACE_SHORT = ("2t", "msl", "10u", "10v")


def index_line(param: str, offset: int, level: int | None = None) -> str:
    entry: dict[str, str | int] = {
        "domain": "g",
        "stream": "oper",
        "levtype": "sfc" if level is None else "pl",
        "step": "0",
        "param": param,
        "_offset": offset,
        "_length": 10,
    }
    if level is not None:
        entry["levelist"] = str(level)
    return json.dumps(entry)


def index_of(*, drop: str | None = None) -> str:
    wanted = [(short, level) for short in PRESSURE_SHORT for level in LEVELS]
    wanted += [(short, None) for short in SURFACE_SHORT]
    wanted = [item for item in wanted if item[0] != drop]
    lines = [index_line("r", 0, 500)]
    lines += [
        index_line(short, 10 + 10 * position, level)
        for position, (short, level) in enumerate(wanted[:40])
    ]
    lines.append(index_line("d", 410, 850))
    lines += [
        index_line(short, 420 + 10 * position, level)
        for position, (short, level) in enumerate(wanted[40:])
    ]
    return "\n".join(lines) + "\n"


@pytest.fixture(scope="module")
def statics() -> SampleStatics:
    coords = {"lat": LAT.astype(np.float32), "lon": LON.astype(np.float32)}
    return SampleStatics(
        xr.DataArray(np.full((181, 360), 12, dtype=np.float32), dims=("lat", "lon"), coords=coords),
        xr.DataArray(
            np.full((181, 360), 0.5, dtype=np.float32), dims=("lat", "lon"), coords=coords
        ),
    )


def quarter_store(names: tuple[str, ...], offset: float = 0) -> xr.Dataset:
    """Fields equal to variable index + 100 per frame + level / 1000 + latitude squared."""
    lat = np.arange(90, -90.25, -0.25)
    levels = np.array(LEVELS_HPA[::-1])
    frames = np.arange(2)[:, None, None] * 100 + lat[None, None, :] ** 2
    fields: dict[str, xr.DataArray] = {}
    for index, name in enumerate(names):
        if name in ATMOSPHERE:
            values = index + offset + frames + levels[None, :, None] / 1000
            dims = ("time", "level", "latitude", "longitude")
        else:
            values = index + offset + frames[:, 0, :]
            dims = ("time", "latitude", "longitude")
        shape = (*values.shape, 1440)
        fields[name] = xr.DataArray(
            np.broadcast_to(values.astype(np.float32)[..., None], shape), dims=dims
        )
    return xr.Dataset(
        fields,
        coords={
            "time": np.array(["2024-10-06T18", "2024-10-07T00"], dtype="datetime64[ns]"),
            "level": levels,
            "latitude": lat,
            "longitude": np.arange(0, 360, 0.25),
        },
    )


def test_urls_use_the_stream_of_each_cycle():
    root = "https://storage.googleapis.com/ecmwf-open-data"
    assert ifs_url(INIT) == (f"{root}/20241007/00z/ifs/0p25/oper/20241007000000-0h-oper-fc.grib2")
    assert ifs_url(INIT - timedelta(hours=6), "index") == (
        f"{root}/20241006/18z/ifs/0p25/scda/20241006180000-0h-scda-fc.index"
    )
    with pytest.raises(ValueError, match="six-hour"):
        ifs_url(INIT + timedelta(hours=1))


def test_ranges_merge_neighbours_and_skip_other_messages():
    # 82 needed messages of 10 bytes, with unneeded ones at 0 and after the 40th needed one.
    assert select_ranges(index_of()) == [(10, 400), (420, 420)]


def test_ranges_report_missing_messages():
    with pytest.raises(ValueError, match="13 required messages"):
        select_ranges(index_of(drop="w"))


def grib_like_frame(scale: float) -> dict[str, xr.DataArray]:
    """Values are 10, 20, 30, 40 times scale at longitudes -180, -90, 0, 90, plus offsets.

    Offsets: 1000 per level, 10000 per pressure field, 10 per surface field.
    """
    grid = {"latitude": np.array([10.0, -10.0]), "longitude": np.array([-180.0, -90.0, 0.0, 90.0])}
    by_longitude = np.array([10.0, 20.0, 30.0, 40.0]) * scale
    fields = {
        short: xr.DataArray(
            np.broadcast_to(
                by_longitude + 10000 * index + 1000 * np.arange(2)[:, None, None], (2, 2, 4)
            ).astype(np.float32),
            dims=("isobaricInhPa", "latitude", "longitude"),
            coords={"isobaricInhPa": np.array([500.0, 850.0]), **grid},
        )
        for index, short in enumerate(PRESSURE_SHORT)
    }
    fields.update(
        {
            short: xr.DataArray(
                np.broadcast_to(by_longitude + 10 * index, (2, 4)).astype(np.float32),
                dims=("latitude", "longitude"),
                coords=grid,
            )
            for index, short in enumerate(SURFACE_SHORT)
        }
    )
    return fields


def test_store_has_arco_names_units_and_zero_based_longitudes():
    times = [np.datetime64("2024-10-06T18", "ns"), np.datetime64("2024-10-07T00", "ns")]
    store = to_store([grib_like_frame(1), grib_like_frame(2)], times)
    assert set(store.data_vars) == set(ATMOSPHERE + SURFACE) - {"sea_surface_temperature"}
    assert store["temperature"].dims == ("time", "level", "latitude", "longitude")
    assert store["2m_temperature"].dims == ("time", "latitude", "longitude")
    np.testing.assert_array_equal(store["longitude"].values, [0, 90, 180, 270])
    np.testing.assert_array_equal(store["time"].values, times)
    np.testing.assert_array_equal(store["level"].values, [500, 850])
    # Longitudes -180, -90, 0, 90 hold 10, 20, 30, 40 and move to 180, 270, 0, 90.
    np.testing.assert_array_equal(
        store["temperature"].isel(time=0, level=0, latitude=0).values, [30, 40, 10, 20]
    )
    np.testing.assert_array_equal(
        store["temperature"].isel(time=1, level=1, latitude=1).values, [1060, 1080, 1020, 1040]
    )
    np.testing.assert_allclose(
        store["geopotential"].isel(time=0, level=0, latitude=0).values,
        np.array([10030, 10040, 10010, 10020]) * 9.80665,
        rtol=1e-6,
    )
    np.testing.assert_array_equal(
        store["10m_v_component_of_wind"].isel(time=1, latitude=0).values, [90, 110, 50, 70]
    )


def test_store_rejects_mismatched_axes():
    frame = grib_like_frame(1)
    frame["u"] = frame["u"].isel(latitude=slice(None, None, -1))
    with pytest.raises(ValueError, match="shared grid"):
        to_store([frame], [np.datetime64("2024-10-07T00", "ns")])
    frame = grib_like_frame(1)
    frame["msl"] = frame["msl"].transpose("longitude", "latitude")
    with pytest.raises(ValueError, match="axes"):
        to_store([frame], [np.datetime64("2024-10-07T00", "ns")])


def test_hybrid_takes_atmosphere_from_ifs_and_sst_from_era5(statics: SampleStatics):
    ifs = quarter_store(ATMOSPHERE + SURFACE[:-1])
    era5 = quarter_store(ATMOSPHERE + SURFACE, offset=1000)
    merged = hybrid_store(ifs, era5)
    assert set(merged.data_vars) == set(ATMOSPHERE + SURFACE)
    assert "sea_surface_temperature" not in ifs
    built = build_hybrid(INIT, ifs, era5, statics=statics, method="subsample")
    # Variable indices: temperature 0 and sea_surface_temperature 10.
    np.testing.assert_allclose(
        built["temperature"].isel(batch=0, time=1, level=0).isel(lat=LAT_45).values,
        100 + 2025 + 0.05,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        built["sea_surface_temperature"].isel(batch=0, time=0).isel(lat=LAT_45).values,
        1000 + 10 + 2025,
        rtol=1e-6,
    )
    assert not np.isnan(built["sea_surface_temperature"].isel(time=slice(0, 2)).values).any()


def test_hybrid_needs_every_ifs_point_in_era5():
    ifs = quarter_store(ATMOSPHERE + SURFACE[:-1])
    era5 = quarter_store(ATMOSPHERE + SURFACE).isel(time=slice(1, None))
    with pytest.raises(KeyError):
        hybrid_store(ifs, era5)


def test_cell_means_replace_point_values_in_the_builder(statics: SampleStatics):
    store = quarter_store(ATMOSPHERE + SURFACE)
    point = build_with_method(INIT, store, statics=statics, method="subsample")
    mean = build_with_method(INIT, store, statics=statics, method="cell_mean")
    offsets = np.arange(-0.5, 0.75, 0.25)
    weights = np.cos(np.deg2rad(45 + offsets)) * np.array([0.5, 1, 1, 1, 0.5])
    squared_mean = np.average((45 + offsets) ** 2, weights=weights)
    assert point["temperature"].sizes == mean["temperature"].sizes
    # Temperature is variable 0; the second input frame adds 100 and 850 hPa adds 0.85.
    level = list(LEVELS_HPA).index(850)
    np.testing.assert_allclose(
        point["temperature"].isel(batch=0, time=1, level=level).isel(lat=LAT_45, lon=20).item(),
        100 + 0.85 + 2025,
        rtol=1e-6,
    )
    np.testing.assert_allclose(
        mean["temperature"].isel(batch=0, time=1, level=level).isel(lat=LAT_45, lon=20).item(),
        100 + 0.85 + squared_mean,
        rtol=1e-6,
    )
    # Surface fields: 2m_temperature is variable 6, in the first frame.
    np.testing.assert_allclose(
        mean["2m_temperature"].isel(batch=0, time=0).isel(lat=LAT_45, lon=20).item(),
        6 + squared_mean,
        rtol=1e-6,
    )
    assert np.isfinite(mean["temperature"].isel(time=slice(0, 2)).values).all()
    assert np.isnan(mean["temperature"].isel(time=slice(2, None)).values).all()
    np.testing.assert_array_equal(mean["geopotential_at_surface"].values, 12)


def two_frame_dataset(values: dict[str, list[list[float]]]) -> xr.Dataset:
    """Variables of shape (batch, time, lat, lon) with two frames and a junk third frame."""
    return xr.Dataset(
        {
            name: xr.DataArray(np.array([frames], dtype=np.float64), dims=("batch", "time", "lat"))
            for name, frames in values.items()
        }
    )


def test_rms_uses_finite_paired_cells_of_the_two_input_frames():
    sample = two_frame_dataset({"a": [[0, 1], [1, 2], [999, 999]]})
    actual = two_frame_dataset({"a": [[1, 3], [5, np.nan], [-999, -999]]})
    # Differences 1, 2, 4 and one unpaired cell: sqrt((1 + 4 + 16) / 3).
    assert rms_difference(actual, sample, "a") == pytest.approx(np.sqrt(7))
    # The sample cells are 0, 1, 1, 2: mean 1, variance 0.5.
    assert sample_std(sample, "a") == pytest.approx(np.sqrt(0.5))
    with pytest.raises(ValueError, match="no finite paired"):
        rms_difference(two_frame_dataset({"a": [[np.nan], [np.nan]]}), sample.isel(lat=[0]), "a")


def test_score_divides_each_field_by_its_own_deviation():
    sample = two_frame_dataset(
        {
            "vertical_velocity": [[0, 2], [0, 2]],
            "specific_humidity": [[0, 4], [0, 4]],
            "mean_sea_level_pressure": [[0, 8], [0, 8]],
        }
    )
    actual = xr.Dataset(
        {name: sample[name] + 2 for name in sample.data_vars},
    )
    # A constant error of 2 against deviations 1, 2 and 4 gives 2, 1 and 0.5.
    assert coarsening_score(actual, sample) == pytest.approx((2 + 1 + 0.5) / 3)


def test_coarsening_choice_prefers_lower_score_and_ties_subsample():
    assert best_coarsening({"subsample": 0.2, "cell_mean": 0.1}) == "cell_mean"
    assert best_coarsening({"subsample": 0.1, "cell_mean": 0.2}) == "subsample"
    assert best_coarsening({"subsample": 0.1, "cell_mean": 0.1}) == "subsample"
    assert METHODS == ("subsample", "cell_mean")


def variable_rms(value: float, **overrides: float) -> dict[str, float]:
    return dict.fromkeys(ATMOSPHERE + SURFACE, value) | overrides


def test_rule_requires_every_atmospheric_variable_within_a_quarter():
    era5 = variable_rms(4.0)
    assert hybrid_is_source(variable_rms(1.0), era5)
    assert not hybrid_is_source(variable_rms(1.0, specific_humidity=1.0001), era5)
    assert hybrid_is_source(variable_rms(1.0, sea_surface_temperature=40.0), era5)


def test_outcome_lines_name_the_decision_and_the_method():
    era5 = variable_rms(4.0)
    passed = decision_line(variable_rms(1.0, geopotential=0.5), era5)
    assert passed == (
        "IC decision: hybrid (IFS open data atmosphere, ERA5 SST) is the source for hindcast "
        "inits from 2023-07-12 onward; the largest RMS ratio to ERA5 is 0.250 (temperature), "
        "limit 0.25."
    )
    failed = decision_line(variable_rms(1.0, vertical_velocity=2.0), era5)
    assert failed == (
        "IC decision: ERA5 is the source for all hindcast inits; "
        "vertical_velocity has a hybrid RMS of 0.500 of ERA5's, limit 0.25."
    )
    assert coarsening_line({"subsample": 0.3, "cell_mean": 0.25}) == (
        "Coarsening: cell_mean (mean normalised RMS difference: subsample 0.3000, "
        "cell_mean 0.2500); both builders use it."
    )


def replace_section(text: str, heading: str, body: str) -> str:
    head, _, rest = text.partition(f"\n{heading}\n")
    _, next_heading, tail = rest.partition("\n## ")
    return f"{head.rstrip()}\n\n{heading}\n\n{body.rstrip()}\n{next_heading}{tail}"


def test_section_replacement_keeps_neighbours_and_is_repeatable():
    text = "# Title\n\n## A\n\nold\n\n## B\n\nkeep\n"
    once = replace_section(text, "## A", "new\nlines")
    assert once == "# Title\n\n## A\n\nnew\nlines\n\n## B\n\nkeep\n"
    assert replace_section(once, "## A", "new\nlines") == once
    assert replace_section("# Title\n", "## A", "text") == "# Title\n\n## A\n\ntext\n"


def table(rows: list[list[str]], header: list[str]) -> list[str]:
    align = ["---", *("---:" for _ in header[1:])]
    return [
        "| " + " | ".join(header) + " |",
        "|" + "|".join(align) + "|",
        *("| " + " | ".join(row) + " |" for row in rows),
    ]


@pytest.mark.network
def test_hybrid_rms_and_coarsening_against_hres(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    sample = xr.load_dataset(SAMPLE_PATH, engine="h5netcdf")
    statics = load_sample_statics(SAMPLE_PATH)
    ifs = fetch_ifs_store(INIT, tmp_path)
    with xr.open_zarr(
        ARCO_STORE, consolidated=True, storage_options={"token": "anon"}, chunks=None
    ) as era5:
        hybrids: dict[Method, xr.Dataset] = {
            method: build_hybrid(INIT, ifs, era5, statics=statics, method=method)
            for method in METHODS
        }
        scores: dict[Method, float] = {
            method: coarsening_score(hybrids[method], sample) for method in METHODS
        }
        chosen = best_coarsening(scores)
        era5_built = build_with_method(INIT, era5, statics=statics, method=chosen)
    for built in (era5_built, *hybrids.values()):
        np.testing.assert_array_equal(
            built.datetime.isel(time=slice(0, 2)).values,
            sample.datetime.isel(time=slice(0, 2)).values,
        )
    era5_rms = {name: rms_difference(era5_built, sample, name) for name in ATMOSPHERE + SURFACE}
    hybrid_rms = {
        name: rms_difference(hybrids[chosen], sample, name) for name in ATMOSPHERE + SURFACE
    }
    fields = ("vertical_velocity", "specific_humidity", "mean_sea_level_pressure")
    coarsening_rows = [
        [
            name,
            *(
                f"{rms_difference(hybrids[name], sample, field) / sample_std(sample, field):.6f}"
                for field in fields
            ),
            f"{scores[name]:.6f}",
        ]
        for name in METHODS
    ]
    decision_rows = [
        [
            name,
            "yes" if name in ATMOSPHERE else "no",
            f"{era5_rms[name]:.9g}",
            f"{hybrid_rms[name]:.9g}",
            f"{hybrid_rms[name] / era5_rms[name]:.3f}",
        ]
        for name in ATMOSPHERE + SURFACE
    ]
    decision = decision_line(hybrid_rms, era5_rms)
    coarsening = coarsening_line(scores)
    lines = [
        (
            "Initialization: 2024-10-07 00:00 UTC. The hybrid builder reads archived IFS open "
            "data (0.25 degree step-0 fields from the 18 UTC and 00 UTC cycles, 13 levels) and "
            "takes sea surface temperature from ERA5."
        ),
        "",
        "RMS differences to the HRES sample use both input frames, finite paired cells and all",
        "model levels. Normalised differences divide each field's RMS by that field's global",
        "standard deviation in the HRES sample over the same cells.",
        "",
        (
            "Reproduce with `UV_CACHE_DIR=/tmp/stormfork-uv-cache uv run pytest -m network "
            "tests/test_ifs_open.py`."
        ),
        "",
        "### Coarsening check",
        "",
        "Hybrid builder, normalised RMS difference by coarsening method:",
        "",
        *table(coarsening_rows, ["Method", *fields, "Mean"]),
        "",
        f"### Source decision with {chosen}",
        "",
        "Only the atmospheric variables enter the rule; surface rows are shown for reference.",
        "",
        *table(
            decision_rows,
            [
                "Variable",
                "In rule",
                "ERA5 RMS (native units)",
                "Hybrid RMS (native units)",
                "Ratio",
            ],
        ),
        "",
        decision,
        "",
        coarsening,
    ]
    reproducibility = Path("docs/reproducibility.md")
    reproducibility.write_text(
        replace_section(
            reproducibility.read_text(),
            "## Hybrid initial-condition comparison",
            "\n".join(lines),
        )
    )
    (adr,) = Path("docs/adr").glob("0002-*.md")
    adr.write_text(replace_section(adr.read_text(), "## Outcome", f"{decision}\n\n{coarsening}"))
    with capsys.disabled():
        sys.stdout.write("\n" + "\n".join(lines) + "\n")
