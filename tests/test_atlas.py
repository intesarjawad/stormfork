import json
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from PIL import Image

from scripts.atlas_sizes import size_table, synthetic_fields
from stormfork.export.atlas import FRAME_SHAPE, export_run, pack_atlas, tile_origin
from stormfork.export.preview import write_grayscale
from stormfork.export.quantise import FILE_NAMES, decode, field_metadata, quantise

FIXTURES = Path(__file__).parent / "fixtures"


@pytest.fixture(scope="module")
def fields() -> dict[str, NDArray[np.float64]]:
    return synthetic_fields()


@pytest.fixture(scope="module", params=[1, 2])
def exported(
    request: pytest.FixtureRequest,
    tmp_path_factory: pytest.TempPathFactory,
    fields: dict[str, NDArray[np.float64]],
) -> tuple[Path, int]:
    resolution = int(request.param)
    destination = tmp_path_factory.mktemp(f"atlas-{resolution}")
    metadata = export_run(fields, destination, "baseline", wind_resolution_deg=resolution)
    assert metadata == field_metadata(wind_resolution_deg=resolution)
    return destination, resolution


def test_all_frames_seams_padding_and_previews(
    fields: dict[str, NDArray[np.float64]], exported: tuple[Path, int]
):
    destination, wind_resolution = exported
    assert len(list(destination.rglob("*.png"))) == 12
    for file in FILE_NAMES:
        resolution = wind_resolution if file in ("u10", "v10") else 1
        width = 360 // resolution
        height = 180 // resolution + 1
        values = fields[file].copy()
        if file == "tp6h":
            values[0] = 0
        expected = quantise(values, file)
        with Image.open(destination / "baseline" / f"{file}.png") as image:
            assert image.mode == "L"
            assert image.size == (5 * (width + 1), 9 * height)
            assert not image.info
            pixels = np.asarray(image)
        for frame in range(41):
            x = frame % 5 * (width + 1)
            y = frame // 5 * height
            tile = pixels[y : y + height, x : x + width + 1]
            np.testing.assert_array_equal(tile[:, :-1], expected[frame, ::resolution, ::resolution])
            np.testing.assert_array_equal(tile[:, -1], tile[:, 0])
            np.testing.assert_array_equal(
                decode(tile[:, :-1], file),
                decode(expected[frame, ::resolution, ::resolution], file),
            )
        assert not pixels[8 * height :, width + 1 :].any()
        with Image.open(destination / "preview" / f"baseline_{file}_f0.png") as image:
            assert image.size == (360, 181)
            assert image.mode == "L"
            assert not image.info
            np.testing.assert_array_equal(np.asarray(image), expected[0])
    assert np.isnan(fields["tp6h"][0]).all()


@pytest.mark.parametrize(("frame", "x", "y"), [(0, 0, 0), (4, 1444, 0), (5, 0, 181), (40, 0, 1448)])
@pytest.mark.parametrize("resolution", [1, 2])
def test_shader_tile_origins(frame: int, x: int, y: int, resolution: int):
    expected = (x, y) if resolution == 1 else (frame % 5 * 181, frame // 5 * 91)
    assert tile_origin(frame, resolution_deg=resolution) == expected
    for seam in (0, 1):
        texel_x = frame % 5 * (360 // resolution + 1) + 0.5 + seam * (360 // resolution)
        assert texel_x == expected[0] + 0.5 + seam * (360 // resolution)


def write_decoder_fixtures(fields: dict[str, NDArray[np.float64]], destination: Path) -> None:
    for file in ("mslp", "u10", "v10"):
        codes = quantise(fields[file], file)
        png_path = destination / f"sample_{file}.png"
        write_grayscale(pack_atlas(codes), png_path)
        with Image.open(png_path) as image:
            frame_zero = np.asarray(image)[:181, :360]
        payload = {
            "file": file,
            "shape": [181, 360],
            "order": "lat ascending -90..90, lon ascending 0..359, row-major",
            "bytes": frame_zero.tolist(),
            "decoded": decode(frame_zero, file).tolist(),
        }
        png_path.with_suffix(".json").write_text(
            json.dumps(payload, separators=(",", ":")) + "\n", encoding="utf-8"
        )


def test_browser_decoder_fixtures(fields: dict[str, NDArray[np.float64]], tmp_path: Path):
    write_decoder_fixtures(fields, tmp_path)
    for file in ("mslp", "u10", "v10"):
        filename = f"sample_{file}"
        with Image.open(FIXTURES / f"{filename}.png") as image:
            assert image.size == (1805, 1629)
            pixels = np.asarray(image)
        with Image.open(tmp_path / f"{filename}.png") as image:
            np.testing.assert_array_equal(pixels, np.asarray(image))
        expected_json = (tmp_path / f"{filename}.json").read_text(encoding="utf-8")
        assert (FIXTURES / f"{filename}.json").read_text(encoding="utf-8") == expected_json
        payload = json.loads(expected_json)
        np.testing.assert_array_equal(payload["bytes"], pixels[:181, :360])
        np.testing.assert_array_equal(payload["decoded"], decode(pixels[:181, :360], file))
        if file in ("u10", "v10"):
            box = pixels[105:116, 295:306]
            assert np.all(box == (191 if file == "u10" else 42))
            assert pixels[0, 0] == 128


@pytest.mark.parametrize("frame", [-1, 41])
def test_invalid_frame(frame: int):
    with pytest.raises(ValueError, match="frame"):
        tile_origin(frame)


@pytest.mark.parametrize("resolution", [0, 3])
def test_invalid_resolution(resolution: int):
    with pytest.raises(ValueError, match="resolution"):
        tile_origin(0, resolution_deg=resolution)


@pytest.mark.parametrize(
    "shape", [(0, 181, 360), (42, 181, 360), (41, 181, 361), (41, 360, 181), (181, 360)]
)
def test_bad_layout(shape: tuple[int, int, int]):
    with pytest.raises(ValueError, match="shape"):
        pack_atlas(np.zeros(shape, dtype=np.uint8))


def test_png_header_and_invalid_pixel_type(tmp_path: Path):
    path = tmp_path / "test.png"
    write_grayscale(np.zeros((181, 360), dtype=np.uint8), path)
    header = path.read_bytes()
    assert header[24:26] == bytes((8, 0))  # PNG bit depth and grayscale colour type.
    for pixels in (np.zeros((181, 360), dtype=np.float64), np.zeros((181, 360, 1), dtype=np.uint8)):
        with pytest.raises(ValueError, match="uint8"):
            write_grayscale(pixels, path)
    with pytest.raises(ValueError, match="uint8"):
        pack_atlas(np.zeros(FRAME_SHAPE, dtype=np.float64))


def test_a_short_run_fills_the_first_tiles_of_a_full_size_atlas(
    fields: dict[str, NDArray[np.float64]], tmp_path: Path
):
    frames = 9
    export_run({file: values[:frames] for file, values in fields.items()}, tmp_path, "baseline")
    with Image.open(tmp_path / "baseline/mslp.png") as image:
        assert image.size == (1805, 1629)
        pixels = np.asarray(image)
    expected = quantise(fields["mslp"][:frames], "mslp")
    for frame in range(41):
        x, y = tile_origin(frame)
        tile = pixels[y : y + 181, x : x + 360]
        if frame < frames:
            np.testing.assert_array_equal(tile, expected[frame])
        else:
            assert not tile.any()


def test_more_frames_than_an_atlas_holds_name_the_limit(
    fields: dict[str, NDArray[np.float64]], tmp_path: Path
):
    longer = {file: np.concatenate([values, values[:1]]) for file, values in fields.items()}
    with pytest.raises(
        ValueError, match=r"an atlas holds 1 to 41 frames.*40 six-hour steps; got 42"
    ):
        export_run(longer, tmp_path, "baseline")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("problem", ["shape", "missing", "extra", "nonfinite", "precipitation"])
def test_invalid_export_writes_nothing(
    fields: dict[str, NDArray[np.float64]], tmp_path: Path, problem: str
):
    changed = dict(fields)
    if problem == "shape":
        changed["t2m"] = changed["t2m"][:-1]
    elif problem == "missing":
        del changed["v10"]
    elif problem == "extra":
        changed["extra"] = changed["mslp"]
    else:
        file = "tp6h" if problem == "precipitation" else "z500"
        changed[file] = changed[file].copy()
        changed[file][1, 0, 0] = np.nan
    with pytest.raises(ValueError, match=r"(shape|fields|finite)"):
        export_run(changed, tmp_path, "baseline")
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("run", ["", "..", "preview", "../baseline", "a/b", "a\\b", "C:baseline"])
def test_invalid_run_name(fields: dict[str, NDArray[np.float64]], tmp_path: Path, run: str):
    with pytest.raises(ValueError, match="directory"):
        export_run(fields, tmp_path, run)
    assert not list(tmp_path.iterdir())


def test_measurements_come_from_files(exported: tuple[Path, int]):
    destination, _ = exported
    table = size_table(destination, "Synthetic test fields.")
    total = sum(path.stat().st_size for path in destination.rglob("*.png"))
    assert f"Run including previews: {total} bytes." in table
    assert "Synthetic test fields." in table
