import io
import zipfile
from collections.abc import Callable
from pathlib import Path

import pytest
from PIL import Image

from stormfork.export.archive import safe_extract

SYMLINK_MODE = 0o120777 << 16


def png(size: tuple[int, int], mode: str = "L") -> bytes:
    buffer = io.BytesIO()
    Image.new(mode, size, 7).save(buffer, format="PNG")
    return buffer.getvalue()


def build(path: Path, entries: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        for name, data in entries.items():
            archive.writestr(name, data)
    return path


def listing(directory: Path) -> list[str]:
    return sorted(str(path.relative_to(directory)) for path in directory.rglob("*"))


def test_a_clean_archive_is_unpacked(tmp_path: Path):
    archive = build(
        tmp_path / "ok.zip",
        {
            "manifest.json": b"{}",
            "baseline/mslp.png": png((1805, 1629)),
            "baseline/u10.png": png((905, 819)),
            "preview/baseline_mslp_f0.png": png((360, 181)),
        },
    )
    dest = tmp_path / "out"
    safe_extract(archive, dest)
    assert listing(dest) == [
        "baseline",
        "baseline/mslp.png",
        "baseline/u10.png",
        "manifest.json",
        "preview",
        "preview/baseline_mslp_f0.png",
    ]
    assert (dest / "manifest.json").read_bytes() == b"{}"


def symlink_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        info = zipfile.ZipInfo("link")
        info.external_attr = SYMLINK_MODE
        archive.writestr(info, "/etc/passwd")
    return path


def duplicate_archive(path: Path) -> Path:
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("a.txt", b"1")
        with pytest.warns(UserWarning, match="Duplicate name"):
            archive.writestr("a.txt", b"2")
    return path


@pytest.mark.parametrize(
    ("name", "message"),
    [
        ("../evil", "plain relative path"),
        ("a/../../evil", "plain relative path"),
        ("/etc/evil", "plain relative path"),
        ("C:/evil", "plain relative path"),
        ("a\\evil", "plain relative path"),
    ],
)
def test_unsafe_names_are_refused_before_writing(tmp_path: Path, name: str, message: str):
    archive = build(tmp_path / "bad.zip", {"fine.txt": b"x", name: b"x"})
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match=message):
        safe_extract(archive, dest)
    assert not dest.exists()
    assert not (tmp_path / "evil").exists()
    assert listing(tmp_path) == ["bad.zip"]


@pytest.mark.parametrize(
    ("make", "message"),
    [(symlink_archive, "link or special"), (duplicate_archive, "appears twice")],
)
def test_links_and_duplicate_names_are_refused(
    tmp_path: Path, make: Callable[[Path], Path], message: str
):
    archive = make(tmp_path / "bad.zip")
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match=message):
        safe_extract(archive, dest)
    assert listing(tmp_path) == ["bad.zip"]


def test_too_many_entries_are_refused(tmp_path: Path):
    archive = build(tmp_path / "many.zip", {f"{index}.txt": b"x" for index in range(5)})
    with pytest.raises(ValueError, match="5 entries"):
        safe_extract(archive, tmp_path / "out", max_entries=4)
    assert listing(tmp_path) == ["many.zip"]


def test_a_declared_total_over_the_cap_is_refused(tmp_path: Path):
    archive = build(tmp_path / "big.zip", {"a.bin": b"0" * 100, "b.bin": b"0" * 100})
    with pytest.raises(ValueError, match="declares more than 150"):
        safe_extract(archive, tmp_path / "out", max_total_bytes=150)
    assert listing(tmp_path) == ["big.zip"]


def test_a_header_that_understates_the_size_leaves_nothing_behind(tmp_path: Path):
    archive = tmp_path / "lie.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
        bundle.writestr("a.bin", b"0" * 1000)
    data = archive.read_bytes()
    # The local header and the central directory both carry the uncompressed size.
    declared = (1000).to_bytes(4, "little")
    assert data.count(declared) == 2
    archive.write_bytes(data.replace(declared, (10).to_bytes(4, "little")))
    dest = tmp_path / "out"
    with pytest.raises(ValueError, match="not a valid zip"):
        safe_extract(archive, dest, max_total_bytes=500)
    assert listing(tmp_path) == ["lie.zip"]


@pytest.mark.parametrize(
    ("name", "size", "mode", "message"),
    [
        ("baseline/mslp.png", (1800, 1629), "L", "1800 x 1629"),
        ("baseline/mslp.png", (905, 819), "L", "905 x 819"),
        ("preview/x.png", (905, 819), "L", "905 x 819"),
        ("baseline/u10.png", (905, 818), "L", "905 x 818"),
        ("baseline/mslp.png", (1805, 1629), "RGB", "mode RGB"),
    ],
)
def test_pngs_must_have_a_known_size_and_mode(
    tmp_path: Path, name: str, size: tuple[int, int], mode: str, message: str
):
    archive = build(tmp_path / "png.zip", {"manifest.json": b"{}", name: png(size, mode)})
    with pytest.raises(ValueError, match=message):
        safe_extract(archive, tmp_path / "out")
    assert listing(tmp_path) == ["png.zip"]


def test_a_file_that_is_not_a_png_is_refused(tmp_path: Path):
    archive = build(tmp_path / "png.zip", {"a.png": b"not an image"})
    with pytest.raises(ValueError, match="not a readable PNG"):
        safe_extract(archive, tmp_path / "out")
    assert listing(tmp_path) == ["png.zip"]


def test_a_corrupt_zip_is_a_value_error(tmp_path: Path):
    archive = tmp_path / "bad.zip"
    archive.write_bytes(b"PK nothing")
    with pytest.raises(ValueError, match="not a valid zip"):
        safe_extract(archive, tmp_path / "out")
