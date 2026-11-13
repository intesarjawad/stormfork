"""Pack an experiment's release zip, and unpack a zip without trusting it.

Every unzip in the repository goes through safe_extract. The archive is judged
from its directory first, so a refused archive writes nothing; the bytes
actually written are then counted, because a header can understate its entry.
"""

import re
import shutil
import stat
import zipfile
from collections.abc import Iterable
from pathlib import Path, PurePosixPath

from PIL import Image

MAX_TOTAL_BYTES = 24_000_000  # The experiment cap in scripts/perf_budget.py; change them together.
MAX_ENTRIES = 400
CHUNK_BYTES = 1 << 16
DRIVE_LETTER = re.compile(r"^[A-Za-z]:")
ATLAS_SIZE: tuple[int, int] = (1805, 1629)
WIND_FALLBACK_SIZE: tuple[int, int] = (905, 819)
WIND_FALLBACK_NAMES = frozenset({"u10.png", "v10.png"})
PREVIEW_SIZE: tuple[int, int] = (360, 181)
PNG_MODE = "L"
# The earliest time a zip header can hold, so an archive does not depend on when it was made.
ZIP_EPOCH = (1980, 1, 1, 0, 0, 0)
REGULAR_FILE_MODE = 0o100644
UNIX_CREATOR = 3


def _check_name(name: str) -> None:
    path = PurePosixPath(name)
    if (
        not name
        or "\\" in name
        or "\0" in name
        or path.is_absolute()
        or DRIVE_LETTER.match(name)
        or ".." in path.parts
    ):
        msg = f"archive entry {name!r} is not a plain relative path"
        raise ValueError(msg)


def _check_png(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> None:
    """Read only the PNG header, so a decompression bomb is never decoded."""
    try:
        with archive.open(info) as source, Image.open(source) as image:
            size, mode = image.size, image.mode
    except (OSError, zipfile.BadZipFile) as error:
        msg = f"archive entry {info.filename!r} is not a readable PNG ({error})"
        raise ValueError(msg) from error
    allowed = {ATLAS_SIZE, PREVIEW_SIZE}
    if PurePosixPath(info.filename).name in WIND_FALLBACK_NAMES:
        allowed.add(WIND_FALLBACK_SIZE)
    if size not in allowed:
        msg = f"archive entry {info.filename!r} is {size[0]} x {size[1]}, which no export uses"
        raise ValueError(msg)
    if mode != PNG_MODE:
        msg = f"archive entry {info.filename!r} has mode {mode}, not {PNG_MODE}"
        raise ValueError(msg)


def _check_directory(
    archive: zipfile.ZipFile, max_total_bytes: int, max_entries: int
) -> list[zipfile.ZipInfo]:
    entries = archive.infolist()
    if len(entries) > max_entries:
        msg = f"archive has {len(entries)} entries; the limit is {max_entries}"
        raise ValueError(msg)
    seen: set[str] = set()
    for info in entries:
        _check_name(info.filename)
        kind = stat.S_IFMT(info.external_attr >> 16)
        if kind not in {0, stat.S_IFREG, stat.S_IFDIR}:
            msg = f"archive entry {info.filename!r} is a link or special file"
            raise ValueError(msg)
        key = PurePosixPath(info.filename).as_posix().rstrip("/").casefold()
        if key in seen:
            msg = f"archive entry {info.filename!r} appears twice"
            raise ValueError(msg)
        seen.add(key)
    if sum(info.file_size for info in entries) > max_total_bytes:
        msg = f"archive declares more than {max_total_bytes} bytes"
        raise ValueError(msg)
    for info in entries:
        if info.filename.casefold().endswith(".png"):
            _check_png(archive, info)
    return entries


def _write(
    archive: zipfile.ZipFile, entries: list[zipfile.ZipInfo], dest: Path, limit: int
) -> None:
    root = dest.resolve()
    written = 0
    for info in entries:
        target = dest / info.filename
        if not target.resolve().is_relative_to(root):
            msg = f"archive entry {info.filename!r} escapes the destination"
            raise ValueError(msg)
        if info.is_dir():
            target.mkdir(parents=True, exist_ok=True)
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        with archive.open(info) as source, target.open("xb") as sink:
            while chunk := source.read(CHUNK_BYTES):
                written += len(chunk)
                if written > limit:
                    msg = f"archive unpacks to more than {limit} bytes"
                    raise ValueError(msg)
                sink.write(chunk)


def safe_extract(
    zip_path: Path,
    dest: Path,
    max_total_bytes: int = MAX_TOTAL_BYTES,
    max_entries: int = MAX_ENTRIES,
) -> None:
    """Unpack a zip into dest after checking every entry.

    Refuses, before writing anything, an entry whose path is absolute, has a
    drive letter, a backslash or a ".." part; a symlink or other special file;
    a repeated name; more than max_entries entries; a declared total above
    max_total_bytes; or a PNG that is not a mode L atlas, wind fallback or
    preview of the sizes an export uses. Then counts the bytes actually written
    and deletes dest if they pass the cap.

    Raises:
        ValueError: The archive is corrupt or breaks one of the rules above.
    """
    try:
        with zipfile.ZipFile(zip_path) as archive:
            entries = _check_directory(archive, max_total_bytes, max_entries)
            dest.mkdir(parents=True, exist_ok=True)
            try:
                _write(archive, entries, dest, max_total_bytes)
            except (ValueError, zipfile.BadZipFile):
                shutil.rmtree(dest, ignore_errors=True)
                raise
    except zipfile.BadZipFile as error:
        msg = f"archive is not a valid zip file ({error})"
        raise ValueError(msg) from error


def write_release_zip(directory: Path, paths: Iterable[str], destination: Path) -> None:
    """Pack the files at paths, relative to directory, into a byte-reproducible zip.

    Entries sit at the archive root in sorted order, with a fixed time, mode and
    creator system, and are stored uncompressed: the atlases are compressed
    PNGs already, and deflate output can change with the zlib build.

    Raises:
        OSError: A file cannot be read or the zip cannot be written.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(destination, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(set(paths)):
            _check_name(path)
            info = zipfile.ZipInfo(path, date_time=ZIP_EPOCH)
            info.create_system = UNIX_CREATOR
            info.external_attr = REGULAR_FILE_MODE << 16
            archive.writestr(info, (directory / path).read_bytes())
