from pathlib import Path

import pytest

from scripts.perf_budget import (
    bundle_gzip_sizes,
    bundle_section,
    check_budgets,
    check_bundle,
    replace_section,
)


def sized_file(path: Path, size: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as stream:
        stream.truncate(size)


def collection(root: Path) -> None:
    for run in ("baseline", "fork", "control"):
        sized_file(root / "hero" / run / "mslp.png", 2_000_000)
    sized_file(root / "stop" / "baseline" / "mslp.png", 2_000_000)


def test_all_budgets_accept_inclusive_caps(tmp_path: Path):
    collection(tmp_path)
    for run in ("baseline", "fork", "control"):
        sized_file(tmp_path / "hero" / run / "wind.png", 5_999_999)
        sized_file(tmp_path / "hero" / "preview" / f"{run}_mslp_f0.png", 1)
    sized_file(tmp_path / "collection.bin", 224_000_000)
    check_budgets(tmp_path, hero="hero", sweep_stops=("stop",))


@pytest.mark.parametrize(
    ("path", "size", "message"),
    [
        ("hero/preview/baseline_mslp_f0.png", 6_000_001, "hero/baseline"),
        ("hero/metadata.bin", 18_000_001, "hero:"),
        ("collection.bin", 242_000_001, "collection"),
        ("hero/baseline/mslp.png", 2_000_001, "default MSLP"),
        ("stop/baseline/mslp.png", 2_000_001, "stop MSLP"),
    ],
)
def test_each_budget_fails(tmp_path: Path, path: str, size: int, message: str):
    collection(tmp_path)
    sized_file(tmp_path / path, size)
    with pytest.raises(ValueError, match=message):
        check_budgets(tmp_path, hero="hero", sweep_stops=("stop",))


def test_missing_hero_file_is_not_zero_bytes(tmp_path: Path):
    collection(tmp_path)
    (tmp_path / "hero" / "fork" / "mslp.png").unlink()
    with pytest.raises(FileNotFoundError):
        check_budgets(tmp_path, hero="hero")


def test_experiment_paths_cannot_escape_collection(tmp_path: Path):
    with pytest.raises(ValueError, match="directory"):
        check_budgets(tmp_path, hero="../other")


def test_bundle_sizes_are_gzip_of_js_only(tmp_path: Path):
    (tmp_path / "assets").mkdir()
    (tmp_path / "assets" / "b.js").write_bytes(b"a" * 100_000)
    (tmp_path / "assets" / "a.mjs").write_bytes(b"x")
    (tmp_path / "index.html").write_bytes(b"<html>")
    sizes = bundle_gzip_sizes(tmp_path)
    assert list(sizes) == ["assets/a.mjs", "assets/b.js"]
    assert 20 <= sizes["assets/a.mjs"] <= 30
    assert 0 < sizes["assets/b.js"] < 1_000


def test_bundle_without_javascript_is_an_error(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        bundle_gzip_sizes(tmp_path)
    with pytest.raises(FileNotFoundError):
        bundle_gzip_sizes(tmp_path / "missing")


def test_bundle_cap_is_inclusive():
    check_bundle({"a.js": 100_000, "b.js": 150_000})
    with pytest.raises(ValueError, match="gzip JS bundle"):
        check_bundle({"a.js": 100_000, "b.js": 150_001})


def test_section_lists_sizes_total_and_commands():
    section = bundle_section({"assets/a.js": 10, "assets/b.js": 32}, "measure --bundle dist")
    assert "| assets/a.js | 10 |" in section
    assert "| Total | 42 |" in section
    assert "(cd site && pnpm run build)\nmeasure --bundle dist\n" in section
    assert "Cap: 250000 bytes." in section


def test_replace_section_keeps_other_content():
    old = "# Atlas\n\nbody\n\n## Perf budget\n\nold\n\n## Later\n\nkeep\n"
    assert replace_section(old, "## Perf budget\n\nnew\n") == (
        "# Atlas\n\nbody\n\n## Perf budget\n\nnew\n\n## Later\n\nkeep\n"
    )
    assert replace_section("# Atlas\n\nbody\n", "## Perf budget\n\nnew\n") == (
        "# Atlas\n\nbody\n\n## Perf budget\n\nnew\n"
    )
