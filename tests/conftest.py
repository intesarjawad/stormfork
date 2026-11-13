from pathlib import Path

import pytest

from stormfork.downloads import CACHE_ENV, OFFLINE_ENV


@pytest.fixture(autouse=True)
def offline_cache(
    request: pytest.FixtureRequest,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path_factory: pytest.TempPathFactory,
) -> Path:
    """Give every test an empty download cache, and no downloads unless it is marked network."""
    cache = tmp_path_factory.mktemp("download-cache")
    monkeypatch.setenv(CACHE_ENV, str(cache))
    if request.node.get_closest_marker("network") is None:
        monkeypatch.setenv(OFFLINE_ENV, "1")
    return cache
