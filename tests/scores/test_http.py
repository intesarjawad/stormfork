from io import BytesIO
from urllib.error import URLError

import pytest

from stormfork.downloads import OFFLINE_ENV
from stormfork.errors import UserError
from stormfork.scores import fetch


class RecordedOpener:
    def __init__(self, failures: int) -> None:
        self.failures = failures
        self.calls = 0

    def open(self, url: str, timeout_seconds: float) -> BytesIO:
        assert url == fetch.METADATA_URL
        assert timeout_seconds == 30
        self.calls += 1
        if self.calls <= self.failures:
            msg = "connection reset"
            raise URLError(msg)
        return BytesIO(b"recorded response")


@pytest.mark.parametrize("failures", [2, 3])
def test_http_retries_with_backoff_and_stops_after_three_attempts(
    monkeypatch: pytest.MonkeyPatch,
    failures: int,
):
    opener = RecordedOpener(failures)
    delays: list[int] = []
    monkeypatch.setattr(fetch, "open_https", opener.open)
    monkeypatch.setattr(fetch, "sleep", delays.append)
    if failures == 2:
        assert fetch.request_bytes(fetch.METADATA_URL) == b"recorded response"
    else:
        with pytest.raises(UserError, match=fetch.METADATA_URL) as raised:
            fetch.request_bytes(fetch.METADATA_URL)
        assert raised.value.exit_code == 3
    assert opener.calls == 3
    assert delays == [1, 2]


def test_http_opens_nothing_while_downloads_are_off():
    with pytest.raises(UserError, match=f"{OFFLINE_ENV} is set") as raised:
        fetch.request_bytes(fetch.METADATA_URL)
    assert raised.value.exit_code == 2


def test_http_rejects_non_https_before_opening():
    with pytest.raises(ValueError, match="HTTPS"):
        fetch.request_bytes("file:///etc/passwd")
