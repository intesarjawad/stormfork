import gzip
from datetime import datetime
from pathlib import Path

import pytest

from stormfork.verify.adeck import adeck_url, fetch_adeck, parse_adeck, read_adeck

FIXTURE = Path(__file__).parent / "fixtures" / "adeck_milton.dat"
FIRST_INIT = datetime.fromisoformat("2024-10-05T00:00:00")
SECOND_INIT = datetime.fromisoformat("2024-10-05T12:00:00")


def test_fixture_reads_tenths_hemispheres_and_fixed_width():
    deck = read_adeck(FIXTURE)
    assert deck.storm == "AL142024"
    assert deck.aids() == ("AEMN", "AVNO", "EEMN", "OFCL")
    assert deck.inits() == (FIRST_INIT, SECOND_INIT)
    north_west = deck.track("AEMN", FIRST_INIT)
    assert north_west[24].lat == 22.5
    assert north_west[24].lon == -94.9
    assert north_west[24].valid_time == datetime.fromisoformat("2024-10-06T00:00:00")
    spaced = deck.track("EEMN", SECOND_INIT)[24]
    assert (spaced.lat, spaced.lon) == (23.0, -94.1)
    south_east = deck.track("AVNO", SECOND_INIT)[48]
    assert (south_east.lat, south_east.lon) == (-10.0, 10.0)


def test_duplicate_radius_rows_collapse_to_one_position():
    deck = read_adeck(FIXTURE)
    assert len(deck.track("OFCL", SECOND_INIT)) == 1


def test_conflicting_duplicate_positions_are_rejected():
    text = (
        "AL, 14, 2024100500, 03, AEMN, 24, 225N, 949W, 25, 1009, XX\n"
        "AL, 14, 2024100500, 03, AEMN, 24, 226N, 949W, 25, 1009, XX\n"
    )
    with pytest.raises(ValueError, match="conflicting lead 24"):
        parse_adeck(text)


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ("AL, 14, 2024100500, 03, AEMN, 24, 225N", "at least 8"),
        (
            "AL, 14, 2024100500, 03, AEMN, 24, 225E, 949W, 25, 1009, XX",
            "must use N or S",
        ),
        (
            "AL, 14, 2024100500, 03, AEMN, 24, 225N, 1900E, 25, 1009, XX",
            "outside",
        ),
        ("AL, 14, 2024, 03, AEMN, 24, 225N, 949W, 25, 1009, XX", "invalid cycle"),
    ],
)
def test_malformed_records_are_rejected(line: str, match: str):
    with pytest.raises(ValueError, match=match):
        parse_adeck(line)


def test_empty_deck_and_multiple_storms_are_rejected():
    with pytest.raises(ValueError, match="no forecast records"):
        parse_adeck("# nothing\n\n")
    text = (
        "AL, 14, 2024100500, 03, AEMN, 24, 225N, 949W, 25, 1009, XX\n"
        "EP, 11, 2024100500, 03, AEMN, 24, 125N, 1100W, 25, 1009, XX\n"
    )
    with pytest.raises(ValueError, match="multiple storms"):
        parse_adeck(text)


def test_gzip_fixture_is_read_like_a_plain_deck(tmp_path: Path):
    compressed = tmp_path / "aal142024.dat.gz"
    compressed.write_bytes(gzip.compress(FIXTURE.read_bytes()))
    assert read_adeck(compressed) == read_adeck(FIXTURE)


def test_url_uses_the_public_aid_directory():
    assert adeck_url("AL142024") == ("https://ftp.nhc.noaa.gov/atcf/aid_public/aal142024.dat.gz")


def test_fetch_decompresses_and_checks_the_storm():
    body = gzip.compress(FIXTURE.read_bytes())

    def request(url: str) -> bytes:
        assert url.endswith("/aal142024.dat.gz")
        return body

    deck = fetch_adeck("AL142024", request)
    assert deck.storm == "AL142024"
    with pytest.raises(ValueError, match="returned EP112024"):
        fetch_adeck(
            "AL142024",
            lambda _url: gzip.compress(
                b"EP, 11, 2024100500, 03, AEMN, 0, 125N, 1100W, 25, 0, XX\n"
            ),
        )
