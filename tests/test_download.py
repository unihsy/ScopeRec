import gzip
import hashlib
from unittest.mock import MagicMock

import pytest
import requests

from scoperec.download import download_source, validate_archive


@pytest.fixture
def archive():
    payload = gzip.compress(b"user_id,parent_asin,rating,timestamp\n", mtime=0)
    source = {
        "filename": "sample.csv.gz",
        "url": "https://example.invalid/sample.csv.gz",
        "expected_bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }
    return payload, source


def make_session(payload):
    response = MagicMock()
    response.__enter__.return_value = response
    response.status_code = 200
    response.headers = {}
    response.iter_content.return_value = [payload]
    session = MagicMock()
    session.get.return_value = response
    return session


def test_verified_download_and_cache(tmp_path, archive):
    payload, source = archive
    session = make_session(payload)
    result = download_source(source, tmp_path, session)
    assert result["sha256"] == source["sha256"]
    assert result["gzip_integrity_passed"] is True
    assert (tmp_path / source["filename"]).read_bytes() == payload
    assert not (tmp_path / (source["filename"] + ".part")).exists()
    session.reset_mock()
    assert download_source(source, tmp_path, session)["reused"] is True
    session.get.assert_not_called()


def test_truncated_download_is_not_published(tmp_path, archive):
    payload, source = archive
    with pytest.raises(requests.ConnectionError, match="Size mismatch"):
        download_source(source, tmp_path, make_session(payload[:-3]))
    assert not (tmp_path / source["filename"]).exists()


def test_changed_checksum_is_rejected(tmp_path, archive):
    payload, source = archive
    path = tmp_path / source["filename"]
    path.write_bytes(payload)
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        validate_archive(path, {**source, "sha256": "0" * 64})


def test_broken_gzip_is_rejected(tmp_path):
    path = tmp_path / "invalid.gz"
    path.write_bytes(b"not gzip")
    with pytest.raises(gzip.BadGzipFile):
        validate_archive(path, {"expected_bytes": 8})


def test_source_cannot_escape_destination(tmp_path, archive):
    _, source = archive
    with pytest.raises(ValueError, match="plain filename"):
        download_source({**source, "filename": "../elsewhere.gz"}, tmp_path, MagicMock())


def test_interrupted_stream_resumes(tmp_path, archive):
    payload, source = archive
    boundary = len(payload) // 2

    def interrupted_chunks():
        yield payload[:boundary]
        raise requests.exceptions.ChunkedEncodingError("connection interrupted")

    first_response = make_session(payload).get.return_value
    first_response.iter_content.return_value = interrupted_chunks()
    second_response = make_session(payload[boundary:]).get.return_value
    second_response.status_code = 206
    second_response.headers = {
        "Content-Range": f"bytes {boundary}-{len(payload) - 1}/{len(payload)}"
    }
    session = MagicMock()
    session.get.side_effect = [first_response, second_response]
    result = download_source(source, tmp_path, session)
    assert result["sha256"] == source["sha256"]
    assert session.get.call_args.kwargs["headers"]["Range"] == (
        f"bytes={boundary}-{len(payload) - 1}"
    )


def test_server_ignoring_range_restarts(tmp_path, archive):
    payload, source = archive
    partial = tmp_path / (source["filename"] + ".part")
    partial.write_bytes(payload[:10])
    download_source(source, tmp_path, make_session(payload))
    assert (tmp_path / source["filename"]).read_bytes() == payload


def test_incorrect_range_does_not_append(tmp_path, archive):
    payload, source = archive
    partial = tmp_path / (source["filename"] + ".part")
    partial.write_bytes(payload[:10])
    session = make_session(payload[10:])
    session.get.return_value.status_code = 206
    session.get.return_value.headers = {"Content-Range": "bytes 0-1/2"}
    with pytest.raises(ValueError, match="Content-Range"):
        download_source(source, tmp_path, session)
    assert partial.read_bytes() == payload[:10]


def test_complete_partial_is_verified_without_network(tmp_path, archive):
    payload, source = archive
    (tmp_path / (source["filename"] + ".part")).write_bytes(payload)
    session = MagicMock()
    download_source(source, tmp_path, session)
    session.get.assert_not_called()
    assert (tmp_path / source["filename"]).read_bytes() == payload


def test_bounded_segments_cover_archive_exactly(tmp_path, archive, monkeypatch):
    payload, source = archive
    monkeypatch.setattr("scoperec.download.RANGE_SIZE", 11)
    responses = []
    for offset in range(0, len(payload), 11):
        end = min(offset + 11, len(payload))
        response = make_session(payload[offset:end]).get.return_value
        response.status_code = 206
        response.headers = {"Content-Range": f"bytes {offset}-{end - 1}/{len(payload)}"}
        responses.append(response)
    session = MagicMock()
    session.get.side_effect = responses
    download_source(source, tmp_path, session)
    assert (tmp_path / source["filename"]).read_bytes() == payload
    assert session.get.call_count == len(responses)
    assert session.get.call_args_list[0].kwargs["headers"]["Range"] == "bytes=0-10"