"""Exercise header framing through urllib3's real parser, pools and retries."""

from __future__ import annotations

import asyncio
import http.client
import io
import socket
from pathlib import Path
from typing import cast

import pytest
import urllib3

from nab_index.cache import OnDiskCache
from nab_index.cached_client import CachedAsyncSimpleClient
from nab_index.transport import HttpError
from nab_index.urllib3_async_transport import (
    Urllib3AsyncTransport,
    _FinalStatusResponse,
    _HTTPConnection,
    _HTTPSConnection,
)

METADATA = b"Metadata-Version: 2.1\nName: demo\nVersion: 1.0\nRequires-Dist: leaf\n"
COMPLETE = (
    b"HTTP/1.1 200 OK\r\nContent-Length: "
    + str(len(METADATA)).encode()
    + b"\r\n\r\n"
    + METADATA
)
INCOMPLETE = [
    pytest.param(b"HTTP/1.1 200 OK\r\n", id="status-only"),
    pytest.param(b"HTTP/1.1 200 OK\r\nContent-Type: text/", id="partial-header"),
    pytest.param(
        b"HTTP/1.1 200 OK\r\nContent-Type: text/plain\r\n", id="missing-terminator"
    ),
]
INFORMATIONAL = [
    pytest.param(b"", id="final-only"),
    pytest.param(b"HTTP/1.1 100 Continue\r\n\r\n", id="continue"),
    pytest.param(b"HTTP/1.1 103 Early Hints\r\n\r\n", id="early-hints"),
]


class _WireSocket:
    """Replace socket IO with a tracked byte stream, leaving HTTP parsing intact."""

    def __init__(self, response: bytes) -> None:
        self.reader = io.BytesIO(response)

    def makefile(self, mode: str) -> io.BytesIO:
        return self.reader

    def settimeout(self, timeout: float | None) -> None:
        pass

    def sendall(self, data: bytes) -> None:
        pass

    def close(self) -> None:
        pass


@pytest.fixture
def wire_sockets(monkeypatch: pytest.MonkeyPatch) -> list[_WireSocket]:
    """Serve queued responses through both pool types without opening sockets."""
    sockets: list[_WireSocket] = []

    def connect(connection: _HTTPConnection | _HTTPSConnection) -> None:
        monkeypatch.setattr(connection, "sock", sockets.pop(0))
        connection.is_verified = True

    monkeypatch.setattr(_HTTPConnection, "connect", connect)
    monkeypatch.setattr(_HTTPSConnection, "connect", connect)
    monkeypatch.setattr("nab_index.urllib3_async_transport.time.sleep", lambda _: None)
    return sockets


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("informational", INFORMATIONAL)
@pytest.mark.parametrize("incomplete", INCOMPLETE)
def test_header_eof_recovers_before_caching_metadata(
    tmp_path: Path,
    wire_sockets: list[_WireSocket],
    scheme: str,
    informational: bytes,
    incomplete: bytes,
) -> None:
    responses = [_WireSocket(informational + incomplete), _WireSocket(COMPLETE)]
    wire_sockets.extend(responses)
    index_url = f"{scheme}://index.example/simple/"
    metadata_url = index_url + "demo/demo-1.0-py3-none-any.whl.metadata"
    cache = OnDiskCache(tmp_path, index_url)

    async def get_twice() -> None:
        async with CachedAsyncSimpleClient(Urllib3AsyncTransport(), cache) as client:
            assert (
                await client.get_metadata_text("demo", "1.0", metadata_url)
                == METADATA.decode()
            )
            assert (
                await client.get_metadata_text("demo", "1.0", metadata_url)
                == METADATA.decode()
            )

    asyncio.run(get_twice())

    assert not wire_sockets
    assert cache.get_metadata("demo", metadata_url) == METADATA.decode()

    assert all(response.reader.closed for response in responses)


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize("informational", INFORMATIONAL)
@pytest.mark.parametrize("incomplete", INCOMPLETE)
def test_persistent_header_eof_does_not_publish_metadata(
    tmp_path: Path,
    wire_sockets: list[_WireSocket],
    scheme: str,
    informational: bytes,
    incomplete: bytes,
) -> None:
    responses = [_WireSocket(informational + incomplete) for _ in range(8)]
    wire_sockets.extend(responses)
    index_url = f"{scheme}://index.example/simple/"
    metadata_url = index_url + "demo/demo-1.0-py3-none-any.whl.metadata"
    cache = OnDiskCache(tmp_path, index_url)

    async def get_twice() -> None:
        async with CachedAsyncSimpleClient(Urllib3AsyncTransport(), cache) as client:
            for _ in range(2):
                with pytest.raises(HttpError) as error:
                    await client.get_metadata_text("demo", "1.0", metadata_url)

                retry_error = error.value.__cause__
                assert isinstance(retry_error, urllib3.exceptions.MaxRetryError)
                reason = retry_error.reason
                assert isinstance(reason, urllib3.exceptions.ProtocolError)
                assert isinstance(reason.args[1], http.client.IncompleteRead)

                assert cache.get_metadata("demo", metadata_url) is None

    asyncio.run(get_twice())

    assert not wire_sockets
    assert not any(path.is_file() for path in tmp_path.rglob("*"))

    assert all(response.reader.closed for response in responses)


@pytest.mark.parametrize("scheme", ["http", "https"])
@pytest.mark.parametrize(
    ("raw", "status", "body"),
    [
        pytest.param(COMPLETE, 200, METADATA, id="length"),
        pytest.param(
            b"HTTP/1.1 200 OK\r\n\r\n" + METADATA, 200, METADATA, id="close-delimited"
        ),
        pytest.param(
            b"HTTP/1.1 200 OK\r\nTransfer-Encoding: chunked\r\n\r\n"
            + f"{len(METADATA):x}\r\n".encode()
            + METADATA
            + b"\r\n0\r\n\r\n",
            200,
            METADATA,
            id="chunked",
        ),
        pytest.param(
            b"HTTP/1.1 200 OK\r\nContent-Length: 0\r\n\r\n", 200, b"", id="empty-length"
        ),
        pytest.param(b"HTTP/1.1 200 OK\r\n\r\n", 200, b"", id="empty-close-delimited"),
        pytest.param(
            b"HTTP/1.1 304 Not Modified\r\nContent-Length: 64\r\n\r\n",
            304,
            b"",
            id="304",
        ),
        pytest.param(COMPLETE.replace(b"\r\n", b"\n"), 200, METADATA, id="lf-only"),
        pytest.param(
            b"HTTP/1.1 100 Continue\r\n\r\nHTTP/1.1 103 Early Hints\r\n\r\n" + COMPLETE,
            200,
            METADATA,
            id="informational",
        ),
    ],
)
def test_complete_headers_preserve_response_body(
    wire_sockets: list[_WireSocket], scheme: str, raw: bytes, status: int, body: bytes
) -> None:
    response = _WireSocket(raw)
    wire_sockets.append(response)

    async def get() -> None:
        transport = Urllib3AsyncTransport()
        try:
            result = await transport.get(f"{scheme}://index.example/demo.metadata")
            assert result.status_code == status
            assert result.content == body
        finally:
            await transport.aclose()

    asyncio.run(get())

    assert not wire_sockets
    assert response.reader.closed


@pytest.mark.parametrize("scheme", ["http", "https"])
def test_invalid_status_closes_header_stream(
    wire_sockets: list[_WireSocket], scheme: str
) -> None:
    responses = [_WireSocket(b"not HTTP\r\n\r\n") for _ in range(4)]
    wire_sockets.extend(responses)

    async def get() -> None:
        transport = Urllib3AsyncTransport()
        try:
            with pytest.raises(HttpError) as error:
                await transport.get(f"{scheme}://index.example/demo.metadata")

            retry_error = error.value.__cause__
            assert isinstance(retry_error, urllib3.exceptions.MaxRetryError)
            reason = retry_error.reason
            assert isinstance(reason, urllib3.exceptions.ProtocolError)
            assert isinstance(reason.args[1], http.client.BadStatusLine)
        finally:
            await transport.aclose()

    asyncio.run(get())

    assert not wire_sockets
    assert all(response.reader.closed for response in responses)


def test_begin_again_preserves_body_reader() -> None:
    wire = _WireSocket(COMPLETE)
    # HTTPResponse needs only makefile() from this socket boundary.
    response = _FinalStatusResponse(cast("socket.socket", wire))
    try:
        response.begin()
        response.begin()
        assert response.fp is wire.reader
        assert response.read() == METADATA
    finally:
        response.close()

    assert wire.reader.closed
