"""Test server handler directly using raw socket connections."""

import json
import pytest
import socket
from http.server import HTTPServer
from threading import Thread

from log_analyzer.server import LogHandler


def _http_request(port, method, path, body=None):
    """Send raw HTTP/1.0 request and return (status, body)."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.settimeout(3)
    s.connect(("127.0.0.1", port))

    request = f"{method} {path} HTTP/1.0\r\nHost: localhost\r\n"
    if body:
        body_bytes = body.encode() if isinstance(body, str) else body
        request += f"Content-Length: {len(body_bytes)}\r\n"
        request += "Content-Type: application/json\r\n"
    request += "\r\n"
    if body:
        request = request.encode() + body_bytes
    else:
        request = request.encode()

    s.sendall(request)
    response = b""
    while True:
        try:
            chunk = s.recv(4096)
            if not chunk:
                break
            response += chunk
        except socket.timeout:
            break
    s.close()

    # Parse response
    header_end = response.find(b"\r\n\r\n")
    if header_end == -1:
        return 0, b""
    status_line = response[: response.find(b"\r\n")].decode()
    status = int(status_line.split(" ")[1])
    body = response[header_end + 4:]
    return status, body


@pytest.fixture(scope="module")
def server_port():
    server = HTTPServer(("127.0.0.1", 0), LogHandler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield port
    server.shutdown()


class TestServer:
    def test_healthz(self, server_port):
        status, data = _http_request(server_port, "GET", "/healthz")
        assert status == 200
        assert json.loads(data)["status"] == "ok"

    def test_ingest_and_query(self, server_port):
        status, _ = _http_request(
            server_port, "POST", "/v1/ingest",
            body=json.dumps({"status": 500, "path": "/fail"}),
        )
        assert status == 201

        status, data = _http_request(server_port, "GET", "/v1/query")
        assert status == 200
        result = json.loads(data)
        assert result["total_requests"] >= 1
        assert result["errors"] >= 1

    def test_metrics(self, server_port):
        status, data = _http_request(server_port, "GET", "/metrics")
        assert status == 200
        assert "http_requests_total" in data.decode()
