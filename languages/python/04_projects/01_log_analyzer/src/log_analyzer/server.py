"""HTTP API server built on stdlib http.server (no external deps for core)."""

from __future__ import annotations

import json
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, parse_qs

from log_analyzer.aggregator import Aggregator
from log_analyzer.parser import parse_line


_aggregator = Aggregator()
_lock = threading.Lock()


class LogHandler(BaseHTTPRequestHandler):
    """Minimal HTTP handler for log ingestion and query."""

    def do_POST(self) -> None:
        if self.path == "/v1/ingest":
            self._handle_ingest()
        else:
            self.send_error(404)

    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/healthz":
            self._json_response({"status": "ok"})
        elif parsed.path == "/v1/query":
            self._handle_query(parsed)
        elif parsed.path == "/metrics":
            self._handle_metrics()
        else:
            self.send_error(404)

    def _handle_ingest(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length).decode() if length else ""
        with _lock:
            entry = parse_line(body)
            if entry:
                _aggregator.add(entry)
        self._json_response({"accepted": True}, status=201)

    def _handle_query(self, parsed) -> None:
        params = parse_qs(parsed.query)
        stats = _aggregator.summary()
        data = {
            "total_requests": stats.total_requests,
            "errors": stats.errors,
            "error_rate": round(stats.error_rate, 4),
            "p50_ms": round(stats.p50, 2),
            "p99_ms": round(stats.p99, 2),
            "by_path": stats.by_path,
            "by_status": {str(k): v for k, v in stats.by_status.items()},
        }
        self._json_response(data)

    def _handle_metrics(self) -> None:
        stats = _aggregator.summary()
        lines = [
            "# HELP http_requests_total Total HTTP requests",
            "# TYPE http_requests_total counter",
            f'http_requests_total {stats.total_requests}',
            "# HELP http_errors_total Total error responses",
            "# TYPE http_errors_total counter",
            f'http_errors_total {stats.errors}',
        ]
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.end_headers()
        self.wfile.write("\n".join(lines).encode())

    def _json_response(self, data: dict, status: int = 200) -> None:
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt: str, *args) -> None:
        pass  # suppress default logging


def serve(host: str = "127.0.0.1", port: int = 8080) -> None:
    server = HTTPServer((host, port), LogHandler)
    print(f"Listening on {host}:{port}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()
