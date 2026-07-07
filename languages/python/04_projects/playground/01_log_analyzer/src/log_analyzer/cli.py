"""CLI entry point using click."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import click

from log_analyzer.aggregator import Aggregator
from log_analyzer.parser import parse_stream
from log_analyzer.server import serve


@click.group()
def cli() -> None:
    pass


@cli.command()
@click.option("--input", "-i", type=click.Path(exists=True, dir_okay=False),
              help="Input log file (default: stdin)")
@click.option("--format", "fmt", type=click.Choice(["auto", "combined", "json"]),
              default="auto", help="Log format")
@click.option("--output", "-o", type=click.Path(dir_okay=False),
              help="Output file (default: stdout)")
@click.option("--filter", "filter_pattern", default="",
              help="Filter by regex pattern")
@click.option("--json", "as_json", is_flag=True, help="JSON output")
def analyze(input: str | None, fmt: str, output: str | None,
            filter_pattern: str, as_json: bool) -> None:
    """Analyze a log file and print statistics."""
    if input:
        stream: click.utils.LazyFile = click.open_file(input, "r")
    else:
        stream = click.get_text_stream("stdin")

    import re
    filter_re = re.compile(filter_pattern) if filter_pattern else None

    agg = Aggregator()
    for entry in parse_stream(iter(stream.readline, ""), fmt=fmt):
        if filter_re and not filter_re.search(entry.body):
            continue
        agg.add(entry)

    stats = agg.summary()
    _emit_output(stats, as_json, output)


@cli.command()
@click.option("--port", default=8080, help="HTTP port")
@click.option("--host", default="127.0.0.1", help="Bind address")
def api(port: int, host: str) -> None:
    """Start the HTTP API server."""
    click.echo(f"Starting API server on {host}:{port}...")
    serve(host=host, port=port)


def _emit_output(stats, as_json: bool, output_path: str | None) -> None:
    if as_json:
        data = {
            "total_requests": stats.total_requests,
            "errors": stats.errors,
            "error_rate": round(stats.error_rate, 4),
            "p50_ms": round(stats.p50, 2),
            "p99_ms": round(stats.p99, 2),
            "top_paths": sorted(stats.by_path.items(),
                                key=lambda x: -x[1])[:10],
        }
        text = json.dumps(data, indent=2)
    else:
        lines = [
            f"{'Total Requests':<20} {stats.total_requests}",
            f"{'Errors':<20} {stats.errors}",
            f"{'Error Rate':<20} {stats.error_rate:.2%}",
            f"{'P50 Latency':<20} {stats.p50:.2f}ms",
            f"{'P99 Latency':<20} {stats.p99:.2f}ms",
            "",
            "--- Top Paths ---",
        ]
        for path, count in sorted(stats.by_path.items(),
                                  key=lambda x: -x[1])[:10]:
            lines.append(f"{path:<30} {count}")
        text = "\n".join(lines)

    if output_path:
        Path(output_path).write_text(text)
    else:
        click.echo(text)
