"""Entry point: routes to CLI or API mode."""

from log_analyzer.cli import cli


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
