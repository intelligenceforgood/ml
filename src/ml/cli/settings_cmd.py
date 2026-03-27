"""Settings inspection commands."""

from __future__ import annotations

import json

import typer

settings_app = typer.Typer(help="Configuration inspection.")


@settings_app.command("show")
def show() -> None:
    """Display current ML Platform settings (resolved from TOML + env vars)."""
    from ml.config import get_settings

    settings = get_settings()
    typer.echo(json.dumps(settings.model_dump(), indent=2, default=str))
