"""Unified CLI entry point for I4G ML Platform.

Config precedence: settings.default.toml -> settings.dev.toml
-> env vars (I4G_ML_* with double underscores) -> CLI flags.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer

from ml.cli.dataset import dataset_app
from ml.cli.deploy import deploy_app
from ml.cli.eval import eval_app
from ml.cli.pipeline import pipeline_app
from ml.cli.retrain import retrain_app
from ml.cli.serve import serve_app
from ml.cli.settings_cmd import settings_app
from ml.cli.smoke import smoke_app

try:
    from importlib.metadata import version

    VERSION = version("i4g-ml")
except Exception:
    VERSION = "unknown"

APP_HELP = (
    "i4g-ml — ML Platform CLI for operators and developers. "
    "Manage datasets, training pipelines, model deployment, evaluation, and monitoring. "
    "Config precedence: settings.default.toml -> settings.dev.toml -> env vars (I4G_ML_* with __) -> CLI flags."
)

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

app = typer.Typer(add_completion=True, help=APP_HELP)

app.add_typer(dataset_app, name="dataset", help="Dataset creation, versioning, and export.")
app.add_typer(pipeline_app, name="pipeline", help="Training pipeline submission and management.")
app.add_typer(deploy_app, name="deploy", help="Model deployment and redeployment.")
app.add_typer(serve_app, name="serve", help="Batch prediction and serving utilities.")
app.add_typer(eval_app, name="eval", help="Evaluation, baseline, and framework comparison.")
app.add_typer(retrain_app, name="retrain", help="Retraining trigger evaluation and submission.")
app.add_typer(smoke_app, name="smoke", help="End-to-end smoke tests.")
app.add_typer(settings_app, name="settings", help="Configuration inspection.")


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context, version: bool = typer.Option(False, "--version", help="Show version and exit.")) -> None:
    """Show help when no subcommand is provided."""
    if version:
        typer.echo(f"i4g-ml {VERSION}")
        raise typer.Exit()
    if ctx.invoked_subcommand is None:
        typer.echo(ctx.get_help())
        raise typer.Exit()


if __name__ == "__main__":
    app()
