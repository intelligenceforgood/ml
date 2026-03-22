"""Serving container entry point.

Re-exports the FastAPI app from ``ml.serving.app`` for use as the
Vertex AI Endpoint serving container.
"""

from ml.serving.app import app  # noqa: F401
