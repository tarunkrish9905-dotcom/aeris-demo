"""
AERIS Backend Package.

FastAPI application factory, REST API routers, background job infrastructure,
and configuration settings.
"""

from __future__ import annotations

from aeris.backend.app import app, create_app
from aeris.backend.config import BackendConfig
from aeris.backend.job_runner import CancellationToken, JobRunner

__all__ = [
    "BackendConfig",
    "CancellationToken",
    "JobRunner",
    "app",
    "create_app",
]
