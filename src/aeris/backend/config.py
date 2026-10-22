"""
AERIS Backend Configuration.

Defines operational settings, workspace root resolution, default loopback host,
upload limits, and background worker concurrency.
"""

from __future__ import annotations

import os
import pathlib

from pydantic import BaseModel, ConfigDict, Field


def _default_workspace_path() -> pathlib.Path:
    env_path = os.environ.get("AERIS_WORKSPACE_DIR")
    if env_path:
        return pathlib.Path(env_path).resolve()
    return (pathlib.Path.home() / ".aeris_workspace").resolve()


class BackendConfig(BaseModel):
    """Configuration settings for AERIS FastAPI backend service."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    host: str = Field(
        default="127.0.0.1",
        description="Loopback bind host (default: 127.0.0.1).",
    )
    port: int = Field(
        default=8000,
        ge=1,
        le=65535,
        description="Bind port for FastAPI service.",
    )
    workspace_root: pathlib.Path = Field(
        default_factory=_default_workspace_path,
        description="Authoritative root directory for AERIS managed storage.",
    )
    max_upload_size_bytes: int = Field(
        default=200 * 1024 * 1024,  # 200 MB
        ge=1024,
        description="Maximum allowed telemetry upload size in bytes.",
    )
    max_workers: int = Field(
        default=2,
        ge=1,
        le=8,
        description="Maximum background thread pool worker concurrency.",
    )
    cors_origins: list[str] = Field(
        default=["http://127.0.0.1:8000", "http://localhost:8000"],
        description="Allowed CORS origins for local web client.",
    )
    app_version: str = Field(
        default="0.1.0",
        description="Semantic application release version.",
    )
