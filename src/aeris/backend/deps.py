"""
AERIS Backend Dependencies.

Provides FastAPI dependency injection providers for configuration,
workspace management, metadata repository, dataset store, and background job runner.
"""

from __future__ import annotations

from typing import cast

from fastapi import Request

from aeris.backend.config import BackendConfig
from aeris.backend.job_runner import JobRunner
from aeris.storage.dataset_store import ManagedDatasetStore
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager


def get_config(request: Request) -> BackendConfig:
    """Retrieve application configuration from app state."""
    return cast(BackendConfig, request.app.state.config)


def get_workspace(request: Request) -> WorkspaceManager:
    """Retrieve authoritative WorkspaceManager instance from app state."""
    return cast(WorkspaceManager, request.app.state.workspace)


def get_repository(request: Request) -> SQLiteMetadataRepository:
    """Retrieve authoritative SQLiteMetadataRepository instance from app state."""
    return cast(SQLiteMetadataRepository, request.app.state.repository)


def get_dataset_store(request: Request) -> ManagedDatasetStore:
    """Retrieve authoritative ManagedDatasetStore instance from app state."""
    return cast(ManagedDatasetStore, request.app.state.dataset_store)


def get_job_runner(request: Request) -> JobRunner:
    """Retrieve authoritative background JobRunner instance from app state."""
    return cast(JobRunner, request.app.state.job_runner)
