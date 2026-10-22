"""
AERIS FastAPI Application Factory.

Configures application lifespan, dependency state, CORS middleware,
structured exception handlers, and versioned REST API routes.
"""

from __future__ import annotations

import contextlib
from collections.abc import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from aeris.backend.config import BackendConfig
from aeris.backend.errors import register_exception_handlers
from aeris.backend.job_runner import JobRunner
from aeris.backend.routes import api_v1_router
from aeris.storage.dataset_store import ManagedDatasetStore
from aeris.storage.reconciliation import WorkspaceReconciler
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager


def create_app(
    config: BackendConfig | None = None,
    include_frontend: bool = False,
) -> FastAPI:
    """
    Construct and configure the AERIS FastAPI backend application.

    Parameters
    ----------
    config : BackendConfig | None
        Optional explicit backend configuration. If None, default loopback configuration is used.
    include_frontend : bool
        Whether to mount the Plotly Dash engineering frontend at root '/'. Defaults to False.

    Returns
    -------
    FastAPI
        Configured application instance with lifespan management.
    """
    cfg = config or BackendConfig()

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
        # Startup: Initialize workspace, database, job runner, and run crash recovery
        workspace = WorkspaceManager(workspace_root=cfg.workspace_root)
        workspace.init_workspace()

        repository = SQLiteMetadataRepository(
            db_path=workspace.db_path,
            wal_mode=True,
            auto_migrate=True,
        )

        dataset_store = ManagedDatasetStore(
            workspace=workspace,
            repository=repository,
            max_upload_size_bytes=cfg.max_upload_size_bytes,
        )

        job_runner = JobRunner(
            workspace=workspace,
            repository=repository,
            dataset_store=dataset_store,
            max_workers=cfg.max_workers,
        )

        # Recover interrupted/crashed jobs from previous execution
        job_runner.recover_interrupted_jobs()

        # Clean stale temporary upload files
        reconciler = WorkspaceReconciler(workspace=workspace, repository=repository)
        reconciler.clean_stale_temporary_files()

        # Attach instances to app state
        app.state.config = cfg
        app.state.workspace = workspace
        app.state.repository = repository
        app.state.dataset_store = dataset_store
        app.state.job_runner = job_runner

        yield

        # Shutdown: Gracefully stop background worker pool
        job_runner.shutdown(wait=True)

    app = FastAPI(
        title="AERIS Flight Test Analysis API",
        description="REST backend service for AERIS vehicle-aware flight test telemetry analysis.",
        version=cfg.app_version,
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    # Attach initial config to state so it's accessible before lifespan if needed
    app.state.config = cfg

    # Configure CORS for local web interface
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register structured exception handlers
    register_exception_handlers(app)

    # Register API routes
    app.include_router(api_v1_router)

    # Mount Plotly Dash frontend at root if requested
    if include_frontend:
        from aeris.frontend.app import mount_dash_app

        mount_dash_app(app)

    return app


app = create_app(include_frontend=True)
