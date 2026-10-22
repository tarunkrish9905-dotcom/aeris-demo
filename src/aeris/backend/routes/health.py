"""
AERIS Backend Health and Version Routes.

Provides system liveness, metadata connectivity, and software version information.
"""

from __future__ import annotations

import datetime
import sys

from fastapi import APIRouter, Depends

from aeris.backend.config import BackendConfig
from aeris.backend.deps import get_config, get_repository
from aeris.backend.models import SystemHealthResponse, SystemVersionResponse
from aeris.storage.repository import SQLiteMetadataRepository

router = APIRouter(tags=["System"])


@router.get("/health", response_model=SystemHealthResponse)
async def get_health(
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> SystemHealthResponse:
    """Check system health, database accessibility, and active job count."""
    db_connected = False
    active_jobs = 0

    try:
        with repository._connection() as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM background_jobs WHERE state = 'RUNNING'")
            active_jobs = cursor.fetchone()[0]
            db_connected = True
    except Exception:
        db_connected = False

    return SystemHealthResponse(
        status="healthy" if db_connected else "degraded",
        database_connected=db_connected,
        active_jobs_count=active_jobs,
        timestamp=datetime.datetime.now(datetime.UTC),
    )


@router.get("/version", response_model=SystemVersionResponse)
async def get_version(
    config: BackendConfig = Depends(get_config),
) -> SystemVersionResponse:
    """Return AERIS release version, runtime platform, and REST API version."""
    return SystemVersionResponse(
        app_name="AERIS",
        version=config.app_version,
        python_version=sys.version.split()[0],
        api_version="v1",
    )
