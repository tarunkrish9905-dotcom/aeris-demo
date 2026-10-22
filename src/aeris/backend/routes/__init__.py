"""
AERIS Backend REST API Routers.
"""

from __future__ import annotations

from fastapi import APIRouter

from aeris.backend.routes.analyses import router as analyses_router
from aeris.backend.routes.configs import router as configs_router
from aeris.backend.routes.datasets import router as datasets_router
from aeris.backend.routes.health import router as health_router
from aeris.backend.routes.plots import router as plots_router
from aeris.backend.routes.projects import router as projects_router
from aeris.backend.routes.reports import router as reports_router

api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(health_router)
api_v1_router.include_router(projects_router)
api_v1_router.include_router(datasets_router)
api_v1_router.include_router(configs_router)
api_v1_router.include_router(analyses_router)
api_v1_router.include_router(plots_router)
api_v1_router.include_router(reports_router)

__all__ = ["api_v1_router"]
