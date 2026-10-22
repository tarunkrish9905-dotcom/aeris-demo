"""
AERIS Backend Projects Router.

Provides CRUD operations and directory scaffolding for flight analysis projects.
"""

from __future__ import annotations

import contextlib
import datetime
import uuid

from fastapi import APIRouter, Depends, Query, status

from aeris.backend.deps import get_repository, get_workspace
from aeris.backend.models import (
    CreateProjectRequest,
    ProjectDetailResponse,
    ProjectListResponse,
    StandardSuccessResponse,
    UpdateProjectRequest,
)
from aeris.storage.exceptions import ProjectNotFoundError
from aeris.storage.models import ProjectRecord, ProjectStatus
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager

router = APIRouter(prefix="/projects", tags=["Projects"])


@router.post("", response_model=ProjectDetailResponse, status_code=status.HTTP_201_CREATED)
async def create_project(
    request: CreateProjectRequest,
    workspace: WorkspaceManager = Depends(get_workspace),
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> ProjectDetailResponse:
    """Create a new flight analysis project and scaffold workspace directories."""
    project_id = f"proj_{uuid.uuid4().hex[:12]}"
    now_dt = datetime.datetime.now(datetime.UTC)

    record = ProjectRecord(
        project_id=project_id,
        name=request.name.strip(),
        description=request.description.strip(),
        analyst_name=request.analyst_name.strip() if request.analyst_name else None,
        created_at=now_dt,
        updated_at=now_dt,
        status=ProjectStatus.ACTIVE,
    )

    repository.create_project(record)
    workspace.create_project_workspace(project_id)

    return ProjectDetailResponse(
        project_id=record.project_id,
        name=record.name,
        description=record.description,
        analyst_name=record.analyst_name,
        status=record.status.value,
        created_at=record.created_at,
        updated_at=record.updated_at,
        dataset_count=0,
        analysis_count=0,
    )


@router.get("", response_model=ProjectListResponse)
async def list_projects(
    status_filter: str | None = Query(default=None, alias="status"),
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> ProjectListResponse:
    """List projects, defaulting to ACTIVE status."""
    filter_enum = None
    if status_filter:
        with contextlib.suppress(ValueError):
            filter_enum = ProjectStatus(status_filter.upper())

    projects = repository.list_projects(status=filter_enum)
    items: list[ProjectDetailResponse] = []

    for p in projects:
        datasets = repository.list_datasets(p.project_id)
        runs = repository.list_analysis_runs(p.project_id)
        items.append(
            ProjectDetailResponse(
                project_id=p.project_id,
                name=p.name,
                description=p.description,
                analyst_name=p.analyst_name,
                status=p.status.value,
                created_at=p.created_at,
                updated_at=p.updated_at,
                dataset_count=len(datasets),
                analysis_count=len(runs),
            )
        )

    return ProjectListResponse(projects=items, total_count=len(items))


@router.get("/{project_id}", response_model=ProjectDetailResponse)
async def get_project(
    project_id: str,
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> ProjectDetailResponse:
    """Retrieve detailed project information, dataset count, and run history."""
    p = repository.get_project(project_id)
    if p is None:
        raise ProjectNotFoundError(f"Project '{project_id}' not found.")

    datasets = repository.list_datasets(p.project_id)
    runs = repository.list_analysis_runs(p.project_id)

    return ProjectDetailResponse(
        project_id=p.project_id,
        name=p.name,
        description=p.description,
        analyst_name=p.analyst_name,
        status=p.status.value,
        created_at=p.created_at,
        updated_at=p.updated_at,
        dataset_count=len(datasets),
        analysis_count=len(runs),
    )


@router.patch("/{project_id}", response_model=ProjectDetailResponse)
async def update_project(
    project_id: str,
    request: UpdateProjectRequest,
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> ProjectDetailResponse:
    """Update project metadata or lifecycle status."""
    status_enum = None
    if request.status:
        with contextlib.suppress(ValueError):
            status_enum = ProjectStatus(request.status.upper())

    updated = repository.update_project(
        project_id=project_id,
        name=request.name.strip() if request.name else None,
        description=request.description.strip() if request.description else None,
        analyst_name=request.analyst_name.strip() if request.analyst_name else None,
        status=status_enum,
    )

    datasets = repository.list_datasets(updated.project_id)
    runs = repository.list_analysis_runs(updated.project_id)

    return ProjectDetailResponse(
        project_id=updated.project_id,
        name=updated.name,
        description=updated.description,
        analyst_name=updated.analyst_name,
        status=updated.status.value,
        created_at=updated.created_at,
        updated_at=updated.updated_at,
        dataset_count=len(datasets),
        analysis_count=len(runs),
    )


@router.delete("/{project_id}", response_model=StandardSuccessResponse)
async def delete_project(
    project_id: str,
    purge: bool = Query(default=False),
    workspace: WorkspaceManager = Depends(get_workspace),
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> StandardSuccessResponse:
    """Soft-delete or permanently purge a project and its workspace storage."""
    existing = repository.get_project(project_id)
    if existing is None:
        raise ProjectNotFoundError(f"Project '{project_id}' not found.")

    if purge:
        workspace.purge_project_workspace(project_id)
        repository.delete_project(project_id, hard_delete=True)
        return StandardSuccessResponse(
            success=True,
            message=f"Project '{project_id}' and all managed files permanently purged.",
        )

    repository.delete_project(project_id, hard_delete=False)
    return StandardSuccessResponse(
        success=True,
        message=f"Project '{project_id}' marked as DELETED.",
    )
