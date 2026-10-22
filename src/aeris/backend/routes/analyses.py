"""
AERIS Backend Analyses & Background Jobs Router.

Triggers asynchronous 15-stage analysis runs and tracks background execution progress.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status

from aeris.backend.deps import get_job_runner, get_repository
from aeris.backend.job_runner import JobRunner
from aeris.backend.models import (
    AnalysisDetailResponse,
    JobQueuedResponse,
    JobStatusResponse,
    StandardSuccessResponse,
    StartAnalysisRequest,
)
from aeris.storage.exceptions import DatasetNotFoundError, ProjectNotFoundError
from aeris.storage.repository import SQLiteMetadataRepository

router = APIRouter(tags=["Analyses & Jobs"])


@router.post(
    "/projects/{project_id}/analyses",
    response_model=JobQueuedResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def start_analysis(
    project_id: str,
    request: StartAnalysisRequest,
    repository: SQLiteMetadataRepository = Depends(get_repository),
    job_runner: JobRunner = Depends(get_job_runner),
) -> JobQueuedResponse:
    """Trigger 15-stage analysis pipeline as an asynchronous background job."""
    project = repository.get_project(project_id)
    if project is None:
        raise ProjectNotFoundError(f"Project '{project_id}' not found.")

    dataset = repository.get_dataset(request.dataset_id)
    if dataset is None or dataset.project_id != project_id:
        raise DatasetNotFoundError(
            f"Dataset '{request.dataset_id}' not found in project '{project_id}'."
        )

    _run_record, job_record = job_runner.submit_analysis_job(
        project_id=project_id,
        request=request,
    )

    return JobQueuedResponse(
        job_id=job_record.job_id,
        analysis_id=job_record.analysis_id,
        state=job_record.state,
        message="Analysis job accepted into background execution queue.",
    )


@router.get("/jobs/{job_id}", response_model=JobStatusResponse)
async def get_job_status(
    job_id: str,
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> JobStatusResponse:
    """Poll real-time background job status, stage milestone, and completion percent."""
    job = repository.get_background_job(job_id)
    if job is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Background job '{job_id}' not found.",
        )

    return JobStatusResponse(
        job_id=job.job_id,
        analysis_id=job.analysis_id,
        state=job.state,
        current_stage=job.current_stage,
        completed_stages=job.completed_stages_count,
        total_stages=job.total_stages_count,
        progress_percent=job.progress_percent,
        error_message=job.error_message,
        started_at=job.created_at,
        updated_at=job.updated_at,
    )


@router.post("/jobs/{job_id}/cancel", response_model=StandardSuccessResponse)
async def cancel_job(
    job_id: str,
    job_runner: JobRunner = Depends(get_job_runner),
) -> StandardSuccessResponse:
    """Request cooperative cancellation of a background job."""
    cancelled = job_runner.cancel_job(job_id)
    if not cancelled:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Job '{job_id}' not found or cannot be cancelled.",
        )

    return StandardSuccessResponse(
        success=True,
        message=f"Job '{job_id}' cancellation signaled successfully.",
    )


@router.get("/projects/{project_id}/analyses/{analysis_id}", response_model=AnalysisDetailResponse)
async def get_analysis_detail(
    project_id: str,
    analysis_id: str,
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> AnalysisDetailResponse:
    """Retrieve detailed analysis run metadata and summary diagnostic counts."""
    run = repository.get_analysis_run(analysis_id)
    if run is None or run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis run '{analysis_id}' not found in project '{project_id}'.",
        )

    return AnalysisDetailResponse(
        analysis_id=run.analysis_id,
        project_id=run.project_id,
        dataset_id=run.dataset_id,
        run_status=run.run_status,
        triggered_at=run.triggered_at,
        completed_at=run.completed_at,
        execution_duration_ms=run.execution_duration_ms,
        total_issues=run.total_issues_count,
        error_count=run.error_count,
        warning_count=run.warning_count,
        result_summary=run.result_summary,
    )
