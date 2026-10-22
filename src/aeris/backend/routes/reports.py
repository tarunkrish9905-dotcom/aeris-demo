"""
AERIS Backend Reports Router.

Generates self-contained HTML and Markdown engineering reports and serves downloads.
"""

from __future__ import annotations

import datetime
import hashlib
import uuid

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import FileResponse

from aeris.backend.deps import get_dataset_store, get_repository, get_workspace
from aeris.backend.models import (
    GenerateReportRequest,
    ReportDetailResponse,
    ReportListResponse,
)
from aeris.core.pipeline.orchestrator import DataPipelineOrchestrator
from aeris.core.reporting.exporter import ReportExportFormat, export_pipeline_report
from aeris.storage.dataset_store import ManagedDatasetStore
from aeris.storage.models import ExportedReportRecord
from aeris.storage.repository import SQLiteMetadataRepository
from aeris.storage.workspace import WorkspaceManager

router = APIRouter(tags=["Reports"])


@router.post(
    "/projects/{project_id}/analyses/{analysis_id}/reports",
    response_model=ReportDetailResponse,
    status_code=status.HTTP_201_CREATED,
)
async def generate_report(
    project_id: str,
    analysis_id: str,
    request: GenerateReportRequest,
    workspace: WorkspaceManager = Depends(get_workspace),
    repository: SQLiteMetadataRepository = Depends(get_repository),
    dataset_store: ManagedDatasetStore = Depends(get_dataset_store),
) -> ReportDetailResponse:
    """Generate offline self-contained HTML or Markdown engineering report from analysis results."""
    run = repository.get_analysis_run(analysis_id)
    if run is None or run.project_id != project_id:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Analysis run '{analysis_id}' not found in project '{project_id}'.",
        )

    # 1. Resolve dataset and re-execute pipeline to extract complete report models
    dataset_path = dataset_store.get_dataset_path(run.dataset_id)
    orchestrator = DataPipelineOrchestrator()
    pipeline_result = orchestrator.run_csv_pipeline(dataset_path)

    # 2. Configure output format and paths
    fmt_upper = request.report_format.upper()
    if fmt_upper == "MARKDOWN":
        format_enum = ReportExportFormat.MARKDOWN
        ext = "md"
    else:
        format_enum = ReportExportFormat.HTML
        ext = "html"

    report_id = f"rep_{uuid.uuid4().hex[:12]}"
    reports_dir = workspace.get_reports_dir(project_id)
    reports_dir.mkdir(parents=True, exist_ok=True)
    target_path = reports_dir / f"{report_id}.{ext}"

    title = request.report_title or "AERIS Flight Test Engineering Report"

    # 3. Export report to disk
    export_res = export_pipeline_report(
        pipeline_result=pipeline_result,
        output_path=target_path,
        format=format_enum,
        overwrite=True,
        report_title=title,
    )

    now_dt = datetime.datetime.now(datetime.UTC)
    rel_path = str(target_path.relative_to(workspace.workspace_root)).replace("\\", "/")
    file_bytes = target_path.read_bytes()
    file_checksum = hashlib.sha256(file_bytes).hexdigest()
    file_size = export_res.bytes_written or len(file_bytes)

    # 4. Record in SQLite repository
    record = ExportedReportRecord(
        report_id=report_id,
        analysis_id=analysis_id,
        report_format=fmt_upper,
        report_title=title,
        storage_relative_path=rel_path,
        sha256_checksum=file_checksum,
        file_size_bytes=file_size,
        generated_at=now_dt,
    )
    repository.create_exported_report(record)

    return ReportDetailResponse(
        report_id=record.report_id,
        analysis_id=record.analysis_id,
        report_format=record.report_format,
        report_title=record.report_title,
        file_size_bytes=record.file_size_bytes,
        sha256_checksum=record.sha256_checksum,
        generated_at=record.generated_at,
    )


@router.get("/projects/{project_id}/reports", response_model=ReportListResponse)
async def list_project_reports(
    project_id: str,
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> ReportListResponse:
    """List all generated engineering reports in a project."""
    reports = repository.list_exported_reports(project_id=project_id)
    items = [
        ReportDetailResponse(
            report_id=r.report_id,
            analysis_id=r.analysis_id,
            report_format=r.report_format,
            report_title=r.report_title,
            file_size_bytes=r.file_size_bytes,
            sha256_checksum=r.sha256_checksum,
            generated_at=r.generated_at,
        )
        for r in reports
    ]
    return ReportListResponse(reports=items, total_count=len(items))


@router.get("/reports/{report_id}/download")
async def download_report(
    report_id: str,
    inline: bool = False,
    workspace: WorkspaceManager = Depends(get_workspace),
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> FileResponse:
    """Download or stream a self-contained HTML or Markdown report file."""
    record = repository.get_exported_report(report_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report '{report_id}' not found.",
        )

    file_path = workspace.resolve_safe_path(record.storage_relative_path, workspace.workspace_root)
    if not file_path.exists() or not file_path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Report file for '{report_id}' missing on disk.",
        )

    media_type = "text/html" if record.report_format.upper() == "HTML" else "text/markdown"

    return FileResponse(
        path=str(file_path),
        filename=file_path.name,
        media_type=media_type,
        content_disposition_type="inline" if inline else "attachment",
    )
