"""
AERIS Backend Datasets & Parameter Mappings Router.

Handles raw telemetry CSV uploads, tabular previews, and column-to-parameter mappings.
"""

from __future__ import annotations

import csv
import uuid

from fastapi import APIRouter, Depends, File, HTTPException, Query, UploadFile, status

from aeris.backend.deps import get_dataset_store, get_repository
from aeris.backend.models import (
    ColumnMappingItem,
    DatasetDetailResponse,
    DatasetListResponse,
    DatasetPreviewResponse,
    DatasetUploadResponse,
    ParameterMappingResponse,
    SaveParameterMappingRequest,
)
from aeris.core.registry.canonical import suggest_canonical_parameter
from aeris.storage.dataset_store import ManagedDatasetStore
from aeris.storage.exceptions import DatasetNotFoundError, ProjectNotFoundError
from aeris.storage.models import ParameterMappingRecord
from aeris.storage.repository import SQLiteMetadataRepository

router = APIRouter(prefix="/projects/{project_id}/datasets", tags=["Datasets"])


@router.post("/upload", response_model=DatasetUploadResponse, status_code=status.HTTP_201_CREATED)
async def upload_dataset(
    project_id: str,
    file: UploadFile = File(...),
    dataset_store: ManagedDatasetStore = Depends(get_dataset_store),
) -> DatasetUploadResponse:
    """Stream raw CSV telemetry file into managed byte-preserving storage."""
    original_filename = file.filename or "telemetry.csv"

    record, _path = dataset_store.store_dataset(
        project_id=project_id,
        source=file.file,
        original_filename=original_filename,
    )

    return DatasetUploadResponse(
        dataset_id=record.dataset_id,
        project_id=record.project_id,
        original_filename=record.original_filename,
        file_size_bytes=record.file_size_bytes,
        row_count=record.row_count,
        column_count=record.column_count,
        detected_columns=record.detected_columns,
        sha256_checksum=record.sha256_checksum,
    )


@router.get("", response_model=DatasetListResponse)
async def list_datasets(
    project_id: str,
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> DatasetListResponse:
    """List all ingested datasets in a project."""
    project = repository.get_project(project_id)
    if project is None:
        raise ProjectNotFoundError(f"Project '{project_id}' not found.")

    datasets = repository.list_datasets(project_id)
    items = [
        DatasetDetailResponse(
            dataset_id=d.dataset_id,
            project_id=d.project_id,
            original_filename=d.original_filename,
            file_size_bytes=d.file_size_bytes,
            row_count=d.row_count,
            column_count=d.column_count,
            detected_columns=d.detected_columns,
            sha256_checksum=d.sha256_checksum,
            ingested_at=d.ingested_at,
        )
        for d in datasets
    ]
    return DatasetListResponse(datasets=items, total_count=len(items))


@router.get("/{dataset_id}/preview", response_model=DatasetPreviewResponse)
async def preview_dataset(
    project_id: str,
    dataset_id: str,
    rows: int = Query(default=50, ge=1, le=500),
    repository: SQLiteMetadataRepository = Depends(get_repository),
    dataset_store: ManagedDatasetStore = Depends(get_dataset_store),
) -> DatasetPreviewResponse:
    """Preview first N rows of a managed telemetry dataset."""
    record = repository.get_dataset(dataset_id)
    if record is None or record.project_id != project_id:
        raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found in project '{project_id}'.")

    file_path = dataset_store.get_dataset_path(dataset_id)
    preview_rows: list[dict[str, str]] = []

    try:
        with file_path.open("r", encoding="utf-8", errors="replace", newline="") as f:
            reader = csv.DictReader(f)
            columns = [col.strip() for col in (reader.fieldnames or [])]
            for i, row in enumerate(reader):
                if i >= rows:
                    break
                preview_rows.append(row)
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed reading dataset preview: {exc}",
        ) from exc

    return DatasetPreviewResponse(
        dataset_id=dataset_id,
        total_rows=record.row_count,
        columns=columns,
        preview_rows=preview_rows,
    )


@router.get("/{dataset_id}/mappings", response_model=ParameterMappingResponse)
async def get_parameter_mappings(
    project_id: str,
    dataset_id: str,
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> ParameterMappingResponse:
    """Retrieve saved or auto-suggested parameter mappings for a dataset."""
    record = repository.get_dataset(dataset_id)
    if record is None or record.project_id != project_id:
        raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found in project '{project_id}'.")

    saved_mappings = repository.get_parameter_mappings(dataset_id)
    if saved_mappings:
        items = [
            ColumnMappingItem(
                raw_column_name=m.raw_column_name,
                target_parameter_id=m.target_parameter_id,
                source_unit=m.source_unit,
                target_unit=m.target_unit,
                is_timestamp=bool(m.is_timestamp),
                data_type=m.data_type,
            )
            for m in saved_mappings
        ]
    else:
        # Auto-suggest canonical or normalized mappings based on detected columns
        items = []
        for col in record.detected_columns:
            canonical_id, unit, is_time = suggest_canonical_parameter(col)
            items.append(
                ColumnMappingItem(
                    raw_column_name=col,
                    target_parameter_id=canonical_id or col.lower().replace(" ", "_"),
                    source_unit=unit,
                    target_unit=unit,
                    is_timestamp=is_time,
                    data_type="float",
                )
            )

    return ParameterMappingResponse(dataset_id=dataset_id, mappings=items)


@router.put("/{dataset_id}/mappings", response_model=ParameterMappingResponse)
async def save_parameter_mappings(
    project_id: str,
    dataset_id: str,
    request: SaveParameterMappingRequest,
    repository: SQLiteMetadataRepository = Depends(get_repository),
) -> ParameterMappingResponse:
    """Save verified column-to-standard-parameter mappings for a dataset."""
    record = repository.get_dataset(dataset_id)
    if record is None or record.project_id != project_id:
        raise DatasetNotFoundError(f"Dataset '{dataset_id}' not found in project '{project_id}'.")

    records = [
        ParameterMappingRecord(
            mapping_id=f"map_{uuid.uuid4().hex[:12]}",
            dataset_id=dataset_id,
            raw_column_name=m.raw_column_name.strip(),
            target_parameter_id=m.target_parameter_id.strip(),
            source_unit=m.source_unit.strip() if m.source_unit else None,
            target_unit=m.target_unit.strip() if m.target_unit else None,
            data_type=m.data_type,
            is_timestamp=bool(m.is_timestamp),
        )
        for m in request.mappings
    ]

    repository.save_parameter_mappings(records)

    return ParameterMappingResponse(dataset_id=dataset_id, mappings=request.mappings)
