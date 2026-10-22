"""
AERIS Storage Domain Models.

Defines immutable, typed Pydantic models representing stored entities:
projects, datasets, parameter mappings, configurations, analysis runs,
background jobs, and exported reports.
"""

from __future__ import annotations

import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class ProjectStatus(StrEnum):
    """Lifecycle status of an AERIS analysis project."""

    ACTIVE = "ACTIVE"
    ARCHIVED = "ARCHIVED"
    DELETED = "DELETED"


class ProjectRecord(BaseModel):
    """Metadata representation of an AERIS project."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str = Field(description="Unique identifier for the project.")
    name: str = Field(description="Human-readable project title.")
    description: str = Field(default="", description="Detailed project description.")
    analyst_name: str | None = Field(default=None, description="Responsible flight test engineer.")
    created_at: datetime.datetime = Field(description="UTC timestamp when created.")
    updated_at: datetime.datetime = Field(description="UTC timestamp when last modified.")
    status: ProjectStatus = Field(default=ProjectStatus.ACTIVE, description="Lifecycle status.")


class DatasetRecord(BaseModel):
    """Metadata representation of an ingested flight telemetry dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str = Field(description="Unique identifier for the dataset.")
    project_id: str = Field(description="Parent project identifier.")
    original_filename: str = Field(description="Original source filename uploaded by engineer.")
    storage_relative_path: str = Field(
        description="Path relative to workspace root where file is stored."
    )
    sha256_checksum: str = Field(description="SHA-256 cryptographic digest of file bytes.")
    file_size_bytes: int = Field(ge=0, description="Size of file in bytes.")
    row_count: int = Field(ge=0, description="Total number of telemetry rows.")
    column_count: int = Field(ge=0, description="Total number of columns.")
    detected_columns: list[str] = Field(
        default_factory=list, description="Ordered list of detected column names."
    )
    ingested_at: datetime.datetime = Field(description="UTC timestamp of ingestion.")


class ParameterMappingRecord(BaseModel):
    """Explicit mapping from raw telemetry column to semantic parameter identifier."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mapping_id: str = Field(description="Unique mapping record ID.")
    dataset_id: str = Field(description="Associated dataset identifier.")
    raw_column_name: str = Field(description="Header name in raw CSV file.")
    target_parameter_id: str = Field(description="Authoritative semantic parameter identifier.")
    source_unit: str | None = Field(default=None, description="Physical unit in raw file.")
    target_unit: str | None = Field(default=None, description="Target canonical engineering unit.")
    data_type: str = Field(default="float", description="Expected parameter data type.")
    is_timestamp: bool = Field(
        default=False, description="Whether this column is the primary time."
    )


class ConfigurationProfileRecord(BaseModel):
    """Reference to a vehicle, limit, envelope, or SOP YAML profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_id: str = Field(description="Unique configuration profile ID.")
    project_id: str = Field(description="Associated project identifier.")
    profile_type: str = Field(description="Type of profile (VEHICLE, ENVELOPE, LIMITS, SOP).")
    profile_identifier: str = Field(description="Domain identifier of profile.")
    storage_relative_path: str = Field(
        description="Path relative to workspace root where YAML is stored."
    )
    sha256_checksum: str = Field(description="SHA-256 checksum of YAML file.")
    loaded_at: datetime.datetime = Field(description="UTC timestamp when attached.")


class AnalysisRunRecord(BaseModel):
    """Metadata record of an executed 15-stage analysis run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_id: str = Field(description="Unique analysis run identifier.")
    project_id: str = Field(description="Associated project identifier.")
    dataset_id: str = Field(description="Source dataset identifier.")
    run_status: str = Field(
        description=(
            "Terminal or current execution status "
            "(SUCCESS, WARNING, ERROR, PARTIAL, ABORTED, CANCELLED)."
        )
    )
    triggered_at: datetime.datetime = Field(description="UTC timestamp when run was initiated.")
    completed_at: datetime.datetime | None = Field(
        default=None, description="UTC timestamp of completion."
    )
    execution_duration_ms: int | None = Field(
        default=None, ge=0, description="Total execution duration in milliseconds."
    )
    total_issues_count: int = Field(default=0, ge=0, description="Total diagnostic issues found.")
    error_count: int = Field(default=0, ge=0, description="Total ERROR severity issues.")
    warning_count: int = Field(default=0, ge=0, description="Total WARNING severity issues.")
    result_summary: dict[str, Any] | None = Field(
        default=None, description="Compact summary dictionary of run results."
    )
    provenance_id: str | None = Field(
        default=None, description="Root provenance record ID linking results."
    )


class BackgroundJobRecord(BaseModel):
    """State tracking record for background asynchronous analysis jobs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str = Field(description="Unique background job identifier.")
    analysis_id: str = Field(description="Associated analysis run identifier.")
    state: str = Field(
        description="Job state (QUEUED, RUNNING, SUCCEEDED, PARTIAL, FAILED, CANCELLED)."
    )
    current_stage: str | None = Field(
        default=None, description="Name of the currently executing pipeline stage."
    )
    completed_stages_count: int = Field(
        default=0, ge=0, le=15, description="Number of completed pipeline stages."
    )
    total_stages_count: int = Field(default=15, ge=1, description="Total defined pipeline stages.")
    progress_percent: int = Field(
        default=0, ge=0, le=100, description="Current milestone progress percent (0-100)."
    )
    error_message: str | None = Field(default=None, description="Error diagnostics if job failed.")
    created_at: datetime.datetime = Field(description="UTC timestamp when job was queued.")
    updated_at: datetime.datetime = Field(description="UTC timestamp when status last updated.")


class ExportedReportRecord(BaseModel):
    """Metadata record of an exported engineering report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str = Field(description="Unique report identifier.")
    analysis_id: str = Field(description="Source analysis run identifier.")
    report_format: str = Field(description="Format of exported report (HTML, MARKDOWN).")
    report_title: str = Field(description="Title of the report.")
    storage_relative_path: str = Field(
        description="Path relative to workspace root where report is stored."
    )
    sha256_checksum: str = Field(description="SHA-256 checksum of generated report.")
    file_size_bytes: int = Field(ge=0, description="Size of generated report in bytes.")
    generated_at: datetime.datetime = Field(description="UTC timestamp when generated.")


class WorkspaceIntegrityReport(BaseModel):
    """Diagnostic outcome of a workspace reconciliation check."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    checked_at: datetime.datetime = Field(description="UTC timestamp of reconciliation.")
    clean_datasets_count: int = Field(ge=0, description="Count of datasets verified intact.")
    missing_files: list[str] = Field(
        default_factory=list, description="Dataset IDs present in DB but missing on disk."
    )
    unindexed_files: list[str] = Field(
        default_factory=list, description="Relative paths on disk not indexed in database."
    )
    corrupted_files: list[str] = Field(
        default_factory=list, description="Dataset IDs where file on disk hash does not match DB."
    )
    stale_temporary_files: list[str] = Field(
        default_factory=list, description="Stale .aeris_tmp_* files detected."
    )

    @property
    def is_healthy(self) -> bool:
        """Return True if workspace has zero missing, corrupted, unindexed, or temporary files."""
        return (
            len(self.missing_files) == 0
            and len(self.unindexed_files) == 0
            and len(self.corrupted_files) == 0
            and len(self.stale_temporary_files) == 0
        )
