"""
AERIS Backend Data Transfer Objects (DTOs).

Defines strongly-typed Pydantic request and response models for all REST endpoints,
ensuring decoupling from internal core and storage models.
"""

from __future__ import annotations

import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

# ===========================================================================
# 1. Common & Error DTOs
# ===========================================================================


class ErrorResponse(BaseModel):
    """RFC 7807 problem details error response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    error_code: str = Field(description="Deterministic error machine code.")
    message: str = Field(description="Human-readable error description.")
    detail: str | None = Field(default=None, description="Optional diagnostic details.")
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="UTC timestamp when error occurred.",
    )
    request_id: str | None = Field(default=None, description="Optional request correlation ID.")


class StandardSuccessResponse(BaseModel):
    """Generic operation success response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    success: bool = Field(default=True, description="Whether the operation succeeded.")
    message: str = Field(description="Operation confirmation message.")


# ===========================================================================
# 2. System Health & Version DTOs
# ===========================================================================


class SystemHealthResponse(BaseModel):
    """Health status and subsystem availability response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    status: str = Field(default="healthy", description="Overall system health status.")
    database_connected: bool = Field(description="Whether metadata database is accessible.")
    active_jobs_count: int = Field(description="Number of currently running background jobs.")
    timestamp: datetime.datetime = Field(
        default_factory=lambda: datetime.datetime.now(datetime.UTC),
        description="UTC timestamp of health check.",
    )


class SystemVersionResponse(BaseModel):
    """AERIS version and runtime platform information."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    app_name: str = Field(default="AERIS", description="Application name.")
    version: str = Field(description="Semantic software release version.")
    python_version: str = Field(description="Host Python runtime version.")
    api_version: str = Field(default="v1", description="REST API contract version.")


# ===========================================================================
# 3. Project Management DTOs
# ===========================================================================


class CreateProjectRequest(BaseModel):
    """Request payload for creating a new flight test project."""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=120, description="Project title.")
    description: str = Field(default="", max_length=1000, description="Project scope or notes.")
    analyst_name: str | None = Field(
        default=None, max_length=100, description="Responsible engineer."
    )


class UpdateProjectRequest(BaseModel):
    """Request payload for modifying project metadata."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=120)
    description: str | None = Field(default=None, max_length=1000)
    analyst_name: str | None = Field(default=None, max_length=100)
    status: str | None = Field(default=None, description="ACTIVE, ARCHIVED, or DELETED.")


class ProjectDetailResponse(BaseModel):
    """Detailed metadata for a project including child resource counts."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    project_id: str
    name: str
    description: str
    analyst_name: str | None
    status: str
    created_at: datetime.datetime
    updated_at: datetime.datetime
    dataset_count: int = 0
    analysis_count: int = 0


class ProjectListResponse(BaseModel):
    """List response of active projects."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    projects: list[ProjectDetailResponse]
    total_count: int


# ===========================================================================
# 4. Dataset Ingestion & Parameter Mapping DTOs
# ===========================================================================


class DatasetDetailResponse(BaseModel):
    """Detailed metadata for an ingested telemetry dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    project_id: str
    original_filename: str
    file_size_bytes: int
    row_count: int
    column_count: int
    detected_columns: list[str]
    sha256_checksum: str
    ingested_at: datetime.datetime


class DatasetUploadResponse(BaseModel):
    """Response returned upon successful dataset upload."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    project_id: str
    original_filename: str
    file_size_bytes: int
    row_count: int
    column_count: int
    detected_columns: list[str]
    sha256_checksum: str
    message: str = "Dataset successfully uploaded and registered."


class DatasetListResponse(BaseModel):
    """List of telemetry datasets in a project."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    datasets: list[DatasetDetailResponse]
    total_count: int


class DatasetPreviewResponse(BaseModel):
    """Tabular preview of raw CSV telemetry data."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    total_rows: int
    columns: list[str]
    preview_rows: list[dict[str, Any]]


class ColumnMappingItem(BaseModel):
    """Single column-to-standard-parameter mapping entry."""

    model_config = ConfigDict(extra="forbid")

    raw_column_name: str = Field(description="Column header in raw telemetry CSV.")
    target_parameter_id: str = Field(description="Standardized AERIS parameter identifier.")
    source_unit: str | None = Field(default=None, description="Unit in raw telemetry.")
    target_unit: str | None = Field(default=None, description="Target normalized unit.")
    is_timestamp: bool = Field(default=False, description="Whether this column provides time.")
    data_type: str = Field(default="float", description="Numeric or string type.")


class SaveParameterMappingRequest(BaseModel):
    """Request payload to save column-to-parameter mappings."""

    model_config = ConfigDict(extra="forbid")

    mappings: list[ColumnMappingItem] = Field(min_length=1)


class ParameterMappingResponse(BaseModel):
    """Response containing verified parameter mappings for a dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    mappings: list[ColumnMappingItem]


# ===========================================================================
# 5. Configuration & Profile DTOs
# ===========================================================================


class BuiltinConfigItem(BaseModel):
    """Summary of built-in configuration profile."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    profile_type: str = Field(description="VEHICLE, LIMIT, or SOP.")
    profile_id: str
    name: str
    description: str


class BuiltinConfigsResponse(BaseModel):
    """List of built-in flight profiles and parameter registries."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    builtin_configs: list[BuiltinConfigItem]


class ConfigUploadResponse(BaseModel):
    """Response returned upon uploading custom configuration YAML."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    config_id: str
    project_id: str
    profile_type: str
    profile_identifier: str
    sha256_checksum: str
    loaded_at: datetime.datetime


class ProjectConfigsResponse(BaseModel):
    """List of custom uploaded configurations for a project."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    configs: list[ConfigUploadResponse]


# ===========================================================================
# 6. Analysis Execution & Background Job DTOs
# ===========================================================================


class StartAnalysisRequest(BaseModel):
    """Request payload to trigger an asynchronous analysis pipeline run."""

    model_config = ConfigDict(extra="forbid")

    dataset_id: str = Field(description="Telemetry dataset identifier to analyze.")
    vehicle_profile_id: str | None = Field(
        default=None, description="Optional vehicle configuration profile."
    )
    limit_profile_id: str | None = Field(
        default=None, description="Optional engineering limits profile."
    )
    procedure_profile_id: str | None = Field(
        default=None, description="Optional SOP procedure profile."
    )
    report_title: str = Field(
        default="AERIS Flight Test Analysis", max_length=150, description="Title for run."
    )


class JobQueuedResponse(BaseModel):
    """Response returned when an analysis job is accepted into the background queue."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    analysis_id: str
    state: str = "QUEUED"
    message: str = "Analysis job successfully queued."


class JobStatusResponse(BaseModel):
    """Response representing the real-time execution status of a background job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    job_id: str
    analysis_id: str
    state: str  # QUEUED, RUNNING, SUCCEEDED, PARTIAL, FAILED, CANCELLED
    current_stage: str | None
    completed_stages: int
    total_stages: int
    progress_percent: int
    error_message: str | None
    started_at: datetime.datetime
    updated_at: datetime.datetime


class AnalysisDetailResponse(BaseModel):
    """Detailed summary of a completed or executing analysis run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_id: str
    project_id: str
    dataset_id: str
    run_status: str  # PENDING, RUNNING, COMPLETED, FAILED, CANCELLED
    triggered_at: datetime.datetime
    completed_at: datetime.datetime | None
    execution_duration_ms: int | None
    total_issues: int
    error_count: int
    warning_count: int
    result_summary: dict[str, Any] | None = None


# ===========================================================================
# 7. Visualization DTOs
# ===========================================================================


class PlotlyFigureResponse(BaseModel):
    """Serialized Plotly figure structure and engineering metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    figure_json: dict[str, Any] = Field(description="JSON-serializable Plotly figure dictionary.")
    parameter_id: str
    unit: str | None = None
    epistemic_level: str = "CALCULATED"


class MultiSignalPlotRequest(BaseModel):
    """Request payload to generate a multi-signal panel visualization."""

    model_config = ConfigDict(extra="forbid")

    signal_ids: list[str] = Field(
        min_length=1, max_length=12, description="List of parameter IDs to include."
    )
    layout_mode: str = Field(default="STACKED", description="STACKED, OVERLAY, or UNIT_GROUPED.")
    show_limits: bool = Field(default=True, description="Render limit envelope overlays.")
    show_events: bool = Field(default=True, description="Render event marker overlays.")
    dark_mode: bool = Field(default=False, description="Apply dark theme.")


class AvailableSignalItem(BaseModel):
    """Signal metadata for UI selection and plotting."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    signal_id: str = Field(description="Signal identifier to pass to plot endpoints.")
    label: str = Field(description="Display label for dropdowns and legends.")
    unit: str | None = Field(default=None, description="Physical unit or dimension.")
    is_mapped: bool = Field(default=False, description="Whether signal is canonically mapped.")
    raw_column_name: str = Field(description="Underlying dataset column name.")


class AvailableSignalsResponse(BaseModel):
    """Available signals and recommended defaults for an analysis dataset."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset_id: str
    signals: list[AvailableSignalItem]
    default_signals: list[str]
    raw_columns: list[str]


# ===========================================================================
# 8. Report DTOs
# ===========================================================================


class GenerateReportRequest(BaseModel):
    """Request payload to generate an offline engineering report."""

    model_config = ConfigDict(extra="forbid")

    report_format: str = Field(default="HTML", description="Report format: HTML or MARKDOWN.")
    include_interactive_plots: bool = Field(
        default=True, description="Embed offline interactive Plotly figures in HTML."
    )
    report_title: str | None = Field(default=None, max_length=150)


class ReportDetailResponse(BaseModel):
    """Metadata representation of an exported engineering report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str
    analysis_id: str
    report_format: str
    report_title: str
    file_size_bytes: int
    sha256_checksum: str
    generated_at: datetime.datetime


class ReportListResponse(BaseModel):
    """List of generated engineering reports in a project."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    reports: list[ReportDetailResponse]
    total_count: int
