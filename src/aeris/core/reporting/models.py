"""
AERIS Engineering Report Domain Models.

Defines immutable, typed domain models representing the structured content of
post-flight engineering reports. The domain representation is decoupled from
presentation layers (Markdown, HTML, PDF) and presentation libraries.

Epistemic levels are strictly preserved across all sections:
- OBSERVED: Raw ingested telemetry, associated observations.
- CALCULATED: Derived metrics, statistics, anomalies, comparisons, correlations.
- LIMIT: Engineering boundaries, tolerance limits, envelopes.
- PROCEDURAL: SOP requirements, verification criteria.
- HYPOTHESIS: Analytical theories or candidate contributors.
- RECOMMENDATION: Bounded advisory proposals for engineering review.
"""

from __future__ import annotations

import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.results.types import Issue, ResultStatus


class StageStatus(StrEnum):
    """Execution status classification for individual pipeline stages in reports."""

    SUCCESS = "SUCCESS"
    """Stage completed cleanly with full validity."""

    WARNING = "WARNING"
    """Stage completed but advisories or non-critical deviations were flagged."""

    ERROR = "ERROR"
    """Stage encountered unrecoverable defects or failed."""

    PARTIAL = "PARTIAL"
    """Stage completed with partial coverage or degraded data."""

    NOT_AVAILABLE = "NOT_AVAILABLE"
    """Stage could not be evaluated due to missing prerequisites or scope limits."""

    SKIPPED = "SKIPPED"
    """Stage was not requested in the pipeline execution."""

    UPSTREAM_ABORTED = "UPSTREAM_ABORTED"
    """Stage was requested but skipped because an upstream prerequisite failed."""


class ReportMetadata(BaseModel):
    """Authoritative metadata describing an engineering report instance."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    report_id: str = Field(..., min_length=1, description="Deterministic or unique report ID.")
    title: str = Field(
        default="AERIS Post-Flight Engineering Report",
        min_length=1,
        description="Formal title of the engineering report.",
    )
    analyst_name: str | None = Field(
        default=None,
        description="Name or callsign of the analyzing engineer.",
    )
    generated_at: datetime.datetime = Field(
        ...,
        description="UTC timestamp of report generation.",
    )
    pipeline_run_id: str | None = Field(
        default=None,
        description="Identifier of the pipeline run or root provenance ID.",
    )
    source_file: str | None = Field(
        default=None,
        description="Path or URI of the source telemetry dataset.",
    )
    source_file_hash: str | None = Field(
        default=None,
        description="Cryptographic SHA-256 hash of the input dataset.",
    )
    execution_duration_seconds: float | None = Field(
        default=None,
        ge=0.0,
        description="Total pipeline execution duration in seconds.",
    )
    pipeline_version: str = Field(
        default="1.0.0",
        description="AERIS pipeline release version.",
    )
    custom_metadata: dict[str, Any] = Field(
        default_factory=dict,
        description="Auxiliary user-provided report metadata.",
    )


class StageExecutionSummary(BaseModel):
    """Summary of execution outcome for a specific pipeline stage."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage_name: str = Field(..., min_length=1, description="Canonical name of the pipeline stage.")
    stage_order: int = Field(..., ge=1, le=15, description="Execution sequence index (1-15).")
    status: StageStatus = Field(..., description="Stage execution outcome status.")
    is_executed: bool = Field(..., description="Whether the stage actually ran.")
    issue_count: int = Field(default=0, ge=0, description="Total issues logged by this stage.")
    error_count: int = Field(default=0, ge=0, description="ERROR or CRITICAL issues count.")
    warning_count: int = Field(default=0, ge=0, description="WARNING issues count.")
    has_errors: bool = Field(
        default=False, description="Whether stage produced error-level issues."
    )
    has_warnings: bool = Field(
        default=False, description="Whether stage produced warning-level issues."
    )
    notes: str | None = Field(default=None, description="Operational notes or failure reason.")


class ExecutiveSummary(BaseModel):
    """High-level executive summary of post-flight analysis findings."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall_status: ResultStatus = Field(..., description="Aggregated pipeline execution status.")
    total_stages: int = Field(default=15, ge=1, description="Total defined pipeline stages.")
    executed_stages_count: int = Field(default=0, ge=0, description="Number of executed stages.")
    failed_stages_count: int = Field(default=0, ge=0, description="Number of failed stages.")
    warning_stages_count: int = Field(default=0, ge=0, description="Number of warning stages.")
    skipped_stages_count: int = Field(
        default=0, ge=0, description="Number of skipped/unexecuted stages."
    )
    total_issues_count: int = Field(default=0, ge=0, description="Total issues across all stages.")
    total_critical_or_error_count: int = Field(
        default=0, ge=0, description="Total critical/error issues."
    )
    total_limit_exceedances: int = Field(
        default=0, ge=0, description="Number of limit exceedances."
    )
    total_procedure_violations: int = Field(
        default=0, ge=0, description="Number of procedure violations."
    )
    total_anomalies_detected: int = Field(
        default=0, ge=0, description="Number of detected anomalies."
    )
    total_recommendations_generated: int = Field(
        default=0, ge=0, description="Number of advisory recommendations."
    )
    high_level_narrative: str = Field(
        ..., min_length=1, description="Deterministic executive narrative."
    )


class DataQualitySummary(BaseModel):
    """Summary of data ingestion, validation, quality, and normalization outcomes."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_valid: bool | None = Field(
        default=None, description="Schema/data validation passed boolean."
    )
    quality_score: float | None = Field(
        default=None, ge=0.0, le=1.0, description="Aggregate quality score [0.0, 1.0]."
    )
    total_rows_ingested: int = Field(default=0, ge=0, description="Raw rows ingested from file.")
    total_columns_ingested: int = Field(default=0, ge=0, description="Raw columns detected.")
    columns_ingested: tuple[str, ...] = Field(
        default_factory=tuple, description="Column names ingested."
    )
    columns_evaluated: tuple[str, ...] = Field(
        default_factory=tuple, description="Columns validated."
    )
    violation_count: int = Field(default=0, ge=0, description="Schema/data validation violations.")
    quality_metrics: dict[str, Any] = Field(
        default_factory=dict, description="Detailed quality metrics dictionary."
    )
    quality_summary_text: str = Field(
        default="", description="High-level quality summary description."
    )
    normalization_applied: bool = Field(
        default=False, description="Whether normalization stage was executed."
    )
    transformations_count: int = Field(
        default=0, ge=0, description="Number of explicit transformations applied."
    )
    parameter_validation_passed: bool | None = Field(
        default=None, description="Semantic parameter validation outcome."
    )
    validated_parameters: tuple[str, ...] = Field(
        default_factory=tuple, description="Successfully validated parameter IDs."
    )
    failed_parameters: tuple[str, ...] = Field(
        default_factory=tuple, description="Parameters failing semantic validation."
    )
    issues: tuple[Issue, ...] = Field(
        default_factory=tuple, description="Data quality and validation issues."
    )


class FlightContextSummary(BaseModel):
    """Contextual flight test metadata, vehicle identification, and envelope configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    vehicle_id: str | None = Field(default=None, description="Evaluated vehicle identifier.")
    profile_id: str | None = Field(default=None, description="Vehicle profile identifier.")
    profile_version: str | None = Field(default=None, description="Vehicle profile version.")
    configuration_id: str | None = Field(
        default=None, description="Active vehicle configuration identifier."
    )
    test_card_id: str | None = Field(default=None, description="Test card operational context.")
    test_point_id: str | None = Field(default=None, description="Test point operational context.")
    observation_id: str | None = Field(
        default=None, description="Test point observation identifier."
    )
    associated_rows_count: int | None = Field(
        default=None, ge=0, description="Rows associated with test point interval."
    )
    configuration_discrepancies: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Any configuration discrepancies detected between data and envelope.",
    )


class LimitEvaluationItem(BaseModel):
    """Individual engineering limit evaluation record for the report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_id: str = Field(..., min_length=1, description="Evaluation record ID.")
    limit_id: str = Field(..., min_length=1, description="Limit rule identifier.")
    parameter_id: str = Field(..., min_length=1, description="Evaluated parameter.")
    outcome: str = Field(..., description="Evaluation outcome classification.")
    limit_type: str | None = Field(
        default=None, description="Limit type (e.g. UPPER, LOWER, BAND)."
    )
    severity: str | None = Field(default=None, description="Limit severity level.")
    observed_value: float | None = Field(default=None, description="Observed numeric value.")
    boundary_value: float | None = Field(default=None, description="Limit boundary threshold.")
    unit: str | None = Field(default=None, description="Engineering unit.")
    margin: float | None = Field(default=None, description="Signed margin to limit boundary.")
    is_exceeded: bool = Field(default=False, description="True if limit boundary was breached.")
    reason: str | None = Field(default=None, description="Explanatory outcome reason.")
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.LIMIT,
        description="Epistemic classification: LIMIT.",
    )


class ProcedureEvaluationItem(BaseModel):
    """Individual SOP procedure requirement evaluation record for the report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evaluation_id: str = Field(..., min_length=1, description="Evaluation record ID.")
    requirement_id: str = Field(..., min_length=1, description="Procedure requirement ID.")
    document_id: str = Field(..., min_length=1, description="Authoritative SOP document ID.")
    category: str = Field(..., description="Procedural category.")
    verification_status: str = Field(..., description="Verification status.")
    outcome: str = Field(..., description="Evaluation outcome classification.")
    observed_value: float | None = Field(default=None, description="Observed value.")
    target_value: float | None = Field(default=None, description="Target value.")
    unit: str | None = Field(default=None, description="Engineering unit.")
    margin: float | None = Field(default=None, description="Margin to criteria.")
    is_satisfied: bool | None = Field(
        default=None, description="True if satisfied, False if violated."
    )
    is_qualitative: bool = Field(default=False, description="Whether requirement is qualitative.")
    qualitative_state: str | None = Field(default=None, description="Observed qualitative state.")
    reason: str = Field(..., description="Deterministic reason for verification outcome.")
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.PROCEDURAL,
        description="Epistemic classification: PROCEDURAL.",
    )


class EngineeringEvaluationSummary(BaseModel):
    """Summary of vehicle engineering evaluation, limits, procedures, and calculations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    is_evaluated: bool = Field(
        default=False, description="Whether vehicle evaluation was executed."
    )
    total_limits_evaluated: int = Field(default=0, ge=0, description="Total limit evaluations.")
    limit_exceedance_count: int = Field(default=0, ge=0, description="Number of limit exceedances.")
    limit_inconclusive_count: int = Field(
        default=0, ge=0, description="Inconclusive limit evaluations."
    )
    total_procedures_evaluated: int = Field(
        default=0, ge=0, description="Total procedure evaluations."
    )
    procedure_violation_count: int = Field(
        default=0, ge=0, description="Number of procedure violations."
    )
    procedure_inconclusive_count: int = Field(
        default=0, ge=0, description="Inconclusive procedure evaluations."
    )
    limit_evaluations: tuple[LimitEvaluationItem, ...] = Field(default_factory=tuple)
    procedure_evaluations: tuple[ProcedureEvaluationItem, ...] = Field(default_factory=tuple)
    calculations_performed: tuple[str, ...] = Field(
        default_factory=tuple,
        description="Identifiers of engineering calculations executed.",
    )


class AnomalyItem(BaseModel):
    """Individual anomaly detection record for the report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    anomaly_id: str = Field(..., min_length=1, description="Anomaly identifier.")
    analysis_id: str = Field(..., min_length=1, description="Originating analysis identifier.")
    target_column: str = Field(..., min_length=1, description="Affected telemetry parameter.")
    classification: str = Field(..., description="Classification (POINT, CONTEXTUAL, COLLECTIVE).")
    severity: str = Field(..., description="Severity classification (INFO, LOW, MEDIUM, HIGH).")
    detection_method: str = Field(..., description="Detection methodology.")
    timestamp: datetime.datetime | None = Field(
        default=None, description="Timestamp of occurrence."
    )
    observed_value: float | str | None = Field(
        default=None, description="Observed anomalous value."
    )
    expected_value: float | str | None = Field(default=None, description="Expected baseline value.")
    deviation_value: float | None = Field(default=None, description="Numerical deviation.")
    explanation: str | None = Field(default=None, description="Analytical explanation.")
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.CALCULATED,
        description="Epistemic classification: CALCULATED.",
    )


class ExpectedObservedComparisonItem(BaseModel):
    """Individual expected-vs-observed comparison item for the report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    comparison_id: str = Field(..., min_length=1, description="Comparison identifier.")
    parameter_id: str = Field(..., min_length=1, description="Observed parameter identifier.")
    expected_value: float | str = Field(..., description="Expected value.")
    observed_value: float | str = Field(..., description="Observed value.")
    deviation: float | None = Field(default=None, description="Calculated deviation.")
    tolerance_evaluated: bool = Field(default=False, description="Whether tolerance was evaluated.")
    within_tolerance: bool | None = Field(default=None, description="Whether within tolerance.")
    status: ResultStatus = Field(..., description="Comparison outcome status.")
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.CALCULATED,
        description="Epistemic classification: CALCULATED.",
    )


class AnomalyAndComparisonSummary(BaseModel):
    """Summary of detected anomalies and expected-vs-observed deviations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    total_anomalies: int = Field(default=0, ge=0, description="Total detected anomalies.")
    high_severity_anomalies: int = Field(
        default=0, ge=0, description="HIGH severity anomaly count."
    )
    medium_severity_anomalies: int = Field(
        default=0, ge=0, description="MEDIUM severity anomaly count."
    )
    low_severity_anomalies: int = Field(default=0, ge=0, description="LOW severity anomaly count.")
    info_severity_anomalies: int = Field(
        default=0, ge=0, description="INFO severity anomaly count."
    )
    anomalies: tuple[AnomalyItem, ...] = Field(default_factory=tuple)
    total_comparisons: int = Field(
        default=0, ge=0, description="Total expected-observed comparisons."
    )
    comparisons_within_tolerance: int = Field(
        default=0, ge=0, description="Comparisons within tolerance."
    )
    comparisons_out_of_tolerance: int = Field(
        default=0, ge=0, description="Comparisons breaching tolerance."
    )
    comparisons: tuple[ExpectedObservedComparisonItem, ...] = Field(default_factory=tuple)


class EvidenceSummaryItem(BaseModel):
    """Structured evidence item extracted for investigation tracking."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_id: str = Field(..., min_length=1, description="Evidence identifier.")
    evidence_type: str = Field(..., description="Evidence type classification.")
    description: str = Field(..., min_length=1, description="Factual description.")
    source_reference: str = Field(..., min_length=1, description="Originating source URI/file.")
    parameter_id: str | None = Field(default=None, description="Associated parameter ID.")
    observed_value: float | None = Field(default=None, description="Associated observed value.")
    margin: float | None = Field(default=None, description="Margin if applicable.")
    epistemic_level: EpistemicLevel = Field(..., description="Epistemic level of this evidence.")


class ContributorItem(BaseModel):
    """Statistical contributor correlation or association ranking item."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis_id: str = Field(..., min_length=1, description="Contributor analysis ID.")
    target_column: str = Field(..., min_length=1, description="Target parameter.")
    factor_column: str = Field(
        ..., min_length=1, description="Potential contributing factor parameter."
    )
    method: str = Field(..., description="Statistical correlation method.")
    coefficient: float | None = Field(default=None, description="Correlation coefficient.")
    p_value: float | None = Field(default=None, description="Statistical significance p-value.")
    status: ResultStatus = Field(..., description="Analysis status.")
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.CALCULATED,
        description="Epistemic classification: CALCULATED.",
    )


class RecommendationItem(BaseModel):
    """Advisory recommendation proposal for post-flight engineering investigation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    recommendation_id: str = Field(..., min_length=1, description="Recommendation identifier.")
    recommendation_type: str = Field(..., description="Actionable recommendation type.")
    title: str = Field(..., min_length=1, description="Recommendation title.")
    description: str = Field(..., min_length=1, description="Recommendation detailed description.")
    rationale: str = Field(..., min_length=1, description="Engineering rationale.")
    status: str = Field(..., description="Lifecycle status.")
    evidence_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="Traceable evidence IDs."
    )
    rule_id: str | None = Field(
        default=None, description="Deterministic rule that generated this recommendation."
    )
    advisory_notice: str = Field(
        default="FOR ENGINEERING REVIEW ONLY — NOT AN OPERATIONAL OR AIRWORTHINESS DIRECTIVE",
        description="Mandatory advisory notice indicating non-autonomous review requirement.",
    )
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.RECOMMENDATION,
        description="Epistemic classification: RECOMMENDATION.",
    )


class InvestigationIntelligenceSummary(BaseModel):
    """Summary of investigation evidence, contributor rankings, and generated recommendations."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    evidence_count: int = Field(default=0, ge=0, description="Total evidence items constructed.")
    evidence_items: tuple[EvidenceSummaryItem, ...] = Field(default_factory=tuple)
    contributor_rankings: tuple[ContributorItem, ...] = Field(default_factory=tuple)
    recommendation_count: int = Field(
        default=0, ge=0, description="Total recommendations generated."
    )
    rules_evaluated: tuple[str, ...] = Field(default_factory=tuple, description="Rules evaluated.")
    rules_triggered: tuple[str, ...] = Field(default_factory=tuple, description="Rules triggered.")
    recommendations: tuple[RecommendationItem, ...] = Field(default_factory=tuple)


class LimitationItem(BaseModel):
    """Explicit analytical limitation, missing prerequisite, or data gap record."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    limitation_id: str = Field(..., min_length=1, description="Limitation identifier.")
    category: str = Field(..., description="Limitation category.")
    stage_name: str | None = Field(default=None, description="Affected pipeline stage.")
    description: str = Field(..., min_length=1, description="Description of the limitation or gap.")
    impact: str = Field(..., min_length=1, description="Engineering impact of the limitation.")


class LimitationsSummary(BaseModel):
    """Summary of analytical caveats, unexecuted stages, and data gaps."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    has_limitations: bool = Field(
        default=False, description="True if any limitations or gaps exist."
    )
    unexecuted_stages: tuple[str, ...] = Field(
        default_factory=tuple, description="Stages not executed."
    )
    failed_stages: tuple[str, ...] = Field(
        default_factory=tuple, description="Stages that failed with errors."
    )
    inconclusive_checks_count: int = Field(
        default=0, ge=0, description="Checks resulting in INCONCLUSIVE."
    )
    items: tuple[LimitationItem, ...] = Field(
        default_factory=tuple, description="Specific limitation items."
    )


class ProvenanceSummary(BaseModel):
    """Audit trail and lineage metadata for the entire analysis run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    root_provenance_id: str | None = Field(
        default=None, description="Root pipeline provenance record ID."
    )
    source_uri: str | None = Field(default=None, description="Source dataset URI or file path.")
    activity_name: str | None = Field(default=None, description="Root activity name.")
    epistemic_level: EpistemicLevel = Field(
        default=EpistemicLevel.CALCULATED,
        description="Root epistemic level.",
    )
    parent_record_ids: tuple[str, ...] = Field(
        default_factory=tuple, description="Parent provenance record IDs."
    )
    stage_provenance_records: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of stage_name -> provenance record_id.",
    )


class EngineeringReport(BaseModel):
    """
    Authoritative, immutable post-flight engineering report domain model.

    Aggregates all extracted stage summaries, data quality metrics, vehicle
    evaluations, anomalies, evidence, contributor correlations, advisory
    recommendations, and analytical limitations.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    metadata: ReportMetadata = Field(..., description="Report metadata and run configuration.")
    executive_summary: ExecutiveSummary = Field(..., description="High-level executive summary.")
    stage_summaries: tuple[StageExecutionSummary, ...] = Field(
        default_factory=tuple,
        description="Ordered execution summaries for all 15 stages.",
    )
    data_quality: DataQualitySummary = Field(
        ..., description="Data quality and validation section."
    )
    flight_context: FlightContextSummary = Field(..., description="Vehicle flight context section.")
    engineering_evaluation: EngineeringEvaluationSummary = Field(
        ..., description="Limits and SOP section."
    )
    anomaly_and_comparison: AnomalyAndComparisonSummary = Field(
        ..., description="Anomalies and comparisons."
    )
    investigation_intelligence: InvestigationIntelligenceSummary = Field(
        ..., description="Intelligence section."
    )
    limitations: LimitationsSummary = Field(..., description="Limitations, caveats, and data gaps.")
    provenance: ProvenanceSummary = Field(..., description="Audit trail and lineage.")
    all_issues: tuple[Issue, ...] = Field(
        default_factory=tuple, description="All logged issues across stages."
    )
