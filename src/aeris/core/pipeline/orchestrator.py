"""
AERIS Core Pipeline Orchestration.

Manages the end-to-end execution of Ingestion, Validation, and Quality
assessment stages for data processing.
"""

from __future__ import annotations

import datetime
import pathlib
import uuid
from typing import TYPE_CHECKING, Any

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

if TYPE_CHECKING:
    from aeris.core.reporting.exporter import ReportExportResult

from aeris.core.anomaly.models import PipelineAnomalyDetectionResult
from aeris.core.anomaly.statistical import (
    StatisticalAnomalyDetectionEngine,
    StatisticalAnomalyDetectionRequest,
)
from aeris.core.anomaly.threshold import ThresholdAnomalyDetectionEngine, ThresholdDetectionRequest
from aeris.core.anomaly.timeseries import (
    TimeSeriesAnomalyDetectionEngine,
    TimeSeriesAnomalyDetectionRequest,
)
from aeris.core.contributors.engine import ContributorAnalysisEngine
from aeris.core.contributors.pipeline import (
    PipelineContributorAnalysisRequest,
    PipelineContributorAnalysisResult,
)
from aeris.core.engineering.executor import EngineeringExecutor
from aeris.core.engineering.models import (
    EngineeringCalculationDefinition,
    EngineeringCalculationResult,
)
from aeris.core.evidence.engine import PipelineEvidenceEngine
from aeris.core.evidence.pipeline import PipelineEvidenceRequest, PipelineEvidenceResult
from aeris.core.expected_observed.engine import ExpectedObservedEngine
from aeris.core.expected_observed.models import (
    ExpectedObservedComparison,
    ObservedValue,
    PipelineExpectedObservedRequest,
    PipelineExpectedObservedResult,
)
from aeris.core.ingestion.csv_ingestor import ingest_csv
from aeris.core.normalization.engine import (
    NormalizationRequest,
    normalize_flight_data,
)
from aeris.core.parameter_validation.validator import ParameterValidationRequest
from aeris.core.provenance.tracker import EpistemicLevel, ProvenanceRecord, create_provenance
from aeris.core.quality.quality_engine import assess_data_quality
from aeris.core.recommendation.generation_engine import RecommendationGenerationEngine
from aeris.core.recommendation.generation_models import (
    RecommendationGenerationPolicy,
    RecommendationGenerationRequest,
    RecommendationGenerationResult,
)
from aeris.core.recommendation.models import RecommendationCollection
from aeris.core.recommendation.pipeline import PipelineRecommendationRequest
from aeris.core.registry import ParameterRegistry
from aeris.core.results.types import (
    DataPipelineResult,
    Issue,
    IssueSeverity,
    ParameterValidationResult,
    ResultStatus,
)
from aeris.core.statistics.engine import StatisticalAnalysisEngine
from aeris.core.statistics.models import StatisticalAnalysisRequest, StatisticalAnalysisResult
from aeris.core.statistics.time_series.engine import TimeSeriesStatisticalAnalysisEngine
from aeris.core.statistics.time_series.models import (
    TimeSeriesAnalysisRequest,
    TimeSeriesAnalysisResult,
)
from aeris.core.test_operations.association import (
    TestPointAssociationEngine,
    TestPointAssociationResult,
)
from aeris.core.test_operations.test_point_observation import TestPointObservation
from aeris.core.validation.validators import validate_flight_data
from aeris.core.vehicle_evaluation.engine import (
    VehicleEngineeringEvaluationEngine,
)
from aeris.core.vehicle_evaluation.models import (
    EvaluationObservation,
    VehicleEvaluationRequest,
    VehicleEvaluationResult,
)
from aeris.core.vehicle_evaluation.pipeline import (
    PipelineVehicleEvaluationRequest,
)


class PipelineAnomalyDetectionRequest(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")
    threshold_requests: tuple[ThresholdDetectionRequest, ...] = Field(default_factory=tuple)
    statistical_requests: tuple[StatisticalAnomalyDetectionRequest, ...] = Field(
        default_factory=tuple
    )
    time_series_requests: tuple[TimeSeriesAnomalyDetectionRequest, ...] = Field(
        default_factory=tuple
    )


class TestPointAssociationRequest(BaseModel):
    """Explicit request model for test point temporal association."""

    __test__ = False

    model_config = ConfigDict(frozen=True, extra="forbid")
    observation: TestPointObservation
    timestamp_column: str


class EngineeringCalculationRequest(BaseModel):
    """Explicit request model for engineering calculation execution."""

    model_config = ConfigDict(frozen=True, extra="forbid")
    calculation_definition: EngineeringCalculationDefinition
    parent_record_ids: list[str] | None = None


class DataPipelineOrchestrator:
    """Orchestrates ingestion, validation, and quality assessment stages."""

    def __init__(self) -> None:
        pass

    def run_csv_pipeline(
        self,
        file_path: str | pathlib.Path,
        normalization_request: NormalizationRequest | None = None,
        parameter_registry: ParameterRegistry | None = None,
        parameter_validation_request: ParameterValidationRequest | None = None,
        association_request: TestPointAssociationRequest | None = None,
        engineering_request: EngineeringCalculationRequest | None = None,
        statistics_request: StatisticalAnalysisRequest | None = None,
        time_series_request: TimeSeriesAnalysisRequest | None = None,
        anomaly_request: PipelineAnomalyDetectionRequest | None = None,
        expected_observed_request: PipelineExpectedObservedRequest | None = None,
        evidence_request: PipelineEvidenceRequest | None = None,
        vehicle_evaluation_request: PipelineVehicleEvaluationRequest | None = None,
        contributor_analysis_request: PipelineContributorAnalysisRequest | None = None,
        recommendation_request: (
            PipelineRecommendationRequest | RecommendationGenerationRequest | None
        ) = None,
    ) -> DataPipelineResult:
        """Execute the CSV processing pipeline.

        Parameters
        ----------
        file_path : str | pathlib.Path
            Path to the input CSV file.
        normalization_request : NormalizationRequest | None
            Optional explicit request for normalizations. If None, normalization is skipped.

        Returns
        -------
        DataPipelineResult
            Aggregated result encompassing ingestion, validation, quality,
            and optionally normalization.
        """
        resolved_path = str(file_path)
        parent_record_ids = []

        # 1. Ingestion Stage
        ingest_res = ingest_csv(resolved_path)
        if ingest_res.provenance:
            parent_record_ids.append(ingest_res.provenance.record_id)

        # If ingestion entirely fails (no data), we degrade gracefully.
        if ingest_res.status == ResultStatus.ERROR or ingest_res.data is None:
            prov = create_provenance(
                source_uri=resolved_path,
                activity_name="csv_data_pipeline",
                epistemic_level=EpistemicLevel.CALCULATED,
                parent_record_ids=parent_record_ids,
            )

            # Generate empty association if requested
            assoc_res = None
            eng_res = None
            if association_request is not None:
                prov_err = create_provenance(
                    source_uri=resolved_path,
                    activity_name="test_point_association",
                    epistemic_level=EpistemicLevel.CALCULATED,
                    parent_record_ids=parent_record_ids,
                )
                assoc_res = TestPointAssociationResult(
                    status=ResultStatus.ERROR,
                    issues=[
                        Issue(
                            issue_code="ERR_ASSOCIATION_DATA_UNAVAILABLE",
                            severity=IssueSeverity.ERROR,
                            message="Association requested but no valid DataFrame is available.",
                            location="DataPipelineOrchestrator",
                        )
                    ],
                    provenance=prov_err,
                    test_point_id=association_request.observation.test_point_id,
                    observation_id=association_request.observation.observation_id,
                    timestamp_column=association_request.timestamp_column,
                    row_count=0,
                    data=None,
                )

            return DataPipelineResult(
                status=ResultStatus.ERROR,
                issues=ingest_res.issues
                + (assoc_res.issues if assoc_res else [])
                + (eng_res.issues if eng_res else []),
                provenance=prov,
                ingestion=ingest_res,
                validation=None,
                quality=None,
                normalization=None,
                association=assoc_res,
                engineering=eng_res,
            )

        # 2. Validation Stage
        val_res = validate_flight_data(ingest_res)
        if val_res.provenance:
            parent_record_ids.append(val_res.provenance.record_id)

        # 3. Quality Stage
        qual_res = assess_data_quality(ingest_res, validation_result=val_res)
        if qual_res.provenance:
            parent_record_ids.append(qual_res.provenance.record_id)

        # 4. Optional Normalization Stage
        norm_res = None
        if normalization_request is not None:
            # We attempt normalization even if validation flagged ERRORs.
            # Normalization engine handles bad schemas internally (e.g., missing requested columns)
            norm_res = normalize_flight_data(
                ingest_res,
                request=normalization_request,
                registry=parameter_registry,
                source_uri=resolved_path,
            )
            if norm_res.provenance:
                parent_record_ids.append(norm_res.provenance.record_id)

        # 5. Optional Parameter Validation Stage
        param_val_res = None
        if parameter_validation_request is not None:
            if parameter_registry is None:
                # Validation request provided without a registry
                prov_err = create_provenance(
                    source_uri=resolved_path,
                    activity_name="parameter_semantic_validation",
                    epistemic_level=EpistemicLevel.CALCULATED,
                    parent_record_ids=parent_record_ids,
                )
                param_val_res = ParameterValidationResult(
                    status=ResultStatus.ERROR,
                    issues=[
                        Issue(
                            issue_code="ERR_PARAMETER_VALIDATION_REGISTRY_REQUIRED",
                            severity=IssueSeverity.ERROR,
                            message="Parameter validation requested but no ParameterRegistry "
                            "provided.",
                            location="DataPipelineOrchestrator",
                        )
                    ],
                    provenance=prov_err,
                    validated_parameters=[],
                    failed_parameters=[],
                )
                parent_record_ids.append(prov_err.record_id)
            else:
                # Both registry and request provided
                from aeris.core.parameter_validation.validator import ParameterValidator

                validator = ParameterValidator(parameter_registry)
                df_to_validate = (
                    norm_res.data
                    if norm_res is not None and norm_res.data is not None
                    else ingest_res.data
                )
                param_val_res = validator.validate(
                    df_to_validate, parameter_validation_request, source_uri=resolved_path
                )
                if param_val_res.provenance:
                    new_prov = param_val_res.provenance.model_copy(
                        update={"parent_record_ids": list(parent_record_ids)}
                    )
                    param_val_res = param_val_res.model_copy(update={"provenance": new_prov})
                    parent_record_ids.append(new_prov.record_id)

        # 6. Optional Association Stage
        assoc_res = None
        if association_request is not None:
            df_to_slice = None
            parent_prov_for_engine = None
            if norm_res is not None and norm_res.data is not None:
                df_to_slice = norm_res.data
                parent_prov_for_engine = norm_res.provenance
            elif ingest_res is not None and ingest_res.data is not None:
                df_to_slice = ingest_res.data
                parent_prov_for_engine = ingest_res.provenance

            if df_to_slice is None or getattr(df_to_slice, "empty", True):
                prov_err = create_provenance(
                    source_uri=resolved_path,
                    activity_name="test_point_association",
                    epistemic_level=EpistemicLevel.CALCULATED,
                    parent_record_ids=parent_record_ids,
                )
                assoc_res = TestPointAssociationResult(
                    status=ResultStatus.ERROR,
                    issues=[
                        Issue(
                            issue_code="ERR_ASSOCIATION_DATA_UNAVAILABLE",
                            severity=IssueSeverity.ERROR,
                            message="Association requested but no valid DataFrame is available.",
                            location="DataPipelineOrchestrator",
                        )
                    ],
                    provenance=prov_err,
                    test_point_id=association_request.observation.test_point_id,
                    observation_id=association_request.observation.observation_id,
                    timestamp_column=association_request.timestamp_column,
                    row_count=0,
                    data=None,
                )
                parent_record_ids.append(prov_err.record_id)
            else:
                engine = TestPointAssociationEngine()
                assoc_res = engine.extract(
                    data=df_to_slice,
                    observation=association_request.observation,
                    timestamp_column=association_request.timestamp_column,
                    parent_provenance=parent_prov_for_engine,
                )
                if assoc_res.provenance:
                    new_prov = assoc_res.provenance.model_copy(
                        update={"parent_record_ids": list(parent_record_ids)}
                    )
                    assoc_res = assoc_res.model_copy(update={"provenance": new_prov})
                    parent_record_ids.append(new_prov.record_id)

        # 7. Optional Engineering Calculation Stage
        eng_res = None
        if engineering_request is not None:
            df_for_eng = None
            parent_prov_for_eng = None

            # Data selection policy
            if association_request is not None:
                if (
                    assoc_res is not None
                    and assoc_res.data is not None
                    and assoc_res.status != ResultStatus.ERROR
                    and assoc_res.status != ResultStatus.NOT_AVAILABLE
                ):
                    df_for_eng = assoc_res.data
                    parent_prov_for_eng = assoc_res.provenance
            elif normalization_request is not None:
                if (
                    norm_res is not None
                    and norm_res.data is not None
                    and norm_res.status != ResultStatus.ERROR
                ):
                    df_for_eng = norm_res.data
                    parent_prov_for_eng = norm_res.provenance
            else:
                if ingest_res is not None and ingest_res.data is not None:
                    df_for_eng = ingest_res.data
                    parent_prov_for_eng = ingest_res.provenance

            if df_for_eng is None or getattr(df_for_eng, "empty", True):
                prov_err = create_provenance(
                    source_uri=resolved_path,
                    activity_name="engineering_calculation",
                    epistemic_level=EpistemicLevel.CALCULATED,
                    parent_record_ids=parent_record_ids,
                )
                eng_res = EngineeringCalculationResult(
                    status=ResultStatus.ERROR,
                    result_id=f"res_{engineering_request.calculation_definition.calculation_id}_err",
                    calculation_id=engineering_request.calculation_definition.calculation_id,
                    issues=[
                        Issue(
                            issue_code="ERR_ENGINEERING_DATA_UNAVAILABLE",
                            severity=IssueSeverity.ERROR,
                            message=(
                                "Engineering calculation requested "
                                "but no valid DataFrame is available."
                            ),
                            location="DataPipelineOrchestrator",
                        )
                    ],
                    provenance=prov_err,
                )
                parent_record_ids.append(prov_err.record_id)
            else:
                executor = EngineeringExecutor()
                eng_res = executor.execute(
                    definition=engineering_request.calculation_definition,
                    data=df_for_eng,
                    parent_record_ids=[parent_prov_for_eng.record_id]
                    if parent_prov_for_eng
                    else [],
                    source_uri=resolved_path,
                )
                if eng_res.provenance:
                    new_prov = eng_res.provenance.model_copy(
                        update={"parent_record_ids": list(parent_record_ids)}
                    )
                    eng_res = eng_res.model_copy(update={"provenance": new_prov})
                    parent_record_ids.append(new_prov.record_id)

        # 8. Optional Statistical Analysis Stage
        stat_res = None
        if statistics_request is not None:
            df_for_stat = None
            parent_prov_for_stat = None

            # Data selection policy
            if association_request is not None:
                if (
                    assoc_res is not None
                    and assoc_res.data is not None
                    and assoc_res.status != ResultStatus.ERROR
                    and assoc_res.status != ResultStatus.NOT_AVAILABLE
                ):
                    df_for_stat = assoc_res.data
                    parent_prov_for_stat = assoc_res.provenance
            elif normalization_request is not None:
                if (
                    norm_res is not None
                    and norm_res.data is not None
                    and norm_res.status != ResultStatus.ERROR
                ):
                    df_for_stat = norm_res.data
                    parent_prov_for_stat = norm_res.provenance
            else:
                if ingest_res is not None and ingest_res.data is not None:
                    df_for_stat = ingest_res.data
                    parent_prov_for_stat = ingest_res.provenance

            if df_for_stat is None or getattr(df_for_stat, "empty", True):
                prov_err = create_provenance(
                    source_uri=resolved_path,
                    activity_name="statistical_analysis",
                    epistemic_level=EpistemicLevel.CALCULATED,
                    parent_record_ids=parent_record_ids,
                )
                stat_res = StatisticalAnalysisResult(
                    status=ResultStatus.ERROR,
                    result_id=f"res_{statistics_request.analysis_id}_err",
                    analysis_id=statistics_request.analysis_id,
                    target_column=statistics_request.target_column,
                    issues=[
                        Issue(
                            issue_code="ERR_STATISTICS_DATA_UNAVAILABLE",
                            severity=IssueSeverity.ERROR,
                            message=(
                                "Statistical analysis requested "
                                "but no valid DataFrame is available."
                            ),
                            location="DataPipelineOrchestrator",
                        )
                    ],
                    provenance=prov_err,
                )
                parent_record_ids.append(prov_err.record_id)
            else:
                stat_executor = StatisticalAnalysisEngine()
                stat_res = stat_executor.execute(
                    request=statistics_request,
                    data=df_for_stat,
                    parent_record_ids=[parent_prov_for_stat.record_id]
                    if parent_prov_for_stat
                    else [],
                    source_uri=resolved_path,
                )
                if stat_res.provenance:
                    new_prov = stat_res.provenance.model_copy(
                        update={"parent_record_ids": list(parent_record_ids)}
                    )
                    stat_res = stat_res.model_copy(update={"provenance": new_prov})
                    parent_record_ids.append(new_prov.record_id)

        # 9. Optional Time-Series Statistical Analysis Stage
        ts_res = None
        if time_series_request is not None:
            df_for_ts = None
            parent_prov_for_ts = None

            # Data selection policy
            if association_request is not None:
                if (
                    assoc_res is not None
                    and assoc_res.data is not None
                    and assoc_res.status != ResultStatus.ERROR
                    and assoc_res.status != ResultStatus.NOT_AVAILABLE
                ):
                    df_for_ts = assoc_res.data
                    parent_prov_for_ts = assoc_res.provenance
            elif normalization_request is not None:
                if (
                    norm_res is not None
                    and norm_res.data is not None
                    and norm_res.status != ResultStatus.ERROR
                ):
                    df_for_ts = norm_res.data
                    parent_prov_for_ts = norm_res.provenance
            else:
                if ingest_res is not None and ingest_res.data is not None:
                    df_for_ts = ingest_res.data
                    parent_prov_for_ts = ingest_res.provenance

            if df_for_ts is None or getattr(df_for_ts, "empty", True):
                prov_err = create_provenance(
                    source_uri=resolved_path,
                    activity_name="time_series_rolling_analysis",
                    epistemic_level=EpistemicLevel.CALCULATED,
                    parent_record_ids=parent_record_ids,
                )
                ts_res = TimeSeriesAnalysisResult(
                    status=ResultStatus.ERROR,
                    result_id=f"res_{time_series_request.analysis_id}_err",
                    analysis_id=time_series_request.analysis_id,
                    target_column=time_series_request.target_column,
                    timestamp_column=time_series_request.timestamp_column,
                    temporal_scope=time_series_request.temporal_scope,
                    window_configuration=time_series_request.window_configuration,
                    issues=[
                        Issue(
                            issue_code="ERR_TIME_SERIES_DATA_UNAVAILABLE",
                            severity=IssueSeverity.ERROR,
                            message=(
                                "Time-series analysis requested "
                                "but no valid DataFrame is available."
                            ),
                            location="DataPipelineOrchestrator",
                        )
                    ],
                    provenance=prov_err,
                )
                parent_record_ids.append(prov_err.record_id)
            else:
                ts_engine = TimeSeriesStatisticalAnalysisEngine()
                ts_res = ts_engine.execute(
                    request=time_series_request,
                    data=df_for_ts,
                    parent_record_ids=[parent_prov_for_ts.record_id] if parent_prov_for_ts else [],
                    source_uri=resolved_path,
                )
                if ts_res.provenance:
                    new_prov = ts_res.provenance.model_copy(
                        update={"parent_record_ids": list(parent_record_ids)}
                    )
                    ts_res = ts_res.model_copy(update={"provenance": new_prov})
                    parent_record_ids.append(new_prov.record_id)

        # 10. Optional Anomaly Detection Stage
        anomaly_res = None
        if anomaly_request is not None:
            # Source representation hierarchy
            df_for_anomaly = None
            parent_prov_for_anomaly = None

            # Check association
            if association_request is not None:
                if assoc_res is not None and assoc_res.status in (
                    ResultStatus.SUCCESS,
                    ResultStatus.PARTIAL,
                    ResultStatus.WARNING,
                ):
                    df_for_anomaly = assoc_res.data
                    parent_prov_for_anomaly = assoc_res.provenance
                else:
                    # Association explicitly requested but failed. No silent fallback.
                    anomaly_res = PipelineAnomalyDetectionResult(
                        status=ResultStatus.ERROR,
                        issues=[
                            Issue(
                                issue_code="ERR_ASSOCIATION_FAILED",
                                severity=IssueSeverity.ERROR,
                                message="Upstream association failed, skipping anomaly detection.",
                            )
                        ],
                    )
            # Check normalization
            elif normalization_request is not None:
                if norm_res is not None and norm_res.status in (
                    ResultStatus.SUCCESS,
                    ResultStatus.PARTIAL,
                    ResultStatus.WARNING,
                ):
                    df_for_anomaly = norm_res.data
                    parent_prov_for_anomaly = norm_res.provenance
                else:
                    # Normalization explicitly requested but failed. No silent fallback.
                    anomaly_res = PipelineAnomalyDetectionResult(
                        status=ResultStatus.ERROR,
                        issues=[
                            Issue(
                                issue_code="ERR_NORMALIZATION_FAILED",
                                severity=IssueSeverity.ERROR,
                                message=(
                                    "Upstream normalization failed, skipping anomaly detection."
                                ),
                            )
                        ],
                    )
            # Default to ingestion
            else:
                df_for_anomaly = ingest_res.data
                parent_prov_for_anomaly = ingest_res.provenance

            if df_for_anomaly is not None and anomaly_res is None:
                # Valid representation found, execute explicitly requested detectors
                det_results = []
                det_issues = []
                det_statuses = []

                # Threshold
                if anomaly_request.threshold_requests:
                    thresh_engine = ThresholdAnomalyDetectionEngine()
                    for req in anomaly_request.threshold_requests:
                        res = thresh_engine.execute(
                            request=req,
                            data=df_for_anomaly,
                            parent_record_ids=[parent_prov_for_anomaly.record_id]
                            if parent_prov_for_anomaly
                            else [],
                            source_uri=resolved_path,
                        )
                        det_results.append(res)
                        det_issues.extend(res.issues)
                        det_statuses.append(res.status)

                # Statistical
                if anomaly_request.statistical_requests:
                    stat_anom_engine = StatisticalAnomalyDetectionEngine()
                    for stat_req in anomaly_request.statistical_requests:
                        res = stat_anom_engine.execute(
                            request=stat_req,
                            data=df_for_anomaly,
                            parent_record_ids=[parent_prov_for_anomaly.record_id]
                            if parent_prov_for_anomaly
                            else [],
                            source_uri=resolved_path,
                        )
                        det_results.append(res)
                        det_issues.extend(res.issues)
                        det_statuses.append(res.status)

                # Time-Series
                if anomaly_request.time_series_requests:
                    ts_anom_engine = TimeSeriesAnomalyDetectionEngine()
                    for ts_req in anomaly_request.time_series_requests:
                        res = ts_anom_engine.execute(
                            request=ts_req,
                            data=df_for_anomaly,
                            parent_record_ids=[parent_prov_for_anomaly.record_id]
                            if parent_prov_for_anomaly
                            else [],
                            source_uri=resolved_path,
                        )
                        det_results.append(res)
                        det_issues.extend(res.issues)
                        det_statuses.append(res.status)

                if ResultStatus.ERROR in det_statuses:
                    ov_status = (
                        ResultStatus.PARTIAL
                        if ResultStatus.SUCCESS in det_statuses
                        else ResultStatus.ERROR
                    )
                elif ResultStatus.PARTIAL in det_statuses:
                    ov_status = ResultStatus.PARTIAL
                elif ResultStatus.WARNING in det_statuses:
                    ov_status = ResultStatus.WARNING
                elif all(s == ResultStatus.NOT_AVAILABLE for s in det_statuses) and det_statuses:
                    ov_status = ResultStatus.NOT_AVAILABLE
                else:
                    ov_status = ResultStatus.SUCCESS

                prov = ProvenanceRecord(
                    record_id=f"prov_anom_pipeline_{datetime.datetime.now(datetime.UTC).timestamp()}",
                    activity_name="pipeline_anomaly_detection",
                    parent_record_ids=list(parent_record_ids),
                    source_uri=resolved_path,
                    epistemic_level=EpistemicLevel.CALCULATED,
                )

                anomaly_res = PipelineAnomalyDetectionResult(
                    status=ov_status,
                    issues=det_issues,
                    provenance=prov,
                    detector_results=tuple(det_results),
                )

        eo_res: PipelineExpectedObservedResult | None = None
        if expected_observed_request is not None:
            eo_source_df = None
            eo_source_prov = None
            eo_source_status = ResultStatus.SUCCESS
            eo_issues = []

            if association_request is not None:
                if (
                    assoc_res
                    and assoc_res.status in (ResultStatus.SUCCESS, ResultStatus.WARNING)
                    and assoc_res.data is not None
                ):
                    eo_source_df = assoc_res.data
                    eo_source_prov = assoc_res.provenance
                else:
                    eo_source_status = ResultStatus.ERROR
                    eo_issues.append(
                        Issue(
                            issue_code="ERR_EXPECTED_OBSERVED_DATA_UNAVAILABLE",
                            severity=IssueSeverity.ERROR,
                            message="Upstream association failed, skipping expected-vs-observed.",
                        )
                    )
            elif normalization_request is not None:
                if (
                    norm_res
                    and norm_res.status in (ResultStatus.SUCCESS, ResultStatus.WARNING)
                    and norm_res.data is not None
                ):
                    eo_source_df = norm_res.data
                    eo_source_prov = norm_res.provenance
                else:
                    eo_source_status = ResultStatus.ERROR
                    eo_issues.append(
                        Issue(
                            issue_code="ERR_EXPECTED_OBSERVED_DATA_UNAVAILABLE",
                            severity=IssueSeverity.ERROR,
                            message="Upstream normalization failed, skipping expected-vs-observed.",
                        )
                    )
            else:
                if (
                    ingest_res.status in (ResultStatus.SUCCESS, ResultStatus.WARNING)
                    and ingest_res.data is not None
                ):
                    eo_source_df = ingest_res.data
                    eo_source_prov = ingest_res.provenance
                else:
                    eo_source_status = ResultStatus.ERROR
                    eo_issues.append(
                        Issue(
                            issue_code="ERR_EXPECTED_OBSERVED_DATA_UNAVAILABLE",
                            severity=IssueSeverity.ERROR,
                            message="Upstream ingestion failed, skipping expected-vs-observed.",
                        )
                    )

            if eo_source_status == ResultStatus.ERROR or eo_source_df is None:
                prov = ProvenanceRecord(
                    record_id=f"prov_eo_pipeline_{datetime.datetime.now(datetime.UTC).timestamp()}",
                    activity_name="pipeline_expected_observed",
                    parent_record_ids=list(parent_record_ids),
                    source_uri=resolved_path,
                    epistemic_level=EpistemicLevel.CALCULATED,
                )
                eo_res = PipelineExpectedObservedResult(
                    status=ResultStatus.ERROR, issues=eo_issues, provenance=prov, comparisons=()
                )
            else:
                comparisons = []
                for eo_req in expected_observed_request.scalar_requests:
                    if eo_req.observed_parameter_id not in eo_source_df.columns:
                        eo_issues.append(
                            Issue(
                                issue_code="ERR_MISSING_OBSERVED_COLUMN",
                                severity=IssueSeverity.ERROR,
                                message=f"Column {eo_req.observed_parameter_id} missing.",
                            )
                        )
                        err_prov = ProvenanceRecord(
                            record_id=str(uuid.uuid4()),
                            activity_name="EXPECTED_OBSERVED_COMPARISON",
                            epistemic_level=EpistemicLevel.CALCULATED,
                            parent_record_ids=[eo_source_prov.record_id] if eo_source_prov else [],
                            created_at=datetime.datetime.now(datetime.UTC),
                            source_uri=resolved_path,
                        )
                        err_comp = ExpectedObservedComparison(
                            comparison_id=eo_req.comparison_id,
                            expected=eo_req.expected,
                            observed=ObservedValue(
                                observed_id=str(uuid.uuid4()),
                                parameter_id=eo_req.observed_parameter_id,
                                value=0.0,
                                unit=eo_req.observed_unit,
                                source_reference=resolved_path,
                            ),
                            status=ResultStatus.ERROR,
                            issues=[
                                Issue(
                                    issue_code="ERR_MISSING_OBSERVED_COLUMN",
                                    severity=IssueSeverity.ERROR,
                                    message=f"Column {eo_req.observed_parameter_id} missing.",
                                )
                            ],
                            provenance=err_prov,
                        )
                        comparisons.append(err_comp)
                        continue

                    if eo_req.observed_row_index < 0 or eo_req.observed_row_index >= len(
                        eo_source_df
                    ):
                        eo_issues.append(
                            Issue(
                                issue_code="ERR_ROW_OUT_OF_BOUNDS",
                                severity=IssueSeverity.ERROR,
                                message=f"Row index {eo_req.observed_row_index} out of bounds.",
                            )
                        )
                        err_prov = ProvenanceRecord(
                            record_id=str(uuid.uuid4()),
                            activity_name="EXPECTED_OBSERVED_COMPARISON",
                            epistemic_level=EpistemicLevel.CALCULATED,
                            parent_record_ids=[eo_source_prov.record_id] if eo_source_prov else [],
                            created_at=datetime.datetime.now(datetime.UTC),
                            source_uri=resolved_path,
                        )
                        err_comp = ExpectedObservedComparison(
                            comparison_id=eo_req.comparison_id,
                            expected=eo_req.expected,
                            observed=ObservedValue(
                                observed_id=str(uuid.uuid4()),
                                parameter_id=eo_req.observed_parameter_id,
                                value=0.0,
                                unit=eo_req.observed_unit,
                                source_reference=resolved_path,
                            ),
                            status=ResultStatus.ERROR,
                            issues=[
                                Issue(
                                    issue_code="ERR_ROW_OUT_OF_BOUNDS",
                                    severity=IssueSeverity.ERROR,
                                    message=f"Row index {eo_req.observed_row_index} out of bounds.",
                                )
                            ],
                            provenance=err_prov,
                        )
                        comparisons.append(err_comp)
                        continue

                    val = eo_source_df.iloc[eo_req.observed_row_index][eo_req.observed_parameter_id]
                    if pd.isna(val):
                        eo_issues.append(
                            Issue(
                                issue_code="ERR_NON_FINITE_OBSERVED",
                                severity=IssueSeverity.ERROR,
                                message="Observed value is non-finite.",
                            )
                        )
                        err_prov = ProvenanceRecord(
                            record_id=str(uuid.uuid4()),
                            activity_name="EXPECTED_OBSERVED_COMPARISON",
                            epistemic_level=EpistemicLevel.CALCULATED,
                            parent_record_ids=[eo_source_prov.record_id] if eo_source_prov else [],
                            created_at=datetime.datetime.now(datetime.UTC),
                            source_uri=resolved_path,
                        )
                        err_comp = ExpectedObservedComparison(
                            comparison_id=eo_req.comparison_id,
                            expected=eo_req.expected,
                            observed=ObservedValue(
                                observed_id=str(uuid.uuid4()),
                                parameter_id=eo_req.observed_parameter_id,
                                value=0.0,
                                unit=eo_req.observed_unit,
                                source_reference=resolved_path,
                            ),
                            status=ResultStatus.ERROR,
                            issues=[
                                Issue(
                                    issue_code="ERR_NON_FINITE_OBSERVED",
                                    severity=IssueSeverity.ERROR,
                                    message="Observed value is non-finite.",
                                )
                            ],
                            provenance=err_prov,
                        )
                        comparisons.append(err_comp)
                        continue

                    obs_val = ObservedValue(
                        observed_id=str(uuid.uuid4()),
                        parameter_id=eo_req.observed_parameter_id,
                        value=float(val),
                        unit=eo_req.observed_unit,
                        source_reference=resolved_path,
                    )

                    comp = ExpectedObservedEngine.compare(
                        expected=eo_req.expected,
                        observed=obs_val,
                        method=eo_req.method,
                        tolerance=eo_req.tolerance,
                        parent_record_ids=[eo_source_prov.record_id] if eo_source_prov else [],
                    )

                    comp = ExpectedObservedComparison(
                        comparison_id=eo_req.comparison_id,
                        expected=comp.expected,
                        observed=comp.observed,
                        deviation=comp.deviation,
                        tolerance=comp.tolerance,
                        tolerance_evaluated=comp.tolerance_evaluated,
                        within_tolerance=comp.within_tolerance,
                        status=comp.status,
                        issues=comp.issues,
                        provenance=comp.provenance,
                    )

                    comparisons.append(comp)

                c_statuses = [c.status for c in comparisons]
                if ResultStatus.ERROR in c_statuses:
                    ov_status = (
                        ResultStatus.PARTIAL
                        if ResultStatus.SUCCESS in c_statuses
                        else ResultStatus.ERROR
                    )
                elif ResultStatus.PARTIAL in c_statuses:
                    ov_status = ResultStatus.PARTIAL
                elif ResultStatus.WARNING in c_statuses:
                    ov_status = ResultStatus.WARNING
                elif all(s == ResultStatus.NOT_AVAILABLE for s in c_statuses) and c_statuses:
                    ov_status = ResultStatus.NOT_AVAILABLE
                else:
                    ov_status = ResultStatus.SUCCESS

                for c in comparisons:
                    eo_issues.extend(c.issues)

                prov = ProvenanceRecord(
                    record_id=f"prov_eo_pipeline_{datetime.datetime.now(datetime.UTC).timestamp()}",
                    activity_name="pipeline_expected_observed",
                    parent_record_ids=[eo_source_prov.record_id]
                    if eo_source_prov
                    else list(parent_record_ids),
                    source_uri=resolved_path,
                    epistemic_level=EpistemicLevel.CALCULATED,
                )

                eo_res = PipelineExpectedObservedResult(
                    status=ov_status,
                    issues=eo_issues,
                    provenance=prov,
                    comparisons=tuple(comparisons),
                )

        # Determine overall pipeline status

        statuses = [ingest_res.status, val_res.status, qual_res.status]
        if norm_res is not None:
            statuses.append(norm_res.status)
        if param_val_res is not None:
            statuses.append(param_val_res.status)
        if assoc_res is not None:
            statuses.append(assoc_res.status)
        if eng_res is not None:
            statuses.append(eng_res.status)
        if stat_res is not None:
            statuses.append(stat_res.status)
        if ts_res is not None:
            statuses.append(ts_res.status)
        if anomaly_res is not None:
            statuses.append(anomaly_res.status)
        if eo_res is not None:
            statuses.append(eo_res.status)

        if ResultStatus.ERROR in statuses:
            overall_status = ResultStatus.ERROR
        elif ResultStatus.PARTIAL in statuses:
            overall_status = ResultStatus.PARTIAL
        elif ResultStatus.NOT_AVAILABLE in statuses:
            overall_status = ResultStatus.NOT_AVAILABLE
        elif ResultStatus.WARNING in statuses:
            overall_status = ResultStatus.WARNING
        else:
            overall_status = ResultStatus.SUCCESS

        # Aggregate issues across stages
        all_issues = list(ingest_res.issues) + list(val_res.issues) + list(qual_res.issues)
        if norm_res is not None:
            all_issues.extend(norm_res.issues)
        if param_val_res is not None:
            all_issues.extend(param_val_res.issues)
        if assoc_res is not None:
            all_issues.extend(assoc_res.issues)
        if eng_res is not None:
            all_issues.extend(eng_res.issues)
        if stat_res is not None:
            all_issues.extend(stat_res.issues)
        if ts_res is not None:
            all_issues.extend(ts_res.issues)
        if anomaly_res is not None:
            all_issues.extend(anomaly_res.issues)
        if eo_res is not None:
            all_issues.extend(eo_res.issues)

        # 10. Optional Vehicle Engineering Evaluation Stage
        veh_eval_res = None
        if vehicle_evaluation_request is not None:
            if val_res is not None and val_res.status == ResultStatus.ERROR:
                veh_err_prov = create_provenance(
                    source_uri=resolved_path,
                    activity_name="vehicle_engineering_evaluation",
                    epistemic_level=EpistemicLevel.LIMIT,
                    parent_record_ids=list(parent_record_ids),
                    parameters={"status": "ERROR", "reason": "Upstream validation failed"},
                )
                val_fail_issue = Issue(
                    issue_code="ERR_UPSTREAM_VALIDATION_FAILED",
                    severity=IssueSeverity.ERROR,
                    message="Vehicle evaluation aborted: upstream data validation failed.",
                    location="DataPipelineOrchestrator",
                )
                veh_eval_res = VehicleEvaluationResult(
                    evaluation_id=f"VEH_EVAL_ABORT_{uuid.uuid4().hex[:8].upper()}",
                    vehicle_id=vehicle_evaluation_request.vehicle_profile.vehicle_id
                    if vehicle_evaluation_request.vehicle_profile
                    else "UNKNOWN",
                    profile_id=vehicle_evaluation_request.vehicle_profile.profile_id
                    if vehicle_evaluation_request.vehicle_profile
                    else "UNKNOWN",
                    profile_version=vehicle_evaluation_request.vehicle_profile.profile_version
                    if vehicle_evaluation_request.vehicle_profile
                    else "0.0.0",
                    configuration_id=vehicle_evaluation_request.configuration_envelope.configuration_id
                    if vehicle_evaluation_request.configuration_envelope
                    else None,
                    limit_evaluations=(),
                    procedure_evaluations=(),
                    configuration_discrepancies=(),
                    status=ResultStatus.ERROR,
                    issues=[val_fail_issue],
                    provenance=veh_err_prov,
                    summary={"error": "Aborted due to upstream validation failure"},
                )
            elif vehicle_evaluation_request.vehicle_profile is None:
                veh_err_prov = create_provenance(
                    source_uri=resolved_path,
                    activity_name="vehicle_engineering_evaluation",
                    epistemic_level=EpistemicLevel.LIMIT,
                    parent_record_ids=list(parent_record_ids),
                    parameters={"status": "NOT_AVAILABLE", "reason": "Missing vehicle_profile"},
                )
                missing_prof_issue = Issue(
                    issue_code="WARN_VEHICLE_PROFILE_MISSING",
                    severity=IssueSeverity.WARNING,
                    message="Vehicle evaluation skipped: required VehicleProfile is absent.",
                    location="DataPipelineOrchestrator",
                )
                veh_eval_res = VehicleEvaluationResult(
                    evaluation_id=f"VEH_EVAL_SKIPPED_{uuid.uuid4().hex[:8].upper()}",
                    vehicle_id="UNKNOWN",
                    profile_id="UNKNOWN",
                    profile_version="0.0.0",
                    configuration_id=vehicle_evaluation_request.configuration_envelope.configuration_id
                    if vehicle_evaluation_request.configuration_envelope
                    else None,
                    limit_evaluations=(),
                    procedure_evaluations=(),
                    configuration_discrepancies=(),
                    status=ResultStatus.NOT_AVAILABLE,
                    issues=[missing_prof_issue],
                    provenance=veh_err_prov,
                    summary={"skipped": True, "reason": "Missing vehicle_profile"},
                )
            else:
                df_for_eval = None
                parent_prov_for_eval = None
                if (
                    assoc_res is not None
                    and assoc_res.data is not None
                    and assoc_res.status != ResultStatus.ERROR
                ):
                    df_for_eval = assoc_res.data
                    parent_prov_for_eval = assoc_res.provenance
                elif (
                    norm_res is not None
                    and norm_res.data is not None
                    and norm_res.status != ResultStatus.ERROR
                ):
                    df_for_eval = norm_res.data
                    parent_prov_for_eval = norm_res.provenance
                elif ingest_res is not None and ingest_res.data is not None:
                    df_for_eval = ingest_res.data
                    parent_prov_for_eval = ingest_res.provenance

                obs_list: list[EvaluationObservation] = list(
                    vehicle_evaluation_request.explicit_observations
                )

                if (
                    vehicle_evaluation_request.auto_extract_observations
                    and df_for_eval is not None
                    and not getattr(df_for_eval, "empty", True)
                ):
                    candidate_cols = set()
                    if vehicle_evaluation_request.limit_profile is not None:
                        for lim in vehicle_evaluation_request.limit_profile.limits:
                            if lim.parameter_id in df_for_eval.columns:
                                candidate_cols.add(lim.parameter_id)
                    if vehicle_evaluation_request.procedure_profile is not None:
                        for proc_req in vehicle_evaluation_request.procedure_profile.requirements:
                            if (
                                proc_req.parameter_id
                                and proc_req.parameter_id in df_for_eval.columns
                            ):
                                candidate_cols.add(proc_req.parameter_id)
                    if vehicle_evaluation_request.vehicle_profile is not None:
                        for p in vehicle_evaluation_request.vehicle_profile.supported_parameter_ids:
                            if p in df_for_eval.columns:
                                candidate_cols.add(p)

                    if not candidate_cols:
                        candidate_cols = {
                            c
                            for c in df_for_eval.columns
                            if c.lower() not in ("timestamp", "time", "date")
                        }

                    cfg_col = None
                    for c_name in (
                        "configuration_id",
                        "config_id",
                        "configuration",
                        "vehicle_configuration",
                    ):
                        if c_name in df_for_eval.columns:
                            cfg_col = c_name
                            break

                    ts_col = None
                    for t_name in ("timestamp", "time", "Time", "Timestamp"):
                        if t_name in df_for_eval.columns:
                            ts_col = t_name
                            break

                    active_registry = (
                        vehicle_evaluation_request.parameter_registry or parameter_registry
                    )
                    extracted_count = 0
                    max_obs = vehicle_evaluation_request.max_extracted_observations

                    for row_idx, row in df_for_eval.iterrows():
                        if extracted_count >= max_obs:
                            break

                        row_cfg = (
                            str(row[cfg_col]) if cfg_col and not pd.isna(row[cfg_col]) else None
                        )
                        row_ts = None
                        if ts_col and not pd.isna(row[ts_col]):
                            raw_ts = row[ts_col]
                            if isinstance(raw_ts, datetime.datetime):
                                row_ts = (
                                    raw_ts
                                    if raw_ts.tzinfo is not None
                                    else raw_ts.replace(tzinfo=datetime.UTC)
                                )
                            else:
                                try:
                                    parsed_ts = datetime.datetime.fromisoformat(
                                        str(raw_ts).replace("Z", "+00:00")
                                    )
                                    row_ts = (
                                        parsed_ts
                                        if parsed_ts.tzinfo is not None
                                        else parsed_ts.replace(tzinfo=datetime.UTC)
                                    )
                                except Exception:
                                    row_ts = None

                        for col in sorted(candidate_cols):
                            if extracted_count >= max_obs:
                                break
                            val = row[col]
                            if pd.isna(val):
                                continue

                            unit = None
                            if active_registry is not None and active_registry.contains(col):
                                p_def = active_registry.get(col)
                                if p_def and p_def.canonical_unit:
                                    unit = p_def.canonical_unit
                            if (
                                unit is None
                                and vehicle_evaluation_request.limit_profile is not None
                            ):
                                lim_prof = vehicle_evaluation_request.limit_profile
                                lims = lim_prof.get_limits_for_parameter(
                                    col, include_all_configurations=True
                                )
                                if lims and lims[0].unit:
                                    unit = lims[0].unit
                            if (
                                unit is None
                                and vehicle_evaluation_request.procedure_profile is not None
                            ):
                                proc_prof = vehicle_evaluation_request.procedure_profile
                                reqs = [r for r in proc_prof.requirements if r.parameter_id == col]
                                if reqs and reqs[0].unit:
                                    unit = reqs[0].unit
                            if unit is None:
                                for sfx, u in (
                                    ("_kts", "kts"),
                                    ("_ft", "ft"),
                                    ("_g", "g"),
                                    ("_deg", "deg"),
                                    ("_c", "degC"),
                                    ("_psi", "psi"),
                                    ("_rpm", "rpm"),
                                ):
                                    if col.lower().endswith(sfx):
                                        unit = u
                                        break

                            try:
                                float_val = float(val)
                                obs = EvaluationObservation(
                                    observation_id=f"OBS-AUTO-{col}-{row_idx}",
                                    parameter_id=col,
                                    value=float_val,
                                    unit=unit,
                                    timestamp=row_ts,
                                    row_index=int(row_idx),
                                    source_reference=resolved_path,
                                    configuration_id=row_cfg,
                                    test_card_id=vehicle_evaluation_request.test_card_id,
                                    test_point_id=vehicle_evaluation_request.test_point_id,
                                    is_qualitative=False,
                                )
                            except (ValueError, TypeError):
                                obs = EvaluationObservation(
                                    observation_id=f"OBS-AUTO-{col}-{row_idx}",
                                    parameter_id=col,
                                    unit=unit,
                                    timestamp=row_ts,
                                    row_index=int(row_idx),
                                    source_reference=resolved_path,
                                    configuration_id=row_cfg,
                                    test_card_id=vehicle_evaluation_request.test_card_id,
                                    test_point_id=vehicle_evaluation_request.test_point_id,
                                    is_qualitative=True,
                                    qualitative_state=str(val),
                                )
                            obs_list.append(obs)
                            extracted_count += 1

                eval_req = VehicleEvaluationRequest(
                    request_id=f"VEH_REQ_{uuid.uuid4().hex[:8].upper()}",
                    vehicle_profile=vehicle_evaluation_request.vehicle_profile,
                    configuration_envelope=vehicle_evaluation_request.configuration_envelope,
                    limit_profile=vehicle_evaluation_request.limit_profile,
                    procedure_profile=vehicle_evaluation_request.procedure_profile,
                    observations=tuple(obs_list),
                    test_card_id=vehicle_evaluation_request.test_card_id,
                    test_point_id=vehicle_evaluation_request.test_point_id,
                    context_id=vehicle_evaluation_request.context_id,
                    parameter_registry=vehicle_evaluation_request.parameter_registry
                    or parameter_registry,
                    evaluate_limits=vehicle_evaluation_request.evaluate_limits,
                    evaluate_procedures=vehicle_evaluation_request.evaluate_procedures,
                    strict_units=vehicle_evaluation_request.strict_units,
                    provenance=parent_prov_for_eval,
                    metadata=vehicle_evaluation_request.metadata,
                )
                veh_engine = VehicleEngineeringEvaluationEngine()
                veh_eval_res = veh_engine.evaluate(eval_req)

            all_issues.extend(veh_eval_res.issues)
            if veh_eval_res.provenance:
                parent_record_ids.append(veh_eval_res.provenance.record_id)
            if veh_eval_res.status == ResultStatus.ERROR:
                overall_status = ResultStatus.ERROR
            elif (
                veh_eval_res.status == ResultStatus.PARTIAL and overall_status != ResultStatus.ERROR
            ):
                overall_status = ResultStatus.PARTIAL
            elif (
                veh_eval_res.status == ResultStatus.WARNING
                and overall_status == ResultStatus.SUCCESS
            ):
                overall_status = ResultStatus.WARNING
            elif (
                veh_eval_res.status == ResultStatus.NOT_AVAILABLE
                and overall_status == ResultStatus.SUCCESS
            ):
                overall_status = ResultStatus.PARTIAL

        # 11. Optional Evidence Construction Stage
        evidence_res = None
        if evidence_request is not None:
            ev_engine = PipelineEvidenceEngine()
            evidence_res = ev_engine.process(
                request=evidence_request,
                parent_record_ids=list(parent_record_ids),
                source_uri=resolved_path,
                anomaly_result=anomaly_res,
                engineering_results=[eng_res]
                if eng_res and eng_res.status != ResultStatus.ERROR
                else None,
                statistical_results=[stat_res]
                if stat_res and stat_res.status != ResultStatus.ERROR
                else None,
                expected_observed_result=eo_res,
                vehicle_evaluation_result=veh_eval_res,
            )

            all_issues.extend(evidence_res.issues)
            if evidence_res.provenance:
                parent_record_ids.append(evidence_res.provenance.record_id)
            if evidence_res.status == ResultStatus.ERROR:
                overall_status = ResultStatus.ERROR
            elif (
                evidence_res.status == ResultStatus.PARTIAL
                and overall_status == ResultStatus.SUCCESS
            ):
                overall_status = ResultStatus.PARTIAL

        # 12. Optional Contributor Analysis Stage
        contributor_analysis_res = None
        if contributor_analysis_request is not None:
            # Source representation hierarchy
            df_for_contrib = None
            parent_prov_for_contrib = None
            contrib_issues = []
            contrib_status = ResultStatus.SUCCESS

            if association_request is not None:
                if assoc_res is not None and assoc_res.status in (
                    ResultStatus.SUCCESS,
                    ResultStatus.PARTIAL,
                    ResultStatus.WARNING,
                ):
                    df_for_contrib = assoc_res.data
                    parent_prov_for_contrib = assoc_res.provenance
                else:
                    contrib_issues.append(
                        Issue(
                            issue_code="ERR_UPSTREAM_ASSOCIATION_FAILED",
                            severity=IssueSeverity.ERROR,
                            message=(
                                "Association was requested but failed; contributor "
                                "analysis cannot fallback."
                            ),
                        )
                    )
                    contrib_status = ResultStatus.ERROR
            elif normalization_request is not None:
                if norm_res is not None and norm_res.status in (
                    ResultStatus.SUCCESS,
                    ResultStatus.PARTIAL,
                    ResultStatus.WARNING,
                ):
                    df_for_contrib = norm_res.data
                    parent_prov_for_contrib = norm_res.provenance
                else:
                    contrib_issues.append(
                        Issue(
                            issue_code="ERR_UPSTREAM_NORMALIZATION_FAILED",
                            severity=IssueSeverity.ERROR,
                            message=(
                                "Normalization was requested but failed; contributor "
                                "analysis cannot fallback."
                            ),
                        )
                    )
                    contrib_status = ResultStatus.ERROR
            else:
                if ingest_res is not None and ingest_res.data is not None:
                    df_for_contrib = ingest_res.data
                    parent_prov_for_contrib = ingest_res.provenance
                else:
                    contrib_issues.append(
                        Issue(
                            issue_code="ERR_UPSTREAM_INGESTION_FAILED",
                            severity=IssueSeverity.ERROR,
                            message="Ingestion failed; contributor analysis cannot run.",
                        )
                    )
                    contrib_status = ResultStatus.ERROR

            contrib_results = []
            if df_for_contrib is not None and contrib_status != ResultStatus.ERROR:
                has_success = False
                has_nonfatal_issue = False

                for contrib_req in contributor_analysis_request.requests:
                    upd_req = contrib_req.model_copy(
                        update={
                            "parent_record_ids": tuple([parent_prov_for_contrib.record_id])
                            if parent_prov_for_contrib
                            else tuple(),
                            "source_reference": resolved_path,
                        }
                    )
                    contrib_res = ContributorAnalysisEngine.analyze(
                        request=upd_req, df=df_for_contrib
                    )
                    contrib_results.append(contrib_res)
                    all_issues.extend(contrib_res.issues)

                    if contrib_res.status == ResultStatus.SUCCESS:
                        has_success = True
                    elif contrib_res.status in (ResultStatus.PARTIAL, ResultStatus.WARNING):
                        has_success = True
                        has_nonfatal_issue = True
                    elif contrib_res.status in (ResultStatus.ERROR, ResultStatus.NOT_AVAILABLE):
                        has_nonfatal_issue = True

                if not contributor_analysis_request.requests:
                    contrib_status = ResultStatus.SUCCESS
                elif has_success and has_nonfatal_issue:
                    contrib_status = ResultStatus.PARTIAL
                elif has_success and not has_nonfatal_issue:
                    contrib_status = ResultStatus.SUCCESS
                elif not has_success and any(
                    r.status == ResultStatus.NOT_AVAILABLE for r in contrib_results
                ):
                    contrib_status = ResultStatus.NOT_AVAILABLE
                else:
                    contrib_status = ResultStatus.ERROR

            contrib_prov = create_provenance(
                source_uri=resolved_path,
                activity_name="contributor_analysis",
                parent_record_ids=[parent_prov_for_contrib.record_id]
                if parent_prov_for_contrib
                else list(parent_record_ids),
                epistemic_level=EpistemicLevel.CALCULATED,
            )

            contributor_analysis_res = PipelineContributorAnalysisResult(
                status=contrib_status,
                issues=contrib_issues,
                provenance=contrib_prov,
                results=tuple(contrib_results),
            )
            all_issues.extend(contrib_issues)
            parent_record_ids.append(contrib_prov.record_id)

            if contrib_status == ResultStatus.ERROR:
                overall_status = ResultStatus.ERROR
            elif contrib_status == ResultStatus.PARTIAL and overall_status == ResultStatus.SUCCESS:
                overall_status = ResultStatus.PARTIAL

        # 13. Optional Recommendation Generation Stage
        recommendation_res = None
        if recommendation_request is not None:
            try:
                actual_rec_req: RecommendationGenerationRequest
                if isinstance(recommendation_request, RecommendationGenerationRequest):
                    actual_rec_req = recommendation_request
                elif isinstance(recommendation_request, PipelineRecommendationRequest):
                    val_to_use = (
                        val_res
                        if (
                            recommendation_request.use_pipeline_validation
                            and recommendation_request.validation_result is None
                        )
                        else recommendation_request.validation_result
                    )
                    qual_to_use = (
                        qual_res
                        if (
                            recommendation_request.use_pipeline_quality
                            and recommendation_request.quality_result is None
                        )
                        else recommendation_request.quality_result
                    )
                    ev_to_use = (
                        evidence_res.collection
                        if (
                            recommendation_request.use_pipeline_evidence
                            and recommendation_request.evidence_collection is None
                            and evidence_res is not None
                            and getattr(evidence_res, "collection", None) is not None
                        )
                        else recommendation_request.evidence_collection
                    )
                    veh_eval_to_use = (
                        veh_eval_res
                        if (
                            recommendation_request.use_pipeline_vehicle_evaluation
                            and recommendation_request.vehicle_evaluation_result is None
                        )
                        else recommendation_request.vehicle_evaluation_result
                    )
                    actual_rec_req = RecommendationGenerationRequest(
                        hypothesis_assessment=recommendation_request.hypothesis_assessment,
                        hypothesis=recommendation_request.hypothesis,
                        evidence_collection=ev_to_use,
                        validation_result=val_to_use,
                        quality_result=qual_to_use,
                        vehicle_evaluation_result=veh_eval_to_use,
                        policy=recommendation_request.policy,
                        investigation_id=recommendation_request.investigation_id,
                        metadata=recommendation_request.metadata,
                    )

                else:
                    req_type = type(recommendation_request).__name__
                    raise TypeError(f"Unsupported recommendation_request type: {req_type}")

                recommendation_res = RecommendationGenerationEngine.generate(actual_rec_req)
                all_issues.extend(recommendation_res.issues)
                if recommendation_res.provenance:
                    parent_record_ids.append(recommendation_res.provenance.record_id)

                if recommendation_res.status == ResultStatus.ERROR:
                    overall_status = ResultStatus.ERROR
                elif (
                    recommendation_res.status == ResultStatus.PARTIAL
                    and overall_status != ResultStatus.ERROR
                ):
                    overall_status = ResultStatus.PARTIAL
                elif (
                    recommendation_res.status == ResultStatus.WARNING
                    and overall_status == ResultStatus.SUCCESS
                ):
                    overall_status = ResultStatus.WARNING
            except Exception as e:
                rec_issue = Issue(
                    issue_code="ERR_RECOMMENDATION_STAGE_FAILED",
                    severity=IssueSeverity.ERROR,
                    message=f"Recommendation generation stage failed: {e}",
                )
                all_issues.append(rec_issue)
                overall_status = ResultStatus.ERROR
                err_prov = create_provenance(
                    activity_name="RecommendationGenerationFailure",
                    source_uri=resolved_path,
                    epistemic_level=EpistemicLevel.RECOMMENDATION,
                    parent_record_ids=list(parent_record_ids),
                )
                err_coll = RecommendationCollection(
                    collection_id="rec-coll-error",
                    recommendations=(),
                    provenance=err_prov,
                )
                recommendation_res = RecommendationGenerationResult(
                    recommendations=(),
                    recommendation_collection=err_coll,
                    rules_evaluated=(),
                    rules_triggered=(),
                    policy_used=recommendation_request.policy
                    if hasattr(recommendation_request, "policy")
                    else RecommendationGenerationPolicy(),
                    provenance=err_prov,
                    status=ResultStatus.ERROR,
                    issues=[rec_issue],
                )

        pipeline_prov = create_provenance(
            source_uri=resolved_path,
            activity_name="csv_data_pipeline",
            epistemic_level=EpistemicLevel.CALCULATED,
            parent_record_ids=list(parent_record_ids),
        )

        return DataPipelineResult(
            status=overall_status,
            issues=all_issues,
            provenance=pipeline_prov,
            ingestion=ingest_res,
            validation=val_res,
            quality=qual_res,
            normalization=norm_res,
            parameter_validation=param_val_res,
            association=assoc_res,
            engineering=eng_res,
            statistics=stat_res,
            time_series=ts_res,
            anomaly=anomaly_res,
            expected_observed=eo_res,
            evidence=evidence_res,
            contributor_analysis=contributor_analysis_res,
            recommendation=recommendation_res,
            vehicle_evaluation=veh_eval_res,
        )

    def export_report(
        self,
        pipeline_result: DataPipelineResult,
        output_path: str | pathlib.Path,
        format: Any = None,
        *,
        figures: Any = None,
        include_plotlyjs: bool | str = False,
        overwrite: bool = False,
        base_dir: str | pathlib.Path | None = None,
        report_title: str = "AERIS Flight Test Engineering Report",
        analyst_name: str | None = None,
        custom_metadata: dict[str, Any] | None = None,
        raise_on_error: bool = False,
    ) -> ReportExportResult:
        """
        Extract and safely export an EngineeringReport from a pipeline result.

        Convenience method delegating to ReportExporter without mutating
        the pipeline result or rerunning any pipeline stages.
        """
        from aeris.core.reporting.exporter import export_pipeline_report

        return export_pipeline_report(
            pipeline_result=pipeline_result,
            output_path=output_path,
            format=format,
            figures=figures,
            include_plotlyjs=include_plotlyjs,
            overwrite=overwrite,
            base_dir=base_dir,
            report_title=report_title,
            analyst_name=analyst_name,
            custom_metadata=custom_metadata,
            raise_on_error=raise_on_error,
        )


def run_csv_pipeline(
    file_path: str | pathlib.Path,
    normalization_request: NormalizationRequest | None = None,
    parameter_registry: ParameterRegistry | None = None,
    parameter_validation_request: ParameterValidationRequest | None = None,
    association_request: TestPointAssociationRequest | None = None,
    engineering_request: EngineeringCalculationRequest | None = None,
    statistics_request: StatisticalAnalysisRequest | None = None,
    time_series_request: TimeSeriesAnalysisRequest | None = None,
    anomaly_request: PipelineAnomalyDetectionRequest | None = None,
    expected_observed_request: PipelineExpectedObservedRequest | None = None,
    evidence_request: PipelineEvidenceRequest | None = None,
    vehicle_evaluation_request: PipelineVehicleEvaluationRequest | None = None,
    contributor_analysis_request: PipelineContributorAnalysisRequest | None = None,
    recommendation_request: (
        PipelineRecommendationRequest | RecommendationGenerationRequest | None
    ) = None,
) -> DataPipelineResult:
    """Convenience entrypoint for executing the CSV data pipeline."""
    orchestrator = DataPipelineOrchestrator()
    return orchestrator.run_csv_pipeline(
        file_path,
        normalization_request=normalization_request,
        parameter_registry=parameter_registry,
        parameter_validation_request=parameter_validation_request,
        association_request=association_request,
        engineering_request=engineering_request,
        statistics_request=statistics_request,
        time_series_request=time_series_request,
        anomaly_request=anomaly_request,
        expected_observed_request=expected_observed_request,
        evidence_request=evidence_request,
        vehicle_evaluation_request=vehicle_evaluation_request,
        contributor_analysis_request=contributor_analysis_request,
        recommendation_request=recommendation_request,
    )


# Rebuild DataPipelineResult type to resolve forward references

DataPipelineResult.model_rebuild(
    _types_namespace={
        "PipelineAnomalyDetectionResult": PipelineAnomalyDetectionResult,
        "PipelineExpectedObservedResult": PipelineExpectedObservedResult,
        "PipelineEvidenceResult": PipelineEvidenceResult,
        "PipelineContributorAnalysisResult": PipelineContributorAnalysisResult,
        "RecommendationGenerationResult": RecommendationGenerationResult,
        "VehicleEvaluationResult": VehicleEvaluationResult,
    }
)
