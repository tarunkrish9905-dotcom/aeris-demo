"""
Tests for AERIS Engineering Report Domain Models & Pipeline Result Extractor (Step 14e).

Validates:
1. Faithful extraction of all 15 stages in strictly defined sequence.
2. Executive summary synthesis (overall status, stage counts, issue counts, narrative).
3. Data quality and validation extraction (metrics, violations, transformations).
4. Flight context extraction (vehicle identity, profile version, configuration).
5. Engineering evaluation extraction (limits, margins, exceedances, procedures, epistemic levels).
6. Anomaly and expected-observed extraction (classifications, severities, tolerances).
7. Investigation intelligence extraction (evidence items, contributor rankings, recommendations).
8. Advisory notice on recommendations ("FOR ENGINEERING REVIEW ONLY").
9. Epistemic level integrity (OBSERVED, CALCULATED, LIMIT, PROCEDURAL, RECOMMENDATION).
10. Honest representation of unexecuted / skipped stages in LimitationsSummary.
11. Upstream failure handling (marking downstream stages as UPSTREAM_ABORTED).
12. Inconclusive check handling and caveat documentation.
13. Immutability of domain models and non-mutation of source DataPipelineResult.
14. Functional interface extract_engineering_report.
"""

from __future__ import annotations

import pathlib

import pandas as pd
import pytest
from pydantic import ValidationError

from aeris.core.anomaly.models import (
    AnomalyClassification,
    AnomalyRecord,
    AnomalyResult,
    AnomalySeverity,
    DetectionMethod,
    PipelineAnomalyDetectionResult,
)
from aeris.core.evidence.pipeline import EvidenceSourceSelection, PipelineEvidenceRequest
from aeris.core.expected_observed.models import (
    ExpectationSource,
    ExpectedObservedComparison,
    ExpectedValue,
    ObservedValue,
    PipelineExpectedObservedResult,
)
from aeris.core.limits.models import (
    EngineeringLimitProfile,
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
)
from aeris.core.pipeline.orchestrator import run_csv_pipeline
from aeris.core.provenance.tracker import EpistemicLevel, create_provenance
from aeris.core.recommendation.generation_models import RecommendationGenerationPolicy
from aeris.core.recommendation.pipeline import PipelineRecommendationRequest
from aeris.core.reporting import (
    EngineeringReport,
    ReportDataExtractor,
    StageStatus,
    extract_engineering_report,
)
from aeris.core.results.types import (
    DataPipelineResult,
    IngestionResult,
    ResultStatus,
)
from aeris.core.sop.models import (
    ProcedureRequirement,
    ProcedureRequirementProfile,
    RequirementCategory,
)
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.profile_models import VehicleProfile
from aeris.core.vehicle_evaluation.models import EvaluationObservation
from aeris.core.vehicle_evaluation.pipeline import PipelineVehicleEvaluationRequest

# ---------------------------------------------------------------------------
# Synthetic Test Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_vehicle_profile() -> VehicleProfile:
    """Return a standard synthetic VehicleProfile."""
    return VehicleProfile(
        profile_id="PROF-SYNTH-X1",
        vehicle_id="AC-SYNTH-01",
        profile_version="1.0.0",
        name="Synthetic Flight Test Model X1",
        supported_parameter_ids=("airspeed_kts", "altitude_ft", "nz_g", "gear_handle"),
        metadata={"category": "experimental"},
    )


@pytest.fixture
def synthetic_envelope() -> ConfigurationEnvelope:
    """Return a standard synthetic ConfigurationEnvelope."""
    return ConfigurationEnvelope(
        configuration_id="CRUISE_CLEAN",
        vehicle_id="AC-SYNTH-01",
        profile_id="PROF-SYNTH-X1",
        profile_version="1.0.0",
        name="Clean Cruise Envelope",
    )


@pytest.fixture
def synthetic_limit_profile() -> EngineeringLimitProfile:
    """Return a synthetic limit profile with a Vne limit."""
    return EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-SYNTH-01",
        vehicle_id="AC-SYNTH-01",
        profile_id="PROF-SYNTH-X1",
        profile_version="1.0.0",
        limits=(
            ParameterLimitDefinition(
                limit_id="LIM-VNE-01",
                parameter_id="airspeed_kts",
                limit_type=LimitType.UPPER_BOUND,
                upper_bound=250.0,
                unit="kts",
                severity=LimitSeverity.WARNING,
                vehicle_id="AC-SYNTH-01",
                profile_id="PROF-SYNTH-X1",
                profile_version="1.0.0",
                source_reference="SYNTH-AFM-SEC2",
                description="Never exceed airspeed (Vne)",
            ),
        ),
    )


@pytest.fixture
def synthetic_procedure_profile() -> ProcedureRequirementProfile:
    """Return a synthetic procedure requirement profile."""
    return ProcedureRequirementProfile(
        profile_id="PROC-PROF-SYNTH-01",
        document_id="SOP-SYNTH-001",
        document_version="1.2",
        vehicle_id="AC-SYNTH-01",
        profile_version="1.0.0",
        requirements=(
            ProcedureRequirement(
                requirement_id="REQ-TARGET-ALT",
                document_id="SOP-SYNTH-001",
                document_version="1.2",
                category=RequirementCategory.TOLERANCE_BAND,
                parameter_id="altitude_ft",
                target_value=10000.0,
                tolerance=100.0,
                unit="ft",
                source_reference="SOP-SYNTH-001 Sec 3.1",
                description="Maintain cruise altitude within 100 ft tolerance",
            ),
        ),
    )


@pytest.fixture
def flight_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    """Generate a clean synthetic flight telemetry CSV."""
    csv_file = tmp_path / "flight_telemetry.csv"
    timestamps = [
        "2026-10-14T12:00:00Z",
        "2026-10-14T12:00:01Z",
        "2026-10-14T12:00:02Z",
    ]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "airspeed_kts": [220.0, 260.0, 225.0],  # 260.0 > 250.0 limit
            "altitude_ft": [10000.0, 10020.0, 9990.0],
            "pitch_deg": [2.0, 2.1, 2.0],
            "roll_deg": [0.0, 0.2, -0.1],
            "engine_rpm": [2400.0, 2410.0, 2405.0],
            "nz_g": [1.0, 1.05, 0.98],
            "gear_handle": ["UP", "UP", "UP"],
            "configuration_id": ["CRUISE_CLEAN", "CRUISE_CLEAN", "CRUISE_CLEAN"],
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestReportDataExtractor:
    """Comprehensive test suite for ReportDataExtractor and report models."""

    def test_full_pipeline_extraction(
        self,
        flight_csv: pathlib.Path,
        synthetic_vehicle_profile: VehicleProfile,
        synthetic_envelope: ConfigurationEnvelope,
        synthetic_limit_profile: EngineeringLimitProfile,
        synthetic_procedure_profile: ProcedureRequirementProfile,
    ) -> None:
        """Verify extraction from an end-to-end executed pipeline with vehicle evaluation."""
        veh_req = PipelineVehicleEvaluationRequest(
            vehicle_profile=synthetic_vehicle_profile,
            configuration_envelope=synthetic_envelope,
            limit_profile=synthetic_limit_profile,
            procedure_profile=synthetic_procedure_profile,
            auto_extract_observations=True,
        )
        ev_req = PipelineEvidenceRequest(
            collection_id="EV-COLL-REPORT-TEST",
            selections=EvidenceSourceSelection(
                include_vehicle_evaluation=True,
            ),
        )
        rec_req = PipelineRecommendationRequest(
            policy=RecommendationGenerationPolicy(
                enable_data_verification_rule=True,
                enable_engineering_review_rule=True,
            ),
            use_pipeline_evidence=True,
            use_pipeline_validation=True,
            use_pipeline_quality=True,
            use_pipeline_vehicle_evaluation=True,
        )

        pipeline_res = run_csv_pipeline(
            file_path=flight_csv,
            vehicle_evaluation_request=veh_req,
            evidence_request=ev_req,
            recommendation_request=rec_req,
        )

        # Extract report
        report = ReportDataExtractor.extract(
            pipeline_result=pipeline_res,
            report_title="Flight Test 401 Post-Flight Engineering Evaluation",
            analyst_name="Lead Flight Test Engineer",
            custom_metadata={"test_flight_number": 401, "airframe": "AC-SYNTH-01"},
        )

        # Assert report structure
        assert isinstance(report, EngineeringReport)
        assert report.metadata.title == "Flight Test 401 Post-Flight Engineering Evaluation"
        assert report.metadata.analyst_name == "Lead Flight Test Engineer"
        assert report.metadata.custom_metadata["test_flight_number"] == 401
        assert report.metadata.report_id.startswith("REP-")
        assert report.metadata.generated_at.tzinfo is not None

        # 15 Stages verification
        assert len(report.stage_summaries) == 15
        expected_stages = [
            ("ingestion", 1),
            ("validation", 2),
            ("quality", 3),
            ("normalization", 4),
            ("parameter_validation", 5),
            ("association", 6),
            ("engineering", 7),
            ("statistics", 8),
            ("time_series", 9),
            ("anomaly", 10),
            ("expected_observed", 11),
            ("vehicle_evaluation", 12),
            ("evidence", 13),
            ("contributor_analysis", 14),
            ("recommendation", 15),
        ]
        for idx, (expected_name, expected_order) in enumerate(expected_stages):
            summary = report.stage_summaries[idx]
            assert summary.stage_name == expected_name
            assert summary.stage_order == expected_order

        # Ingestion stage should be executed and success
        assert report.stage_summaries[0].is_executed is True
        assert report.stage_summaries[0].status == StageStatus.SUCCESS
        assert report.stage_summaries[0].notes is not None
        assert "Ingested 3 rows" in report.stage_summaries[0].notes

        # Vehicle evaluation stage should be executed
        veh_stage = report.stage_summaries[11]
        assert veh_stage.stage_name == "vehicle_evaluation"
        assert veh_stage.is_executed is True

        # Evidence stage should be executed
        ev_stage = report.stage_summaries[12]
        assert ev_stage.stage_name == "evidence"
        assert ev_stage.is_executed is True

        # Recommendation stage should be executed
        rec_stage = report.stage_summaries[14]
        assert rec_stage.stage_name == "recommendation"
        assert rec_stage.is_executed is True

        # Data quality checks
        assert report.data_quality.total_rows_ingested == 3
        assert report.data_quality.total_columns_ingested == 9
        assert report.data_quality.is_valid is True
        assert report.data_quality.quality_score is not None

        # Flight context checks
        assert report.flight_context.vehicle_id == "AC-SYNTH-01"
        assert report.flight_context.profile_id == "PROF-SYNTH-X1"
        assert report.flight_context.configuration_id == "CRUISE_CLEAN"

        # Engineering evaluation checks
        assert report.engineering_evaluation.is_evaluated is True
        assert report.engineering_evaluation.total_limits_evaluated > 0
        assert report.engineering_evaluation.limit_exceedance_count == 1  # 260.0 > 250.0 Vne

        # Check epistemic levels in limit evaluations
        for lim_item in report.engineering_evaluation.limit_evaluations:
            assert lim_item.epistemic_level == EpistemicLevel.LIMIT
            if lim_item.parameter_id == "airspeed_kts" and lim_item.observed_value == 260.0:
                assert lim_item.is_exceeded is True
                assert lim_item.boundary_value == 250.0

        for proc_item in report.engineering_evaluation.procedure_evaluations:
            assert proc_item.epistemic_level == EpistemicLevel.PROCEDURAL

        # Investigation intelligence
        assert report.investigation_intelligence.evidence_count > 0
        for ev_item in report.investigation_intelligence.evidence_items:
            assert ev_item.epistemic_level in (EpistemicLevel.LIMIT, EpistemicLevel.PROCEDURAL)

        # Recommendations should have mandatory advisory notice
        for rec_item in report.investigation_intelligence.recommendations:
            assert rec_item.epistemic_level == EpistemicLevel.RECOMMENDATION
            assert "FOR ENGINEERING REVIEW ONLY" in rec_item.advisory_notice

        # Executive summary narrative
        assert "overall pipeline status" in report.executive_summary.high_level_narrative
        assert report.executive_summary.total_stages == 15
        assert report.executive_summary.total_limit_exceedances == 1

        # Provenance summary
        assert report.provenance.source_uri is not None
        assert "ingestion" in report.provenance.stage_provenance_records
        assert "vehicle_evaluation" in report.provenance.stage_provenance_records

    def test_minimal_pipeline_extraction(self, flight_csv: pathlib.Path) -> None:
        """Verify extraction when only basic ingestion, validation, and quality run."""
        pipeline_res = run_csv_pipeline(file_path=flight_csv)

        report = extract_engineering_report(pipeline_result=pipeline_res)

        assert isinstance(report, EngineeringReport)
        assert report.executive_summary.overall_status in (
            ResultStatus.SUCCESS,
            ResultStatus.WARNING,
        )

        # Unexecuted stages should be cleanly documented in Limitations
        assert report.limitations.has_limitations is True
        assert "vehicle_evaluation" in report.limitations.unexecuted_stages
        assert "evidence" in report.limitations.unexecuted_stages
        assert "recommendation" in report.limitations.unexecuted_stages

        # Engineering evaluation should reflect not evaluated
        assert report.engineering_evaluation.is_evaluated is False
        assert report.engineering_evaluation.total_limits_evaluated == 0
        assert report.engineering_evaluation.limit_exceedance_count == 0

        # Limitation items must explain absence of vehicle evaluation
        missing_veh_lim = [
            it for it in report.limitations.items if it.category == "MISSING_ENGINEERING_LIMITS"
        ]
        assert len(missing_veh_lim) == 1
        assert "No vehicle profile" in missing_veh_lim[0].description

        # Stage summaries for skipped stages must have is_executed = False
        skipped_stages = [s for s in report.stage_summaries if not s.is_executed]
        assert len(skipped_stages) == 12  # 15 - 3 executed (ingestion, validation, quality)
        for s in skipped_stages:
            assert s.status == StageStatus.SKIPPED

    def test_upstream_failure_handling(self, tmp_path: pathlib.Path) -> None:
        """Verify downstream stages are marked UPSTREAM_ABORTED when ingestion fails."""
        non_existent_file = tmp_path / "does_not_exist.csv"

        pipeline_res = run_csv_pipeline(file_path=non_existent_file)
        assert pipeline_res.status == ResultStatus.ERROR

        report = ReportDataExtractor.extract(pipeline_result=pipeline_res)

        assert report.executive_summary.overall_status == ResultStatus.ERROR
        assert report.stage_summaries[0].stage_name == "ingestion"
        assert report.stage_summaries[0].status == StageStatus.ERROR
        assert report.stage_summaries[0].has_errors is True

        # Validation was aborted due to ingestion failure
        val_summary = report.stage_summaries[1]
        assert val_summary.stage_name == "validation"
        assert val_summary.status == StageStatus.UPSTREAM_ABORTED
        assert val_summary.is_executed is False

        # Limitations section must capture upstream aborts and ingestion failure
        assert "ingestion" in report.limitations.failed_stages
        assert "validation" in report.limitations.unexecuted_stages

    def test_inconclusive_evaluation_extraction(
        self,
        flight_csv: pathlib.Path,
        synthetic_vehicle_profile: VehicleProfile,
        synthetic_limit_profile: EngineeringLimitProfile,
    ) -> None:
        """Verify inconclusive evaluations (unit mismatch / missing param) are properly flagged."""
        obs_wrong_unit = EvaluationObservation(
            observation_id="OBS-WRONG-UNIT",
            parameter_id="airspeed_kts",
            value=240.0,
            unit="m/s",  # Telemetry in m/s, limit is configured in kts
            source_reference=str(flight_csv),
        )

        veh_req = PipelineVehicleEvaluationRequest(
            vehicle_profile=synthetic_vehicle_profile,
            limit_profile=synthetic_limit_profile,
            explicit_observations=(obs_wrong_unit,),
            auto_extract_observations=False,
            strict_units=True,
        )

        pipeline_res = run_csv_pipeline(
            file_path=flight_csv,
            vehicle_evaluation_request=veh_req,
        )

        report = ReportDataExtractor.extract(pipeline_result=pipeline_res)

        # Limit should be inconclusive
        assert report.engineering_evaluation.limit_inconclusive_count >= 1

        # Limitations section must explicitly caveat this inconclusive check
        inconcl_items = [
            it for it in report.limitations.items if it.category == "INCONCLUSIVE_LIMIT_EVALUATION"
        ]
        assert len(inconcl_items) >= 1
        assert "airspeed_kts" in inconcl_items[0].description
        assert "Do NOT assume parameter was within allowable limits" in inconcl_items[0].impact

    def test_anomaly_and_comparison_extraction(self) -> None:
        """Verify extraction of anomalies and comparisons into report domain models."""
        prov = create_provenance(
            activity_name="test_anom",
            source_uri="test.csv",
            epistemic_level=EpistemicLevel.CALCULATED,
        )
        anom_record = AnomalyRecord(
            anomaly_id="ANOM-01",
            analysis_id="ANALYSIS-01",
            target_column="altitude_ft",
            classification=AnomalyClassification.POINT_ANOMALY,
            severity=AnomalySeverity.HIGH,
            detection_method=DetectionMethod.STATISTICAL,
            observed_value=12500.0,
            expected_value=10000.0,
            deviation_value=2500.0,
            explanation="Altitude spike exceeding 3-sigma band",
            provenance=prov,
        )
        anom_res = AnomalyResult(
            analysis_id="ANALYSIS-01",
            target_column="altitude_ft",
            detection_method=DetectionMethod.STATISTICAL,
            anomalies=(anom_record,),
            status=ResultStatus.WARNING,
        )
        pipe_anom = PipelineAnomalyDetectionResult(
            detector_results=(anom_res,),
            status=ResultStatus.WARNING,
            issues=[],
            provenance=prov,
        )

        eo_comp = ExpectedObservedComparison(
            comparison_id="COMP-01",
            expected=ExpectedValue(
                expected_id="EXP-01",
                parameter_id="airspeed_kts",
                value=250.0,
                unit="kts",
                source=ExpectationSource.TEST_CARD,
                source_reference="TC-01",
                epistemic_level=EpistemicLevel.PROCEDURAL,
            ),
            observed=ObservedValue(
                observed_id="OBS-01",
                parameter_id="airspeed_kts",
                value=255.0,
                unit="kts",
                source_reference="flight.csv",
            ),
            tolerance_evaluated=True,
            within_tolerance=True,
            status=ResultStatus.SUCCESS,
            provenance=prov,
        )
        pipe_eo = PipelineExpectedObservedResult(
            comparisons=(eo_comp,),
            status=ResultStatus.SUCCESS,
            issues=[],
            provenance=prov,
        )

        ingest_res = IngestionResult(
            status=ResultStatus.SUCCESS,
            row_count=100,
            column_names=["timestamp", "airspeed_kts", "altitude_ft"],
            source_path="flight.csv",
        )

        mock_pipeline_result = DataPipelineResult(
            status=ResultStatus.WARNING,
            issues=[],
            provenance=prov,
            ingestion=ingest_res,
            anomaly=pipe_anom,
            expected_observed=pipe_eo,
        )

        report = ReportDataExtractor.extract(mock_pipeline_result)

        assert report.anomaly_and_comparison.total_anomalies == 1
        assert report.anomaly_and_comparison.high_severity_anomalies == 1
        assert len(report.anomaly_and_comparison.anomalies) == 1
        anom_item = report.anomaly_and_comparison.anomalies[0]
        assert anom_item.anomaly_id == "ANOM-01"
        assert anom_item.severity == "HIGH"
        assert anom_item.epistemic_level == EpistemicLevel.CALCULATED

        assert report.anomaly_and_comparison.total_comparisons == 1
        assert report.anomaly_and_comparison.comparisons_within_tolerance == 1
        assert len(report.anomaly_and_comparison.comparisons) == 1
        comp_item = report.anomaly_and_comparison.comparisons[0]
        assert comp_item.comparison_id == "COMP-01"
        assert comp_item.within_tolerance is True
        assert comp_item.epistemic_level == EpistemicLevel.CALCULATED

    def test_source_result_immutability(self, flight_csv: pathlib.Path) -> None:
        """Verify extracting a report does not mutate the source DataPipelineResult."""
        pipeline_res = run_csv_pipeline(file_path=flight_csv)
        original_status = pipeline_res.status
        original_issues = list(pipeline_res.issues)
        original_df = pipeline_res.ingestion.data.copy()

        # Extract report
        _ = ReportDataExtractor.extract(pipeline_result=pipeline_res)

        # Assert no mutations occurred
        assert pipeline_res.status == original_status
        assert pipeline_res.issues == original_issues
        pd.testing.assert_frame_equal(pipeline_res.ingestion.data, original_df)

    def test_report_domain_model_immutability(self, flight_csv: pathlib.Path) -> None:
        """Verify that EngineeringReport and its sections are frozen."""
        pipeline_res = run_csv_pipeline(file_path=flight_csv)
        report = extract_engineering_report(pipeline_result=pipeline_res)

        # Attempting to reassign an attribute on the frozen report should raise ValidationError
        with pytest.raises(ValidationError):
            report.metadata = report.metadata  # type: ignore[misc]

        with pytest.raises(ValidationError):
            report.executive_summary.high_level_narrative = "Mutated"  # type: ignore[misc]

        with pytest.raises(ValidationError):
            report.data_quality.total_rows_ingested = 9999  # type: ignore[misc]

    def test_contributor_analysis_extraction(self) -> None:
        """Verify extraction of contributor correlation rankings."""
        from aeris.core.contributors.models import (
            ContributorAnalysisMethod,
            ContributorAnalysisResult,
        )
        from aeris.core.contributors.pipeline import PipelineContributorAnalysisResult

        prov = create_provenance(
            activity_name="contributor_test",
            source_uri="test.csv",
            epistemic_level=EpistemicLevel.CALCULATED,
        )

        contrib_res = ContributorAnalysisResult(
            analysis_id="CONTRIB-ANALYSIS-01",
            target_column="altitude_ft",
            factor_column="engine_rpm",
            analysis_method=ContributorAnalysisMethod.PEARSON,
            total_input_rows=100,
            usable_observation_count=98,
            excluded_observation_count=2,
            coefficient=0.87,
            p_value=0.001,
            status=ResultStatus.SUCCESS,
            provenance=prov,
        )

        pipe_contrib = PipelineContributorAnalysisResult(
            results=(contrib_res,),
            status=ResultStatus.SUCCESS,
            issues=[],
            provenance=prov,
        )

        ingest_res = IngestionResult(
            status=ResultStatus.SUCCESS,
            row_count=100,
            column_names=["timestamp", "altitude_ft", "engine_rpm"],
            source_path="test.csv",
            provenance=prov,
        )

        pipe_res = DataPipelineResult(
            status=ResultStatus.SUCCESS,
            issues=[],
            provenance=prov,
            ingestion=ingest_res,
            contributor_analysis=pipe_contrib,
        )

        report = ReportDataExtractor.extract(pipe_res)

        assert len(report.investigation_intelligence.contributor_rankings) == 1
        rank = report.investigation_intelligence.contributor_rankings[0]
        assert rank.analysis_id == "CONTRIB-ANALYSIS-01"
        assert rank.target_column == "altitude_ft"
        assert rank.factor_column == "engine_rpm"
        assert rank.coefficient == 0.87
        assert rank.p_value == 0.001
        assert rank.method == "PEARSON"
        assert rank.epistemic_level == EpistemicLevel.CALCULATED

        # Check stage summary notes
        contrib_stage = next(
            s for s in report.stage_summaries if s.stage_name == "contributor_analysis"
        )
        assert contrib_stage.is_executed is True
        assert contrib_stage.status == StageStatus.SUCCESS
        assert contrib_stage.notes is not None
        assert "1 contributor correlation analyses evaluated" in contrib_stage.notes

    def test_normalization_and_parameter_validation_extraction(self) -> None:
        """Verify extraction of normalization transformations and semantic parameter validation."""
        from aeris.core.results.types import (
            NormalizationResult,
            ParameterValidationResult,
            TransformationRecord,
            TransformationType,
        )

        prov = create_provenance(
            activity_name="norm_param_test",
            source_uri="test.csv",
            epistemic_level=EpistemicLevel.CALCULATED,
        )

        tx_record = TransformationRecord(
            transformation_id="TX-KTS-TO-MS",
            transformation_type=TransformationType.UNIT_CONVERSION,
            source_column="airspeed_kts",
            destination_column="airspeed_mps",
            source_unit="kts",
            destination_unit="m/s",
            parameters={"scale": 0.514444},
            is_lossless=True,
        )

        norm_res = NormalizationResult(
            status=ResultStatus.SUCCESS,
            transformations=[tx_record],
            input_row_count=50,
            output_row_count=50,
            input_columns=["airspeed_kts"],
            output_columns=["airspeed_mps"],
            provenance=prov,
        )

        param_val_res = ParameterValidationResult(
            status=ResultStatus.WARNING,
            validated_parameters=["airspeed_kts", "altitude_ft"],
            failed_parameters=["unknown_custom_sensor"],
            provenance=prov,
        )

        ingest_res = IngestionResult(
            status=ResultStatus.SUCCESS,
            row_count=50,
            column_names=["airspeed_kts", "altitude_ft", "unknown_custom_sensor"],
            source_path="test.csv",
            provenance=prov,
        )

        pipe_res = DataPipelineResult(
            status=ResultStatus.WARNING,
            issues=[],
            provenance=prov,
            ingestion=ingest_res,
            normalization=norm_res,
            parameter_validation=param_val_res,
        )

        report = ReportDataExtractor.extract(pipe_res)

        assert report.data_quality.normalization_applied is True
        assert report.data_quality.transformations_count == 1
        assert report.data_quality.parameter_validation_passed is False
        assert report.data_quality.validated_parameters == ("airspeed_kts", "altitude_ft")
        assert report.data_quality.failed_parameters == ("unknown_custom_sensor",)

        norm_stage = next(s for s in report.stage_summaries if s.stage_name == "normalization")
        assert norm_stage.is_executed is True
        assert norm_stage.notes is not None
        assert "1 explicit transformations applied" in norm_stage.notes

        pv_stage = next(s for s in report.stage_summaries if s.stage_name == "parameter_validation")
        assert pv_stage.is_executed is True
        assert pv_stage.status == StageStatus.WARNING
        assert pv_stage.notes is not None
        assert "2 parameters validated, 1 failed" in pv_stage.notes

    def test_test_point_association_extraction(self) -> None:
        """Verify extraction of test point association metadata and row counts."""
        from aeris.core.test_operations.association import TestPointAssociationResult

        prov = create_provenance(
            activity_name="assoc_test",
            source_uri="test.csv",
            epistemic_level=EpistemicLevel.CALCULATED,
        )

        assoc_res = TestPointAssociationResult(
            status=ResultStatus.SUCCESS,
            test_point_id="TP-CRUISE-STAB-01",
            observation_id="OBS-INTERVAL-401",
            timestamp_column="timestamp",
            row_count=120,
            provenance=prov,
        )

        ingest_res = IngestionResult(
            status=ResultStatus.SUCCESS,
            row_count=500,
            column_names=["timestamp", "airspeed_kts"],
            source_path="test.csv",
            provenance=prov,
        )

        pipe_res = DataPipelineResult(
            status=ResultStatus.SUCCESS,
            issues=[],
            provenance=prov,
            ingestion=ingest_res,
            association=assoc_res,
        )

        report = ReportDataExtractor.extract(pipe_res)

        assert report.flight_context.test_point_id == "TP-CRUISE-STAB-01"
        assert report.flight_context.observation_id == "OBS-INTERVAL-401"
        assert report.flight_context.associated_rows_count == 120

        assoc_stage = next(s for s in report.stage_summaries if s.stage_name == "association")
        assert assoc_stage.is_executed is True
        assert assoc_stage.notes is not None
        assert "Associated test point TP-CRUISE-STAB-01 with 120 rows" in assoc_stage.notes
