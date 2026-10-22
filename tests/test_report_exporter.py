"""
Tests for AERIS Safe Report Exporter & Pipeline Orchestrator Integration — Step 14g.

Validates:
1. Successful Markdown (.md, .markdown) and HTML (.html, .htm) report exports.
2. Direct end-to-end extraction and export from DataPipelineResult instances.
3. Convenience method integration on DataPipelineOrchestrator.
4. Automatic format inference from file extensions and rejection of mismatched formats.
5. Rejection of invalid extensions and unknown format strings.
6. Path traversal rejection (preventing '..' escapes and base_dir violations).
7. Accidental overwrite protection (overwrite=False fails safely with ERR_FILE_EXISTS).
8. Explicit overwrite support (overwrite=True updates destination atomically).
9. Atomic write failure safety: temporary file cleanup on disk failure, no partial corruption.
10. Automatic creation of missing parent directory hierarchies.
11. Rejection of directory targets and empty/whitespace paths.
12. Strict immutability of source EngineeringReport domain models and DataPipelineResults.
13. HTML export with embedded Plotly figures and offline/online asset configurations.
14. Graceful handling of degraded or partial pipeline results without false positives.
"""

from __future__ import annotations

import datetime
import pathlib
from unittest.mock import patch

import pandas as pd
import plotly.graph_objects as go
import pytest

from aeris.core.evidence.pipeline import EvidenceSourceSelection, PipelineEvidenceRequest
from aeris.core.limits.models import (
    EngineeringLimitProfile,
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
)
from aeris.core.pipeline.orchestrator import DataPipelineOrchestrator, run_csv_pipeline
from aeris.core.provenance.tracker import EpistemicLevel
from aeris.core.recommendation.generation_models import RecommendationGenerationPolicy
from aeris.core.recommendation.pipeline import PipelineRecommendationRequest
from aeris.core.reporting import (
    AnomalyAndComparisonSummary,
    DataQualitySummary,
    EngineeringEvaluationSummary,
    EngineeringReport,
    ExecutiveSummary,
    FlightContextSummary,
    InvestigationIntelligenceSummary,
    LimitationsSummary,
    ProvenanceSummary,
    ReportExporter,
    ReportExportFormat,
    ReportExportResult,
    ReportMetadata,
    StageExecutionSummary,
    StageStatus,
    export_engineering_report,
    export_pipeline_report,
    export_report,
    extract_engineering_report,
)
from aeris.core.results.types import DataPipelineResult, ResultStatus
from aeris.core.sop.models import (
    ProcedureRequirement,
    ProcedureRequirementProfile,
    RequirementCategory,
)
from aeris.core.vehicle.envelope_models import ConfigurationEnvelope
from aeris.core.vehicle.profile_models import VehicleProfile
from aeris.core.vehicle_evaluation.pipeline import PipelineVehicleEvaluationRequest

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_vehicle_profile() -> VehicleProfile:
    """Return a standard synthetic VehicleProfile."""
    return VehicleProfile(
        profile_id="PROF-SYNTH-X1",
        vehicle_id="AC-SYNTH-01",
        profile_version="1.0.0",
        name="Synthetic Model X1",
        supported_parameter_ids=("airspeed_kts", "altitude_ft", "nz_g", "gear_handle"),
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
    """Return a synthetic limit profile with upper bound airspeed limit."""
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
                source_reference="AFM-SEC2",
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
            "airspeed_kts": [220.0, 260.0, 225.0],  # 260.0 breaches 250.0 limit
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


@pytest.fixture
def pipeline_result(
    flight_csv: pathlib.Path,
    synthetic_vehicle_profile: VehicleProfile,
    synthetic_envelope: ConfigurationEnvelope,
    synthetic_limit_profile: EngineeringLimitProfile,
    synthetic_procedure_profile: ProcedureRequirementProfile,
) -> DataPipelineResult:
    """Return a full pipeline result execution."""
    veh_req = PipelineVehicleEvaluationRequest(
        vehicle_profile=synthetic_vehicle_profile,
        configuration_envelope=synthetic_envelope,
        limit_profile=synthetic_limit_profile,
        procedure_profile=synthetic_procedure_profile,
        auto_extract_observations=True,
    )
    ev_req = PipelineEvidenceRequest(
        collection_id="EV-COLL-EXP-TEST",
        selections=EvidenceSourceSelection(include_vehicle_evaluation=True),
    )
    rec_req = PipelineRecommendationRequest(
        policy=RecommendationGenerationPolicy(
            enable_data_verification_rule=True,
            enable_engineering_review_rule=True,
        ),
        use_pipeline_evidence=True,
        use_pipeline_validation=True,
    )
    return run_csv_pipeline(
        file_path=flight_csv,
        vehicle_evaluation_request=veh_req,
        evidence_request=ev_req,
        recommendation_request=rec_req,
    )


@pytest.fixture
def sample_report(pipeline_result: DataPipelineResult) -> EngineeringReport:
    """Return an extracted EngineeringReport."""
    return extract_engineering_report(
        pipeline_result=pipeline_result,
        report_title="AERIS Flight Test 501 Engineering Report",
        analyst_name="Test Lead Chuck Yeager",
        custom_metadata={"Sortie": "Envelope Clearance", "Target M": 0.85},
    )


@pytest.fixture
def minimal_report() -> EngineeringReport:
    """Return a minimal valid EngineeringReport domain model."""
    now = datetime.datetime.now(datetime.UTC)
    return EngineeringReport(
        metadata=ReportMetadata(
            report_id="REP-MIN-001",
            title="AERIS Minimal Report",
            analyst_name="Ada Lovelace",
            generated_at=now,
            pipeline_run_id="RUN-MIN-001",
            source_file="telemetry.csv",
            source_file_hash="abc123hash",
            execution_duration_seconds=0.05,
            pipeline_version="1.0.0",
        ),
        executive_summary=ExecutiveSummary(
            overall_status=ResultStatus.SUCCESS,
            total_stages=15,
            executed_stages_count=2,
            failed_stages_count=0,
            warning_stages_count=0,
            skipped_stages_count=13,
            total_issues_count=0,
            total_critical_or_error_count=0,
            total_limit_exceedances=0,
            total_procedure_violations=0,
            total_anomalies_detected=0,
            total_recommendations_generated=0,
            high_level_narrative="Nominal checkout baseline run.",
        ),
        stage_summaries=(
            StageExecutionSummary(
                stage_name="ingestion",
                stage_order=1,
                status=StageStatus.SUCCESS,
                is_executed=True,
                issue_count=0,
                error_count=0,
                warning_count=0,
            ),
            StageExecutionSummary(
                stage_name="validation",
                stage_order=2,
                status=StageStatus.SUCCESS,
                is_executed=True,
                issue_count=0,
                error_count=0,
                warning_count=0,
            ),
        ),
        data_quality=DataQualitySummary(
            is_valid=True,
            quality_score=1.0,
            total_rows_ingested=10,
            total_columns_ingested=5,
            columns_ingested=("timestamp", "airspeed_kts", "altitude_ft"),
            columns_evaluated=("airspeed_kts", "altitude_ft"),
            violation_count=0,
            quality_metrics={},
            quality_summary_text="",
            normalization_applied=False,
            transformations_count=0,
            parameter_validation_passed=True,
            validated_parameters=("airspeed_kts", "altitude_ft"),
            failed_parameters=(),
            issues=(),
        ),
        flight_context=FlightContextSummary(
            vehicle_id="AC-SYNTH-MIN",
            profile_id=None,
            profile_version=None,
            configuration_id=None,
            test_card_id=None,
            test_point_id=None,
            observation_id=None,
            associated_rows_count=None,
            configuration_discrepancies=(),
        ),
        engineering_evaluation=EngineeringEvaluationSummary(
            is_evaluated=False,
            total_limits_evaluated=0,
            limit_exceedance_count=0,
            limit_inconclusive_count=0,
            total_procedures_evaluated=0,
            procedure_violation_count=0,
            procedure_inconclusive_count=0,
            limit_evaluations=(),
            procedure_evaluations=(),
            calculations_performed=(),
        ),
        anomaly_and_comparison=AnomalyAndComparisonSummary(
            total_anomalies=0,
            high_severity_anomalies=0,
            medium_severity_anomalies=0,
            low_severity_anomalies=0,
            info_severity_anomalies=0,
            anomalies=(),
            total_comparisons=0,
            comparisons_within_tolerance=0,
            comparisons_out_of_tolerance=0,
            comparisons=(),
        ),
        investigation_intelligence=InvestigationIntelligenceSummary(
            evidence_count=0,
            evidence_items=(),
            contributor_rankings=(),
            recommendation_count=0,
            rules_evaluated=(),
            rules_triggered=(),
            recommendations=(),
        ),
        limitations=LimitationsSummary(
            has_limitations=False,
            unexecuted_stages=("anomaly", "recommendation"),
            failed_stages=(),
            inconclusive_checks_count=0,
            items=(),
        ),
        provenance=ProvenanceSummary(
            root_provenance_id="PROV-ROOT-MIN",
            source_uri="telemetry.csv",
            activity_name="test_pipeline",
            epistemic_level=EpistemicLevel.CALCULATED,
        ),
    )


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestReportExporterBasics:
    """Test core export functionality for Markdown and HTML."""

    def test_export_markdown_success(
        self,
        sample_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Export an EngineeringReport to Markdown and verify contents and provenance."""
        out_file = tmp_path / "engineering_report.md"
        exporter = ReportExporter()

        result = exporter.export(sample_report, out_file)

        assert isinstance(result, ReportExportResult)
        assert result.status == ResultStatus.SUCCESS
        assert result.is_successful is True
        assert result.format == ReportExportFormat.MARKDOWN
        assert result.output_path == out_file.resolve()
        assert result.bytes_written > 0
        assert out_file.is_file()

        content = out_file.read_text(encoding="utf-8")
        assert "# AERIS Flight Test 501 Engineering Report" in content
        assert "Test Lead Chuck Yeager" in content
        assert "[OBSERVED]" in content
        assert "[LIMIT]" in content
        assert result.provenance is not None
        assert result.provenance.source_checksum is not None
        assert result.provenance.parameters["format"] == "markdown"

    def test_export_html_success(
        self,
        sample_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Export an EngineeringReport to HTML and verify structure and accessibility."""
        out_file = tmp_path / "engineering_report.html"

        result = export_report(sample_report, out_file)

        assert result.status == ResultStatus.SUCCESS
        assert result.format == ReportExportFormat.HTML
        assert result.output_path == out_file.resolve()
        assert result.bytes_written > 0
        assert out_file.is_file()

        content = out_file.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "AERIS Flight Test 501 Engineering Report" in content
        assert "Chuck Yeager" in content
        assert "@media print" in content

    def test_alias_export_engineering_report(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Verify the explicit alias export_engineering_report matches export_report."""
        out_file = tmp_path / "alias_report.markdown"
        result = export_engineering_report(minimal_report, out_file)

        assert result.status == ResultStatus.SUCCESS
        assert result.format == ReportExportFormat.MARKDOWN
        assert out_file.is_file()


class TestPipelineResultIntegration:
    """Test direct export from completed DataPipelineResult instances."""

    def test_export_pipeline_report_markdown(
        self,
        pipeline_result: DataPipelineResult,
        tmp_path: pathlib.Path,
    ) -> None:
        """Extract and export directly from a DataPipelineResult to Markdown."""
        out_file = tmp_path / "pipeline_report.md"

        result = export_pipeline_report(
            pipeline_result=pipeline_result,
            output_path=out_file,
            report_title="Direct Pipeline Export",
            analyst_name="Dr. von Braun",
        )

        assert result.status == ResultStatus.SUCCESS
        assert result.format == ReportExportFormat.MARKDOWN
        assert out_file.is_file()

        content = out_file.read_text(encoding="utf-8")
        assert "# Direct Pipeline Export" in content
        assert "Dr. von Braun" in content

    def test_export_pipeline_report_html(
        self,
        pipeline_result: DataPipelineResult,
        tmp_path: pathlib.Path,
    ) -> None:
        """Extract and export directly from a DataPipelineResult to HTML."""
        out_file = tmp_path / "pipeline_report.html"

        result = export_pipeline_report(
            pipeline_result=pipeline_result,
            output_path=out_file,
            report_title="Direct HTML Pipeline Export",
        )

        assert result.status == ResultStatus.SUCCESS
        assert result.format == ReportExportFormat.HTML
        assert out_file.is_file()

        content = out_file.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in content
        assert "Direct HTML Pipeline Export" in content

    def test_orchestrator_export_report_convenience_method(
        self,
        pipeline_result: DataPipelineResult,
        tmp_path: pathlib.Path,
    ) -> None:
        """Verify convenience method on DataPipelineOrchestrator."""
        orchestrator = DataPipelineOrchestrator()
        out_file = tmp_path / "orchestrator_export.html"

        result = orchestrator.export_report(
            pipeline_result=pipeline_result,
            output_path=out_file,
            report_title="Orchestrator Direct Call",
        )

        assert result.status == ResultStatus.SUCCESS
        assert result.output_path == out_file.resolve()
        assert out_file.is_file()

    def test_source_pipeline_result_immutability(
        self,
        pipeline_result: DataPipelineResult,
        tmp_path: pathlib.Path,
    ) -> None:
        """Verify pipeline_result is not mutated during extraction and export."""
        # Record baseline hashes / attributes
        original_status = pipeline_result.status
        original_issues_count = len(pipeline_result.issues)
        original_ingestion_rows = pipeline_result.ingestion.row_count

        out_file = tmp_path / "immutability_test.md"
        result = export_pipeline_report(pipeline_result, out_file)

        assert result.status == ResultStatus.SUCCESS
        assert pipeline_result.status == original_status
        assert len(pipeline_result.issues) == original_issues_count
        assert pipeline_result.ingestion.row_count == original_ingestion_rows


class TestPathValidationAndSecurity:
    """Test path resolution, directory confinement, traversal prevention, and extension checks."""

    def test_infer_format_from_extensions(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Infer Markdown and HTML formats from recognized file extensions."""
        for ext, expected_fmt in [
            (".md", ReportExportFormat.MARKDOWN),
            (".markdown", ReportExportFormat.MARKDOWN),
            (".html", ReportExportFormat.HTML),
            (".htm", ReportExportFormat.HTML),
        ]:
            out_file = tmp_path / f"test_infer{ext}"
            res = export_report(minimal_report, out_file)
            assert res.status == ResultStatus.SUCCESS
            assert res.format == expected_fmt
            assert out_file.is_file()

    def test_mismatched_extension_and_format_rejected(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Reject mismatch when explicit format differs from file extension."""
        out_file = tmp_path / "report.html"

        # Explicit markdown requested for .html file
        res = export_report(
            minimal_report,
            out_file,
            format=ReportExportFormat.MARKDOWN,
        )
        assert res.status == ResultStatus.ERROR
        assert res.has_errors is True
        assert res.issues[0].issue_code == "ERR_PATH_VALIDATION"
        assert not out_file.exists()

        # With raise_on_error=True
        with pytest.raises(ValueError, match="does not match format"):
            export_report(
                minimal_report,
                out_file,
                format=ReportExportFormat.MARKDOWN,
                raise_on_error=True,
            )

    def test_unsupported_extension_rejected(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Reject files with unsupported extensions (.txt, .pdf, .json)."""
        for bad_ext in (".txt", ".pdf", ".json", ".csv"):
            out_file = tmp_path / f"report{bad_ext}"
            res = export_report(minimal_report, out_file)
            assert res.status == ResultStatus.ERROR
            assert res.issues[0].issue_code == "ERR_PATH_VALIDATION"
            assert not out_file.exists()

    def test_invalid_format_string_rejected(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Reject invalid format string."""
        out_file = tmp_path / "report.md"
        res = export_report(minimal_report, out_file, format="pdf")
        assert res.status == ResultStatus.ERROR
        assert res.issues[0].issue_code == "ERR_PATH_VALIDATION"
        assert "Unsupported report export format" in res.issues[0].message

        with pytest.raises(ValueError, match="Unsupported report export format"):
            export_report(minimal_report, out_file, format="pdf", raise_on_error=True)

    def test_empty_or_whitespace_path_rejected(
        self,
        minimal_report: EngineeringReport,
    ) -> None:
        """Reject empty, whitespace, or None output paths."""
        for invalid_path in ("", "   ", "\t"):
            res = export_report(minimal_report, invalid_path)
            assert res.status == ResultStatus.ERROR
            assert res.issues[0].issue_code == "ERR_PATH_VALIDATION"

            with pytest.raises(ValueError, match="empty or whitespace"):
                export_report(minimal_report, invalid_path, raise_on_error=True)

    def test_existing_directory_as_target_rejected(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Reject export when target path is an existing directory."""
        dir_target = tmp_path / "reports_dir"
        dir_target.mkdir()

        res = export_report(minimal_report, dir_target, format=ReportExportFormat.MARKDOWN)
        assert res.status == ResultStatus.ERROR
        assert res.issues[0].issue_code == "ERR_PATH_VALIDATION"
        assert "points to an existing directory" in res.issues[0].message

    def test_path_traversal_without_base_dir_rejected(
        self,
        minimal_report: EngineeringReport,
    ) -> None:
        """Reject paths with explicit '..' traversal parts."""
        res = export_report(minimal_report, "../forbidden_report.md")
        assert res.status == ResultStatus.ERROR
        assert res.issues[0].issue_code == "ERR_PATH_VALIDATION"
        assert "traversal components" in res.issues[0].message

        with pytest.raises(ValueError, match="traversal components"):
            export_report(minimal_report, "../forbidden_report.md", raise_on_error=True)

    def test_path_traversal_escaping_base_dir_rejected(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Reject paths that resolve outside base_dir."""
        allowed_dir = tmp_path / "allowed"
        allowed_dir.mkdir()

        traversal_path = allowed_dir / ".." / "escaped.md"

        res = export_report(
            minimal_report,
            traversal_path,
            base_dir=allowed_dir,
        )
        assert res.status == ResultStatus.ERROR
        assert res.issues[0].issue_code == "ERR_PATH_VALIDATION"
        assert "outside base directory" in res.issues[0].message

        with pytest.raises(ValueError, match="outside base directory"):
            export_report(
                minimal_report,
                traversal_path,
                base_dir=allowed_dir,
                raise_on_error=True,
            )

    def test_automatic_creation_of_missing_parent_directories(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Create deeply nested directories automatically during export."""
        deep_target = tmp_path / "a" / "b" / "c" / "deep_report.md"
        assert not deep_target.parent.exists()

        res = export_report(minimal_report, deep_target)
        assert res.status == ResultStatus.SUCCESS
        assert deep_target.is_file()


class TestOverwriteProtection:
    """Test file existence checks and explicit overwrite controls."""

    def test_overwrite_false_prevents_clobbering(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Default overwrite=False returns error result when file exists."""
        target = tmp_path / "existing.md"
        target.write_text("PRIOR CONTENT", encoding="utf-8")

        res = export_report(minimal_report, target, overwrite=False)

        assert res.status == ResultStatus.ERROR
        assert res.issues[0].issue_code == "ERR_FILE_EXISTS"
        assert "already exists and overwrite is disabled" in res.issues[0].message
        assert target.read_text(encoding="utf-8") == "PRIOR CONTENT"

    def test_overwrite_false_with_raise_on_error(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Default overwrite=False raises FileExistsError when raise_on_error=True."""
        target = tmp_path / "existing2.md"
        target.write_text("PRIOR CONTENT", encoding="utf-8")

        with pytest.raises(FileExistsError, match="already exists"):
            export_report(minimal_report, target, overwrite=False, raise_on_error=True)

        assert target.read_text(encoding="utf-8") == "PRIOR CONTENT"

    def test_overwrite_true_atomically_updates_file(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Explicit overwrite=True updates the destination file."""
        target = tmp_path / "update_target.md"
        target.write_text("OLD CONTENT", encoding="utf-8")

        res = export_report(minimal_report, target, overwrite=True)

        assert res.status == ResultStatus.SUCCESS
        new_content = target.read_text(encoding="utf-8")
        assert "OLD CONTENT" not in new_content
        assert "# AERIS Minimal Report" in new_content


class TestAtomicWriteAndFailureSafety:
    """Test atomic file writing and temporary file cleanup on disk failure."""

    def test_cleanup_temporary_file_on_write_failure(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Verify temporary files are cleaned up if a write failure occurs."""
        target = tmp_path / "fail_report.md"

        with (
            patch("os.fsync", side_effect=OSError("Disk full error")),
            pytest.raises(OSError, match="Disk full error"),
        ):
            export_report(minimal_report, target, raise_on_error=True)

        # Confirm target does not exist
        assert not target.exists()

        # Confirm no temporary files were left behind in tmp_path
        tmp_files = list(tmp_path.glob(".aeris_export_tmp_*"))
        assert len(tmp_files) == 0

    def test_write_failure_returns_error_result_when_not_strict(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Write failure returns ReportExportResult with status=ERROR when raise_on_error=False."""
        target = tmp_path / "fail_report2.md"

        with patch("os.fsync", side_effect=OSError("I/O device fault")):
            res = export_report(minimal_report, target, raise_on_error=False)

        assert res.status == ResultStatus.ERROR
        assert res.issues[0].issue_code == "ERR_WRITE_FAILED"
        assert not target.exists()

        tmp_files = list(tmp_path.glob(".aeris_export_tmp_*"))
        assert len(tmp_files) == 0

    def test_existing_file_preserved_if_overwrite_write_fails(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """If overwrite fails during write, the original target file remains intact."""
        target = tmp_path / "preserve_on_fail.md"
        target.write_text("CANNOT BE CORRUPTED", encoding="utf-8")

        with patch("os.fsync", side_effect=OSError("Simulated crash during write")):
            res = export_report(minimal_report, target, overwrite=True, raise_on_error=False)

        assert res.status == ResultStatus.ERROR
        assert target.read_text(encoding="utf-8") == "CANNOT BE CORRUPTED"

        tmp_files = list(tmp_path.glob(".aeris_export_tmp_*"))
        assert len(tmp_files) == 0


class TestFiguresAndDegradedPipelines:
    """Test figures embedding and export of partial/degraded pipeline results."""

    def test_html_export_with_embedded_plotly_figure(
        self,
        minimal_report: EngineeringReport,
        tmp_path: pathlib.Path,
    ) -> None:
        """Export HTML report with embedded Plotly figure."""
        out_file = tmp_path / "report_with_figure.html"

        fig = go.Figure(
            data=[go.Scatter(x=[1, 2, 3], y=[10, 20, 30], mode="lines", name="Test")],
            layout=go.Layout(title="Telemetry Trend"),
        )

        res = export_report(
            minimal_report,
            out_file,
            figures=[fig],
            include_plotlyjs=False,
        )

        assert res.status == ResultStatus.SUCCESS
        content = out_file.read_text(encoding="utf-8")
        assert "plotly-graph-div" in content or "js-plotly-plot" in content

    def test_export_degraded_pipeline_result(
        self,
        flight_csv: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        """Export report from a minimal pipeline result where optional stages were not run."""
        # Run barebones pipeline (only ingestion, validation, quality)
        basic_res = run_csv_pipeline(file_path=flight_csv)

        out_file = tmp_path / "degraded_pipeline.md"
        res = export_pipeline_report(
            pipeline_result=basic_res,
            output_path=out_file,
            report_title="Degraded Pipeline Report",
        )

        assert res.status == ResultStatus.SUCCESS
        assert out_file.is_file()
        content = out_file.read_text(encoding="utf-8")
        assert "# Degraded Pipeline Report" in content
        assert "Analytical Limitations, Caveats & Data Gaps" in content
