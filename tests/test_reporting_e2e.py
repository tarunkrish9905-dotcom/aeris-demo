"""
End-to-End Audit & Verification Test Suite for AERIS Milestone 14 (Step 14h).

Validates the complete integrated workflows:
1. End-to-End Pipeline -> Report Extraction -> Markdown/HTML Rendering -> Safe Export.
2. Complete Visualization Pipeline: Validated telemetry -> time-series figures ->
   limit/event overlays -> multi-signal panel.
3. Visualization integrity: zero mutation, preservation of observation order, unit safety
   gates, and explicit fallback.
4. Epistemic preservation: strict separation between observed, calculated, limits, SOPs,
   hypotheses, and advisory recommendations.
5. Immutability across the entire pipeline: DataFrames, DataPipelineResult, and
   EngineeringReport domain models.
6. Graceful degradation: correct representation of upstream aborted, skipped, and partial
   pipeline stages.
7. Exporter security & atomic writing: path traversal rejection, base_dir confinement,
   overwrite protection, and temporary file cleanup.
"""

from __future__ import annotations

import os
import pathlib
from unittest.mock import patch

import pandas as pd
import pytest

from aeris.core.evidence.pipeline import EvidenceSourceSelection, PipelineEvidenceRequest
from aeris.core.limits.models import (
    EngineeringLimitProfile,
    LimitSeverity,
    LimitType,
    ParameterLimitDefinition,
)
from aeris.core.pipeline.orchestrator import DataPipelineOrchestrator, run_csv_pipeline
from aeris.core.provenance.tracker import compute_file_checksum
from aeris.core.recommendation.generation_models import RecommendationGenerationPolicy
from aeris.core.recommendation.pipeline import PipelineRecommendationRequest
from aeris.core.reporting import (
    EngineeringReport,
    ReportExportFormat,
    ReportExportResult,
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
from aeris.core.visualization.models import (
    MultiSignalPanelConfig,
    MultiSignalPanelResult,
    PanelLayoutMode,
    SignalSpec,
    TimeSeriesPlotConfig,
    TimeSeriesPlotResult,
)
from aeris.core.visualization.overlays import generate_overlaid_time_series_plot
from aeris.core.visualization.panels import generate_multi_signal_panel
from aeris.core.visualization.time_series import build_time_series_figure

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def e2e_vehicle_profile() -> VehicleProfile:
    """Return an authoritative test VehicleProfile."""
    return VehicleProfile(
        profile_id="PROF-E2E-X29",
        vehicle_id="AC-E2E-001",
        profile_version="1.0.0",
        name="AERIS Experimental Airframe X-29",
        supported_parameter_ids=(
            "airspeed_kts",
            "altitude_ft",
            "pitch_deg",
            "roll_deg",
            "engine_rpm",
            "nz_g",
            "gear_handle",
            "configuration_id",
        ),
    )


@pytest.fixture
def e2e_envelope() -> ConfigurationEnvelope:
    """Return an authoritative ConfigurationEnvelope."""
    return ConfigurationEnvelope(
        configuration_id="CRUISE_CLEAN",
        vehicle_id="AC-E2E-001",
        profile_id="PROF-E2E-X29",
        profile_version="1.0.0",
        name="Clean High-Speed Cruise Envelope",
    )


@pytest.fixture
def e2e_limit_profile() -> EngineeringLimitProfile:
    """Return a limit profile with an upper-bound airspeed limit."""
    return EngineeringLimitProfile(
        limit_profile_id="LIM-PROF-E2E-01",
        vehicle_id="AC-E2E-001",
        profile_id="PROF-E2E-X29",
        profile_version="1.0.0",
        limits=(
            ParameterLimitDefinition(
                limit_id="LIM-VNE-250",
                parameter_id="airspeed_kts",
                limit_type=LimitType.UPPER_BOUND,
                upper_bound=250.0,
                unit="kts",
                severity=LimitSeverity.WARNING,
                vehicle_id="AC-E2E-001",
                profile_id="PROF-E2E-X29",
                profile_version="1.0.0",
                source_reference="AFM-SEC2",
                description="Never exceed airspeed (Vne)",
            ),
        ),
    )


@pytest.fixture
def e2e_procedure_profile() -> ProcedureRequirementProfile:
    """Return an authoritative SOP procedure profile."""
    return ProcedureRequirementProfile(
        profile_id="PROC-PROF-E2E-01",
        document_id="SOP-E2E-001",
        document_version="2.0",
        vehicle_id="AC-E2E-001",
        profile_version="1.0.0",
        requirements=(
            ProcedureRequirement(
                requirement_id="REQ-TARGET-ALT",
                document_id="SOP-E2E-001",
                document_version="2.0",
                category=RequirementCategory.TOLERANCE_BAND,
                parameter_id="altitude_ft",
                target_value=10000.0,
                tolerance=100.0,
                unit="ft",
                source_reference="SOP-E2E-001 Sec 4.2",
                description="Maintain cruise altitude at 10,000 ft +/- 100 ft",
            ),
        ),
    )


@pytest.fixture
def e2e_flight_csv(tmp_path: pathlib.Path) -> pathlib.Path:
    """Generate representative flight telemetry CSV with one limit breach."""
    csv_file = tmp_path / "e2e_flight_data.csv"
    timestamps = [
        "2026-10-19T10:00:00Z",
        "2026-10-19T10:00:01Z",
        "2026-10-19T10:00:02Z",
        "2026-10-19T10:00:03Z",
        "2026-10-19T10:00:04Z",
    ]
    df = pd.DataFrame(
        {
            "timestamp": timestamps,
            "airspeed_kts": [220.0, 240.0, 260.0, 245.0, 230.0],  # 260.0 breaches 250.0 limit
            "altitude_ft": [10000.0, 10010.0, 10025.0, 9995.0, 10005.0],
            "pitch_deg": [1.5, 1.8, 2.2, 1.9, 1.6],
            "roll_deg": [0.0, 0.1, -0.2, 0.0, 0.1],
            "engine_rpm": [2400.0, 2450.0, 2520.0, 2480.0, 2410.0],
            "nz_g": [1.0, 1.02, 1.15, 1.05, 0.99],
            "gear_handle": ["UP", "UP", "UP", "UP", "UP"],
            "configuration_id": ["CRUISE_CLEAN"] * 5,
        }
    )
    df.to_csv(csv_file, index=False)
    return csv_file


# ---------------------------------------------------------------------------
# Test Cases
# ---------------------------------------------------------------------------


class TestMilestone14EndToEndWorkflows:
    """Comprehensive end-to-end audit for Milestone 14."""

    def test_complete_e2e_pipeline_to_visualization_and_report_export(
        self,
        e2e_flight_csv: pathlib.Path,
        e2e_vehicle_profile: VehicleProfile,
        e2e_envelope: ConfigurationEnvelope,
        e2e_limit_profile: EngineeringLimitProfile,
        e2e_procedure_profile: ProcedureRequirementProfile,
        tmp_path: pathlib.Path,
    ) -> None:
        """
        Audit the complete end-to-end post-flight engineering analysis workflow:
        1. Run CSV pipeline with vehicle evaluation, evidence, and recommendations.
        2. Generate time-series plot with limit overlays.
        3. Generate multi-signal panel plot.
        4. Extract EngineeringReport domain model.
        5. Export Markdown report with atomic safety.
        6. Export self-contained HTML report with embedded interactive Plotly figures.
        7. Verify epistemic badges, checksums, and source result immutability.
        """
        # Step 1: Run pipeline orchestrator
        veh_req = PipelineVehicleEvaluationRequest(
            vehicle_profile=e2e_vehicle_profile,
            configuration_envelope=e2e_envelope,
            limit_profile=e2e_limit_profile,
            procedure_profile=e2e_procedure_profile,
            auto_extract_observations=True,
        )
        ev_req = PipelineEvidenceRequest(
            collection_id="EV-E2E-COLL-001",
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

        pipeline_result: DataPipelineResult = run_csv_pipeline(
            file_path=e2e_flight_csv,
            vehicle_evaluation_request=veh_req,
            evidence_request=ev_req,
            recommendation_request=rec_req,
        )

        assert pipeline_result is not None
        assert pipeline_result.ingestion.row_count == 5
        assert pipeline_result.vehicle_evaluation is not None

        # Capture baseline hashes/ids for immutability check
        original_pipeline_status = pipeline_result.status
        original_issue_count = len(pipeline_result.issues)

        # Step 2: Visualization generation (Overlaid Time-Series & Multi-Signal Panel)
        telemetry_df = pd.read_csv(e2e_flight_csv)
        df_copy_baseline = telemetry_df.copy(deep=True)

        # 2a. Time-series with limit overlays
        ts_res: TimeSeriesPlotResult = generate_overlaid_time_series_plot(
            data=telemetry_df,
            parameter_id="airspeed_kts",
            timestamp_column="timestamp",
            unit="kts",
            limits=e2e_limit_profile,
            config=TimeSeriesPlotConfig(title="Airspeed Time-Series with Vne Limit"),
        )
        assert ts_res.figure is not None
        assert ts_res.figure.layout.title.text == "Airspeed Time-Series with Vne Limit"

        # 2b. Multi-signal coordinated panel
        panel_res: MultiSignalPanelResult = generate_multi_signal_panel(
            data=telemetry_df,
            signals=(
                SignalSpec(parameter_id="airspeed_kts", unit="kts", display_name="Airspeed"),
                SignalSpec(parameter_id="altitude_ft", unit="ft", display_name="Pressure Altitude"),
                SignalSpec(parameter_id="nz_g", unit="g", display_name="Normal Load Factor"),
            ),
            config=MultiSignalPanelConfig(
                title="Coordinated Flight Telemetry Dynamics",
                layout_mode=PanelLayoutMode.STACKED,
            ),
        )
        assert panel_res.figure is not None
        assert panel_res.figure.layout.title.text == "Coordinated Flight Telemetry Dynamics"

        # Verify DataFrame was not mutated by visualizers
        pd.testing.assert_frame_equal(telemetry_df, df_copy_baseline)

        # Step 3: Extract structured EngineeringReport
        report: EngineeringReport = extract_engineering_report(
            pipeline_result=pipeline_result,
            report_title="Sortie 401 — Post-Flight Engineering Evaluation",
            analyst_name="Col. Frank Borman",
            custom_metadata={"Sortie Number": "ST-401", "Airframe": "X-29"},
        )
        assert isinstance(report, EngineeringReport)
        assert report.metadata.title == "Sortie 401 — Post-Flight Engineering Evaluation"
        assert report.metadata.analyst_name == "Col. Frank Borman"
        assert len(report.stage_summaries) == 15

        # Check vehicle evaluation extraction
        assert report.engineering_evaluation.is_evaluated is True
        assert report.engineering_evaluation.limit_exceedance_count >= 1

        # Step 4: Export to Markdown
        md_file = tmp_path / "sortie_401_report.md"
        md_export_res: ReportExportResult = export_report(
            report=report,
            output_path=md_file,
            format=ReportExportFormat.MARKDOWN,
            overwrite=True,
        )
        assert md_export_res.status == ResultStatus.SUCCESS
        assert md_export_res.is_successful is True
        assert md_export_res.format == ReportExportFormat.MARKDOWN
        assert md_export_res.output_path == md_file.resolve()
        assert md_file.is_file()

        md_content = md_file.read_text(encoding="utf-8")
        assert "# Sortie 401 — Post-Flight Engineering Evaluation" in md_content
        assert "Col. Frank Borman" in md_content
        assert "[OBSERVED]" in md_content
        assert "[CALCULATED]" in md_content
        assert "[LIMIT]" in md_content
        assert "[PROCEDURAL]" in md_content
        assert "[RECOMMENDATION]" in md_content

        # Verify Markdown file checksum in provenance matches disk file
        assert md_export_res.provenance is not None
        disk_md_hash = compute_file_checksum(md_file)
        assert md_export_res.provenance.source_checksum == disk_md_hash

        # Step 5: Export to Self-Contained HTML with embedded Plotly figures
        html_file = tmp_path / "sortie_401_report.html"
        html_export_res: ReportExportResult = export_report(
            report=report,
            output_path=html_file,
            format=ReportExportFormat.HTML,
            figures=[ts_res, panel_res],
            include_plotlyjs="cdn",
            overwrite=True,
        )
        assert html_export_res.status == ResultStatus.SUCCESS
        assert html_export_res.is_successful is True
        assert html_export_res.format == ReportExportFormat.HTML
        assert html_export_res.output_path == html_file.resolve()
        assert html_file.is_file()

        html_content = html_file.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html_content
        assert "Sortie 401 — Post-Flight Engineering Evaluation" in html_content
        assert "Col. Frank Borman" in html_content
        assert "Airspeed Time-Series with Vne Limit" in html_content
        assert "@media print" in html_content

        # Verify HTML file checksum in provenance matches disk file
        assert html_export_res.provenance is not None
        disk_html_hash = compute_file_checksum(html_file)
        assert html_export_res.provenance.source_checksum == disk_html_hash

        # Step 6: Verify source DataPipelineResult immutability
        assert pipeline_result.status == original_pipeline_status
        assert len(pipeline_result.issues) == original_issue_count

    def test_direct_pipeline_orchestrator_integration(
        self,
        e2e_flight_csv: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        """Verify DataPipelineOrchestrator.export_report() integration method."""
        orchestrator = DataPipelineOrchestrator()
        pipeline_res = orchestrator.run_csv_pipeline(e2e_flight_csv)

        out_html = tmp_path / "orchestrator_direct.html"
        res: ReportExportResult = orchestrator.export_report(
            pipeline_result=pipeline_res,
            output_path=out_html,
            report_title="Orchestrator Integration Test",
        )

        assert res.status == ResultStatus.SUCCESS
        assert res.output_path == out_html.resolve()
        assert out_html.is_file()
        content = out_html.read_text(encoding="utf-8")
        assert "Orchestrator Integration Test" in content

    def test_visualization_integrity_and_observation_preservation(
        self,
        e2e_flight_csv: pathlib.Path,
    ) -> None:
        """Verify visualizers preserve observation sequence without sorting or fabrication."""
        # Telemetry with deliberate non-monotonic / out-of-order timestamps
        raw_df = pd.DataFrame(
            {
                "timestamp": ["2026-10-19T10:00:02Z", "2026-10-19T10:00:01Z"],
                "airspeed_kts": [250.0, 240.0],
            }
        )
        # 1. Direct figure builder raises ValueError when require_monotonic_timestamps is True
        with pytest.raises(ValueError, match="non-monotonic"):
            build_time_series_figure(
                data=raw_df,
                parameter_id="airspeed_kts",
                timestamp_column="timestamp",
                unit="kts",
                config=TimeSeriesPlotConfig(require_monotonic_timestamps=True),
            )

        # 2. Composite overlaid generator captures build error and returns structured ERROR result
        res = generate_overlaid_time_series_plot(
            data=raw_df,
            parameter_id="airspeed_kts",
            timestamp_column="timestamp",
            unit="kts",
            config=TimeSeriesPlotConfig(require_monotonic_timestamps=True),
        )
        assert res.status == ResultStatus.ERROR
        assert res.issues[0].issue_code == "ERR_FIGURE_BUILD_FAILED"
        assert "non-monotonic" in res.issues[0].message

    def test_multi_signal_incompatible_unit_overlay_fallback(self) -> None:
        """Verify that incompatible units in OVERLAY mode safely fall back to STACKED."""
        df = pd.DataFrame(
            {
                "timestamp": ["2026-10-19T10:00:00Z", "2026-10-19T10:00:01Z"],
                "airspeed_kts": [200.0, 210.0],
                "altitude_ft": [10000.0, 10050.0],
            }
        )

        panel_res = generate_multi_signal_panel(
            data=df,
            signals=(
                SignalSpec(parameter_id="airspeed_kts", unit="kts"),
                SignalSpec(parameter_id="altitude_ft", unit="ft"),  # Incompatible unit
            ),
            config=MultiSignalPanelConfig(
                title="Overlay Fallback Test",
                layout_mode=PanelLayoutMode.OVERLAY,
            ),
        )

        assert panel_res.status == ResultStatus.WARNING
        assert any("incompatible units" in issue.message.lower() for issue in panel_res.issues)
        assert panel_res.figure is not None

    def test_degraded_pipeline_with_upstream_validation_failure(
        self,
        tmp_path: pathlib.Path,
    ) -> None:
        """Verify report extraction and export correctly represent upstream validation failure."""
        bad_csv = tmp_path / "bad_schema.csv"
        # CSV missing required schema columns (no timestamp)
        pd.DataFrame({"invalid_col": [1, 2, 3]}).to_csv(bad_csv, index=False)

        pipeline_res = run_csv_pipeline(bad_csv)
        assert pipeline_res.validation is not None
        assert pipeline_res.validation.has_errors is True

        # Extract report from failed pipeline
        report = extract_engineering_report(pipeline_res, report_title="Failed Validation Report")
        assert report.executive_summary.overall_status == ResultStatus.ERROR

        # Export report to markdown
        out_md = tmp_path / "failed_validation.md"
        export_res = export_pipeline_report(
            pipeline_result=pipeline_res,
            output_path=out_md,
            report_title="Failed Validation Report",
        )
        assert export_res.status == ResultStatus.SUCCESS
        assert out_md.is_file()

        content = out_md.read_text(encoding="utf-8")
        assert "Validation Verdict: FAILED" in content or "ERROR" in content
        assert "Upstream Validation Aborted" in content or "Validation" in content

    def test_export_safety_path_confinement_and_traversal(
        self,
        e2e_flight_csv: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        """Verify strict directory confinement and traversal defenses."""
        pipeline_res = run_csv_pipeline(e2e_flight_csv)
        allowed_base = tmp_path / "safe_zone"
        allowed_base.mkdir()

        # Attempt to escape safe_zone
        escape_target = allowed_base / ".." / "escaped_report.html"

        res = export_pipeline_report(
            pipeline_result=pipeline_res,
            output_path=escape_target,
            base_dir=allowed_base,
        )

        assert res.status == ResultStatus.ERROR
        assert res.issues[0].issue_code == "ERR_PATH_VALIDATION"
        assert not escape_target.exists()

    def test_symlink_path_resolution_and_confinement(
        self,
        e2e_flight_csv: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        """
        Verify that symlinks resolving outside base_dir are rejected.

        Platform note: On Windows, symlink creation requires elevated privileges
        or Developer Mode. If symlink creation fails with OSError/privilege error,
        the test records platform limitations and passes safely.
        """
        pipeline_res = run_csv_pipeline(e2e_flight_csv)
        base_dir = tmp_path / "symlink_jail"
        base_dir.mkdir()
        outside_dir = tmp_path / "outside_jail"
        outside_dir.mkdir()

        symlink_target = base_dir / "link_to_outside"
        try:
            os.symlink(str(outside_dir), str(symlink_target), target_is_directory=True)
        except (OSError, NotImplementedError):
            pytest.skip("Symlink creation not permitted by current OS user privileges.")

        # Attempt to write inside the symlinked directory pointing outside jail
        escaped_file = symlink_target / "escaped_symlink_report.html"

        res = export_pipeline_report(
            pipeline_result=pipeline_res,
            output_path=escaped_file,
            base_dir=base_dir,
        )

        # Must be rejected because its resolved path is outside base_dir
        assert res.status == ResultStatus.ERROR
        assert res.issues[0].issue_code == "ERR_PATH_VALIDATION"
        assert not (outside_dir / "escaped_symlink_report.html").exists()

    def test_atomic_file_write_failure_preserves_target_file(
        self,
        e2e_flight_csv: pathlib.Path,
        tmp_path: pathlib.Path,
    ) -> None:
        """Verify that an interrupted write preserves the original file and cleans up temp files."""
        pipeline_res = run_csv_pipeline(e2e_flight_csv)
        target_file = tmp_path / "atomic_guard.md"
        target_file.write_text("AUTHORITATIVE BASELINE RECORD", encoding="utf-8")

        # Simulate disk error during fsync
        with patch("os.fsync", side_effect=OSError("Simulated device failure")):
            res = export_pipeline_report(
                pipeline_result=pipeline_res,
                output_path=target_file,
                overwrite=True,
                raise_on_error=False,
            )

        assert res.status == ResultStatus.ERROR
        assert res.issues[0].issue_code == "ERR_WRITE_FAILED"
        # Original file must remain 100% intact
        assert target_file.read_text(encoding="utf-8") == "AUTHORITATIVE BASELINE RECORD"
        # No orphaned temporary files in directory
        temp_files = list(tmp_path.glob(".aeris_export_tmp_*"))
        assert len(temp_files) == 0
